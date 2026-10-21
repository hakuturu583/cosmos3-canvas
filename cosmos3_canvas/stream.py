"""Cosmos3-Nano-Sim-Auto one chunk at a time, with the controls handed over per chunk.

cosmos_framework's causal multiview inference reads every control from files
before it generates anything. Its rolling sampler (``iter_rolling_transfer_ar``)
is incremental, though: chunk k only reads the control latents of chunk k, at
the time chunk k is generated. This module drives it as a stream:

* the model, the plans, the text tokens and the caption schedule are built by
  framework code from a one-chunk bootstrap request (black controls), for a
  horizon given up front (the schedule and the RoPE need one);
* the control and target latents are then swapped for buffers of the whole
  horizon, zero until written;
* :meth:`Cosmos3Stream.step` encodes one chunk's camera control frames and
  LiDAR control sweeps with the VAEs' causal streaming encoders, writes them in
  the buffers, advances the sampler by one chunk and decodes the chunk with the
  streaming decoders.

Chunks follow the model's teacher-forcing clock (``joint_ar_chunks``): chunk 0
is camera frame 0 and LiDAR sweep 0 (t = 0); chunk k >= 1 is camera frames
12k-11 .. 12k (30 fps) and sweeps 4k-3 .. 4k (10 Hz), i.e. (0.4(k-1), 0.4k].
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

CAMERA_FPS = 30
LIDAR_FPS = 10
FRAMES_PER_CHUNK = 12  # camera frames of a chunk after the first
SWEEPS_PER_CHUNK = 4
CHUNK_SECONDS = 0.4
CONTROL_SIZE = (720, 1280)  # WSM control frames (H, W)
LIDAR_CONTROL_SHAPE = (128, 1800)


def chunk_camera_frames(k: int) -> range:
    return range(0, 1) if k == 0 else range(12 * k - 11, 12 * k + 1)


def chunk_lidar_sweeps(k: int) -> range:
    return range(0, 1) if k == 0 else range(4 * k - 3, 4 * k + 1)


def chunk_of_time(t: float) -> int:
    """The chunk generating sim time ``t`` (seconds since the stream's start)."""
    return 0 if t <= 1e-6 else math.ceil(t / CHUNK_SECONDS - 1e-6)


@dataclass
class Chunk:
    index: int
    camera_frames: range  # stream frame indexes (30 fps)
    lidar_sweeps: range  # stream sweep indexes (10 Hz)
    rgb: torch.Tensor  # [V,F,H,W,3] uint8, CPU
    range_m: torch.Tensor  # [S,128,1800] float32 metres, 0: no return
    intensity: torch.Tensor  # [S,128,1800] float32 [0,1]
    seconds: dict[str, float]


def _init_single_process() -> None:
    for key, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"),
                       ("MASTER_ADDR", "127.0.0.1")):
        os.environ.setdefault(key, value)
    if "MASTER_PORT" not in os.environ:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])


class Cosmos3Stream:
    """One rollout of Cosmos3-Nano-Sim-Auto (cameras + LiDAR), stepped a chunk at a time.

    ``cameras``: Hyperion 8 camera keys (``camera_front_wide_120fov``, ...), in
    the order of the control and output tensors. ``captions``: one per camera.
    ``horizon_seconds``: the longest rollout; :meth:`step` refuses chunks past it.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        cameras: Sequence[str],
        captions: Sequence[str],
        *,
        horizon_seconds: float,
        workdir: str | Path,
        seed: int = 0,
        resolution: str = "480",
        extra_setup_args: Sequence[str] = (),
    ) -> None:
        self.cameras = list(cameras)
        self.num_views = len(self.cameras)
        if len(captions) != self.num_views:
            raise ValueError("one caption per camera")
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.num_chunks = chunk_of_time(horizon_seconds) + 1
        self.horizon_frames = 12 * (self.num_chunks - 1) + 1  # camera frames
        self.horizon_sweeps = 4 * (self.num_chunks - 1) + 1
        self._load(Path(checkpoint), list(captions), seed, resolution, list(extra_setup_args))
        self._reset_state()

    # -- setup ----------------------------------------------------------------

    def _bootstrap_request(self, captions: list[str], seed: int, resolution: str) -> Path:
        """A two-chunk request (13 frames, 4 sweeps) with black controls, read by framework code."""
        from safetensors.torch import save_file
        import torchvision.io

        boot = self.workdir / "bootstrap"
        boot.mkdir(exist_ok=True)
        frames = 13
        video = boot / "control.mp4"
        torchvision.io.write_video(str(video), torch.zeros(frames, *CONTROL_SIZE, 3, dtype=torch.uint8), CAMERA_FPS,
                                   options={"crf": "18"})
        lidar = boot / "lidar_control.safetensors"
        save_file({"frames": torch.zeros(3, 4, *LIDAR_CONTROL_SHAPE)}, str(lidar),
                  metadata={"source": json.dumps({"clip": "bootstrap", "camera_frame_ids": [0, 3, 6, 9],
                                                  "kind": "geometry_hdmap_control", "range_units": "metres"})})
        views = []
        for camera, caption in zip(self.cameras, captions, strict=True):
            path = boot / f"caption_{camera}.json"
            key = f"chunk_0_{self.horizon_frames}"
            path.write_text(json.dumps({"caption_structured": {key: {
                "caption": caption, "start_frame": 0, "end_frame": self.horizon_frames}}}))
            views.append({"camera_key": camera, "control_path": str(video), "caption_path": str(path)})
        request = {
            "name": "stream", "model_mode": "text2video", "resolution": resolution, "aspect_ratio": "16,9",
            "num_frames": frames, "fps": CAMERA_FPS, "seed": seed, "prompt": "", "wsm": {},
            "num_conditional_frames": 0, "condition_frame_indexes_vision": [],
            "lidar": {"control_path": str(lidar)},
            "multiview": {"condition_video_as_image": False, "load_caption_from_data": True, "views": views},
        }
        path = boot / "request.json"
        path.write_text(json.dumps(request, indent=1))
        return path

    def _load(self, checkpoint: Path, captions: list[str], seed: int, resolution: str, extra: list[str]) -> None:
        _init_single_process()
        from cosmos_framework.inference.common.init import init_script

        init_script()
        import tyro

        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.inference.multiview import _get_multiview_control_pixel_frames
        from cosmos_framework.inference.multiview_causal import (
            load_caption_chunks,
            per_view_prompt_data,
            prepare_joint_batch,
            prepare_rolling_prompt_schedule,
        )
        from cosmos_framework.model.generator.omni_mot_causal_model import _per_view_caption_groups
        from cosmos_framework.data.generator.sequence_packing import build_sequence_plans_from_data_batch
        from cosmos_framework.scripts.inference import InferenceArgs

        request = self._bootstrap_request(captions, seed, resolution)
        argv = ["-i", str(request), "-o", str(self.workdir / "out"), "--checkpoint-path", str(checkpoint),
                "--cp-size", "1", "--cfgp-size", "1", "--no-use-torch-compile", "--no-diffusion-cache",
                "--no-guardrails", *extra]
        args = tyro.cli(InferenceArgs, args=argv, config=(tyro.conf.OmitArgPrefixes, tyro.conf.CascadeSubcommandArgs,
                                                          tyro.conf.OmitSubcommandPrefixes))
        setup = args.setup.build_setup()
        init_output_dir(setup.output_dir)
        overrides = setup.get_sample_overrides_cls().from_files([request], overrides=setup.sample_overrides)[0]
        overrides.output_dir = self.workdir / "out" / "stream"
        self.pipe = pipe = setup.get_inference_cls().create(setup)
        sample = overrides.build_sample(model_config=pipe.model_config)
        self.sample = sample
        self.model = model = pipe.model
        height_width = sample.vision_size  # (W, H)
        self.size = (height_width[1], height_width[0])
        with torch.no_grad():
            # The captions as the CLI reads them (load_caption_from_data): its loader prefixes each
            # camera's caption with the rig's description, which the model was trained with.
            horizon = sample.model_copy(update={"num_frames": self.horizon_frames})
            chunks = load_caption_chunks(sample.multiview.views, first_only=sample.multiview.use_first_chunk_only,
                                         first_caption_paragraph_only=sample.multiview.first_caption_paragraph_only)
            first = chunks[0]
            prompts = (first.view_prompts if pipe.per_view_captions and first.view_prompts is not None
                       else [first.prompt] * self.num_views)
            controls = _get_multiview_control_pixel_frames(sample, height=self.size[0], width=self.size[1])
            data = prepare_joint_batch(model, sample, controls, [None] * self.num_views, prompts)
            data.update(per_view_prompt_data(sample, model, prompts))
            schedule, self.prompt_metadata = prepare_rolling_prompt_schedule(
                horizon, model, chunks, per_view_captions=pipe.per_view_captions, system_prompt=data["system_prompt"])
            # What generate_joint_samples_from_batch_autoregressive does before sampling.
            plans = build_sequence_plans_from_data_batch(data_batch=data, input_video_key=model.input_video_key,
                                                         input_image_key=model.input_image_key)
            groups = _per_view_caption_groups(data[model.input_caption_key])
            model._apply_inference_caption_plan(plans, groups)
            clean = model.get_data_and_condition(data, vision_condition_indexes=[plans[0].condition_frame_indexes_vision],
                                                 retain_raw_state_vision=False)
            has_negative = "neg_" + model.input_caption_key in data
            self._text = model._get_inference_text_tokens(data, has_negative, groups)
        self._plans, self._schedule = plans, schedule
        self._bootstrap = clean
        self._seed = sample.seed if sample.seed is not None else seed
        self._sampler_mode = "distilled" if model.config.fixed_step_sampler_config is not None else "rf"
        from cosmos_framework.inference.lidar_transfer import lidar_range_projection

        self._projection = lidar_range_projection(model)
        self.lidar_tokenizer = model.tokenizer_lidar_gen
        self.vae = model.tokenizer_vision_gen.model  # WanVAE: .model is WanVAE_ (the caches)

    def _reset_state(self) -> None:
        clean = self._bootstrap
        assert clean.x0_tokens_vision is not None and clean.x0_tokens_lidar is not None
        v, l = clean.x0_tokens_vision[0], clean.x0_tokens_lidar[0]  # [1,Cv,V*T0,H,W], [1,Cl,S0,H,W]
        latents = (self.horizon_frames - 1) // 4 + 1
        vision = v.new_zeros((1, v.shape[1], self.num_views * latents, *v.shape[-2:]))
        lidar = l.new_zeros((1, l.shape[1], self.horizon_sweeps, *l.shape[-2:]))
        self._latents_per_view = latents
        self.data = replace(clean, x0_tokens_vision=[vision, vision.clone()], x0_tokens_lidar=[lidar, lidar.clone()])
        self._enc_caches: list[tuple[Any, Any] | None] = [None] * self.num_views
        self._dec_caches: list[Any] = [None] * self.num_views
        self._lidar_enc_cache = None
        self._lidar_dec_cache = None
        self.next_chunk = 0
        self._iterator: Iterator[dict[str, torch.Tensor]] | None = None

    def reset(self) -> None:
        """Start a new rollout (same model, captions and horizon)."""
        self._reset_state()

    # -- encoders ---------------------------------------------------------------

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        """[F,H,W,3] uint8 control frames -> [3,F,h,w] in [-1,1], as the framework's video loader."""
        from cosmos_framework.inference.vision import _resize_and_center_crop

        x = frames.permute(0, 3, 1, 2).float()  # [F,3,H,W]
        # The loader rounds the resized frames to uint8 before the model normalizes them.
        x = _resize_and_center_crop(x, *self.size).round().clamp(0, 255) / 127.5 - 1.0
        return x.permute(1, 0, 2, 3)

    def _encode_camera(self, view: int, frames: torch.Tensor) -> torch.Tensor:  # -> [1,Cv,T,h,w]
        inner = self.vae.model
        saved = self._enc_caches[view]
        if saved is None:
            inner.clear_encoder_cache()
        else:
            inner._enc_cache, inner._enc_stream_shape = saved
        device = self.model.tensor_kwargs["device"]
        x = self._resize(frames).unsqueeze(0).to(device)  # [1,3,F,h,w]
        out = []
        if saved is None:  # the prime: one frame, then multiples of four
            out.append(self.vae.encode_streaming(x[:, :, :1]))
            x = x[:, :, 1:]
        if x.shape[2]:
            out.append(self.vae.encode_streaming(x))
        self._enc_caches[view] = (inner._enc_cache, inner._enc_stream_shape)
        return torch.cat(out, dim=2).float()

    def _encode_lidar(self, sweeps: torch.Tensor) -> torch.Tensor:  # [3,S,128,1800] metric -> [1,Cl,S,h,w]
        from cosmos_framework.inference.lidar_transfer import _conform_lidar_control_width
        from cosmos_framework.model.generator.tokenizers.lidar.normalization import metric_lidar_to_network

        tok = self.lidar_tokenizer
        clip = _conform_lidar_control_width(sweeps.float(), range_projection=self._projection, control_path="stream")
        state = clip.unsqueeze(0).to(device=tok.device, dtype=tok.dtype)
        normalized, _ = metric_lidar_to_network(state, min_range=tok.min_range, max_range=tok.max_range)
        net = tok.model
        cache = net._limit_streaming_cache(self._lidar_enc_cache, tok.streaming_context_frames,
                                           new_frames=normalized.shape[2])
        sample, (mu, _), self._lidar_enc_cache = net.encode_stream(normalized, cache)
        latent = sample if tok.sample_posterior else mu
        return ((latent.to(dtype=tok.dtype) - tok.latent_mean) / tok.latent_std).float()

    # -- decoders ---------------------------------------------------------------

    def _decode_camera(self, view: int, latent: torch.Tensor) -> torch.Tensor:  # [1,Cv,T,h,w] -> [F,h,w,3] uint8
        inner = self.vae.model
        inner._dec_cache = self._dec_caches[view] if self._dec_caches[view] is not None else inner._new_dec_cache()
        video = self.vae.decode(latent, clear_decoder_cache=False)  # [1,3,F,H,W] in [-1,1]
        self._dec_caches[view] = inner._dec_cache
        video = ((1 + video.float()) / 2).clamp(0, 1)
        return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()

    def _decode_lidar(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from cosmos_framework.inference.lidar_transfer import (
            DEFAULT_RANGE_SMOOTHING,
            smooth_metric_range,
            undo_model_width_transform,
            validity_mask,
        )
        from cosmos_framework.model.generator.tokenizers.lidar.normalization import network_lidar_to_metric_clip
        from cosmos_framework.model.generator.tokenizers.lidar.postprocessing import resolved_validity_channel

        tok = self.lidar_tokenizer
        net = tok.model
        z = latent.to(device=tok.device, dtype=tok.dtype) * tok.latent_std + tok.latent_mean
        cache = net._limit_streaming_cache(self._lidar_dec_cache, tok.streaming_context_frames, new_frames=z.shape[2])
        output, self._lidar_dec_cache = net.decode_stream(z, cache, return_validity=False)
        output, validity = tok._split_decode_output(output)
        threshold = tok._resolve_validity_threshold(None)
        output = output.clone()  # as decode_streaming_normalized(apply_mask=False)
        output[:, 2:3] = resolved_validity_channel(validity, should_mask=False, threshold=threshold)
        metric = network_lidar_to_metric_clip(output, validity, min_range=tok.min_range, max_range=tok.max_range,
                                              apply_validity_mask=tok.apply_validity_mask, validity_threshold=threshold)
        # What save_generated_lidar does with a decoded clip.
        projection = self._projection
        semantic = undo_model_width_transform(metric[0], range_projection=projection)  # [3,S,H,W]
        range_m, intensity = semantic[0].float(), semantic[1].float()
        valid = validity_mask(semantic[2].float(), threshold=projection.validity_threshold) & torch.isfinite(range_m)
        valid &= (range_m >= projection.min_range_m) & (range_m <= projection.max_range_m)
        if DEFAULT_RANGE_SMOOTHING is not None:
            range_m, valid = smooth_metric_range(range_m, valid, smoothing=DEFAULT_RANGE_SMOOTHING)
        range_m = range_m.clamp(min=projection.min_range_m, max=projection.max_range_m)
        range_m = torch.where(valid, range_m, torch.zeros_like(range_m)).cpu()
        intensity = torch.where(valid & torch.isfinite(intensity), intensity.clamp(0, 1), torch.zeros_like(intensity)).cpu()
        return range_m, intensity

    # -- stepping ---------------------------------------------------------------

    def _sampler(self) -> Iterator[dict[str, torch.Tensor]]:
        from cosmos_framework.model.generator.utils.rolling_kv.rolling_transfer_ar import iter_rolling_transfer_ar

        sample = self.sample
        conditional, unconditional = self._text
        return iter_rolling_transfer_ar(
            self.model, plans=self._plans, data=self.data, conditional_text=conditional,
            unconditional_text=unconditional, guidance=sample.guidance, seed=self._seed, num_steps=sample.num_steps,
            shift=sample.shift, sampler_mode=self._sampler_mode, distilled_num_steps=None,
            prompt_schedule=self._schedule,
        )

    @torch.no_grad()
    def step(self, camera_controls: torch.Tensor, lidar_controls: torch.Tensor) -> Chunk:
        """Generate chunk :attr:`next_chunk`.

        ``camera_controls``: [V,F,720,1280,3] uint8 WSM control frames of the
        chunk's camera frames (F = 1 for chunk 0, else 12). ``lidar_controls``:
        [3,S,128,1800] float32 LiDAR controls (range m, hit, valid) of its sweeps
        (S = 1, else 4).
        """
        k = self.next_chunk
        if k >= self.num_chunks:
            raise RuntimeError(f"the stream's horizon is {self.num_chunks} chunks")
        frames, sweeps = chunk_camera_frames(k), chunk_lidar_sweeps(k)
        if camera_controls.shape[:2] != (self.num_views, len(frames)):
            raise ValueError(f"chunk {k} needs [{self.num_views},{len(frames)},H,W,3] camera controls")
        if lidar_controls.shape[:2] != (3, len(sweeps)):
            raise ValueError(f"chunk {k} needs [3,{len(sweeps)},128,1800] LiDAR controls")
        timing: dict[str, float] = {}
        sync = torch.cuda.synchronize
        t0 = time.perf_counter()
        control_v, control_l = self.data.x0_tokens_vision[0], self.data.x0_tokens_lidar[0]
        latent_frames = range(0, 1) if k == 0 else range(3 * k - 2, 3 * k + 1)
        for view in range(self.num_views):
            latent = self._encode_camera(view, camera_controls[view])
            index = [view * self._latents_per_view + f for f in latent_frames]
            control_v[:, :, index] = latent.to(control_v.dtype)
        control_l[:, :, sweeps.start:sweeps.stop] = self._encode_lidar(lidar_controls).to(control_l.dtype)
        sync()
        timing["encode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if self._iterator is None:
            self._iterator = self._sampler()
        part = next(self._iterator)  # vision [1,Cv,V*Tc,h,w], lidar [1,Cl,Sc,h,w]
        sync()
        timing["generate"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vision = part["vision"].reshape(1, part["vision"].shape[1], self.num_views, -1, *part["vision"].shape[-2:])
        rgb = torch.stack([self._decode_camera(view, vision[:, :, view]) for view in range(self.num_views)])
        range_m, intensity = self._decode_lidar(part["lidar"])
        sync()
        timing["decode"] = time.perf_counter() - t0
        self.next_chunk += 1
        return Chunk(k, frames, sweeps, rgb, range_m, intensity, timing)

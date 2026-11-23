"""Cosmos3-Nano-Sim-Auto through FlashDreams, one chunk at a time.

FlashDreams (https://github.com/NVIDIA/flashdreams) is NVIDIA's
runtime for the model, about 3x faster than cosmos-framework's rolling sampler
at 6-7 cameras on one RTX PRO 6000. Its ``ChunkRollout`` already takes the
camera controls a chunk at a time from a provider (``RenderedControls``), so
this stream hands it the WSM control frames the renderer draws (controls.py),
shrunk to the generated size the way FlashDreams' own map renderer feeds them,
and decodes each chunk with its streaming TAEHV decoder.

Same interface and chunk clock as :class:`cosmos3_canvas.stream.Cosmos3Stream`:
chunk 0 is camera frame 0, chunk k >= 1 camera frames 12k-11 .. 12k at 30 fps.

Differences:

* cameras only: the LiDAR is not generated (``lidar`` is False);
* the rollout has no end: the cache keeps a bounded history, so
  ``horizon_seconds`` only sets the duration the captions state;
* the first chunk needs the next chunk's controls too. The rollout's prefill
  reads the control one chunk past the history it keeps (frames 0..12 for
  chunk 0), so :meth:`step` takes them as ``next_camera_controls`` at chunk 0
  and chunk 1 then takes ``None``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from .stream import CAMERA_FPS, Chunk, chunk_camera_frames, chunk_lidar_sweeps, chunk_of_time

#: no end: the renderer's horizon checks never trigger
UNBOUNDED_CHUNKS = 1 << 40


def shrink_controls(frames: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """``[N, H, W, 3]`` uint8 control frames -> ``[N, h, w, 3]`` uint8.

    Bilinear without antialiasing, as FlashDreams' map renderer shrinks its own
    drawings (``ludus_controls._shrunk``): what the model is fed there.
    """
    height, width = size_hw
    if tuple(frames.shape[1:3]) == (height, width):
        return frames[..., :3].contiguous()
    planes = frames[..., :3].permute(0, 3, 1, 2).to(torch.float32)
    smaller = F.interpolate(planes, size=(height, width), mode="bilinear", align_corners=False)
    return smaller.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class FlashDreamsStream:
    """One rollout of Cosmos3-Nano-Sim-Auto's cameras with FlashDreams, stepped a chunk at a time.

    ``checkpoint``: the HF export (a directory with ``model.safetensors.index.json``,
    the index itself, or its URL). ``cameras``: Hyperion 8 camera keys, in the
    order of the control and output tensors; ``captions``: one per camera.
    ``compile_denoise``: FlashDreams' ``torch.compile`` mode for the denoiser
    (``"default"``), off when ``None``.
    """

    lidar = False
    #: chunks of controls past its own the first chunk needs (see the module docstring)
    lookahead = 1

    def __init__(
        self,
        checkpoint: str | Path,
        cameras: Sequence[str],
        captions: Sequence[str],
        *,
        horizon_seconds: float,
        seed: int = 0,
        resolution: str = "480",
        compile_denoise: str | None = None,
        denoise_steps: int | None = None,
        device: str = "cuda",
    ) -> None:
        self.cameras = list(cameras)
        self.num_views = len(self.cameras)
        if len(captions) != self.num_views:
            raise ValueError("one caption per camera")
        self.num_chunks = UNBOUNDED_CHUNKS
        self.seed = int(seed)
        self.device = device
        self.compile_denoise = compile_denoise
        self._load(str(checkpoint), list(captions), resolution, horizon_seconds, denoise_steps)
        self._reset_state()

    # -- setup ----------------------------------------------------------------

    def _load(self, checkpoint: str, captions: list[str], resolution: str, horizon_seconds: float,
              denoise_steps: int | None) -> None:
        from cosmos3_nano_sim_auto.checkpoint import load_exported_checkpoint
        from cosmos3_nano_sim_auto.config import (DEFAULT_ATTENTION_SCOPE, DEFAULT_CHECKPOINT, DEFAULT_DENOISE_STEPS,
                                                  DEFAULT_HISTORY_FRAMES, DEFAULT_TEMPORAL_WINDOW_SECONDS,
                                                  denoise_schedule)
        from cosmos3_nano_sim_auto.impl.packing import DEFAULT_CHUNK_FRAMES
        from cosmos3_nano_sim_auto.impl.transformer.attention import attention_implementation
        from cosmos3_nano_sim_auto.inputs import MultiviewSample, prompt_token_ids
        from cosmos_interactive.checkpoint_load import resolve_checkpoint_source

        source = resolve_checkpoint_source(str(Path(checkpoint).expanduser()) if "://" not in checkpoint
                                           else checkpoint, default=DEFAULT_CHECKPOINT)
        self.network = network = load_exported_checkpoint(source, device=self.device).eval()
        spec = network.config.spec
        self.sample = sample = MultiviewSample(
            name="carla", model_mode="text2video", prompt=captions[0], resolution=resolution,
            cameras=tuple(self.cameras), control_paths=(), view_captions=tuple(captions))
        width, height = sample.view_size
        self.size = (height, width)
        # Three latent frames (12 camera frames, 0.4 s) per chunk, as the rig was trained: the
        # shared spec's ar_chunk_latent_frames (4) is the single-view weights'.
        self.geometry = sample.geometry(pixel_frames=None, latent_patch_size=spec.vfm.latent_patch_size,
                                        frames_per_chunk=DEFAULT_CHUNK_FRAMES, fps=float(CAMERA_FPS))
        horizon_frames = 12 * chunk_of_time(horizon_seconds) + 1
        self.prompt = prompt_token_ids(sample, pixel_frames=horizon_frames, fps=float(CAMERA_FPS), lidar=False)
        # What the checkpoint was trained with (cosmos3_nano_sim_auto.config): run at anything
        # else, the picture degrades without an error.
        self._rollout_args = dict(
            fps=float(CAMERA_FPS), seed=self.seed, attention_impl=attention_implementation(flex=False),
            accelerated_rms_norm=True, history_frames=DEFAULT_HISTORY_FRAMES, token_frames=DEFAULT_HISTORY_FRAMES,
            compile_denoise=self.compile_denoise,
            t_list=denoise_schedule(spec.vfm.fixed_step_t_list, denoise_steps or DEFAULT_DENOISE_STEPS),
            attention_scope=DEFAULT_ATTENTION_SCOPE, maskless_attention=True, fused_attention_merge=True,
            decomposed_temporal_window_seconds=DEFAULT_TEMPORAL_WINDOW_SECONDS)
        self._latent_patch_size = spec.vfm.latent_patch_size

    def _reset_state(self) -> None:
        self._rollout = None
        self._decode = None
        self._decoded_to = 0
        self._pixels: dict[int, torch.Tensor] = {}  # pixel frame -> [V,h,w,3] uint8 controls, not yet encoded
        self.next_chunk = 0

    def reset(self) -> None:
        """Start a new rollout (same model, captions)."""
        self._reset_state()

    # -- controls ---------------------------------------------------------------

    def _put(self, frames: range, camera_controls: torch.Tensor) -> None:
        if camera_controls.shape[:2] != (self.num_views, len(frames)):
            raise ValueError(f"frames {frames.start}..{frames.stop - 1} need "
                             f"[{self.num_views},{len(frames)},H,W,3] camera controls")
        shrunk = shrink_controls(camera_controls.reshape(-1, *camera_controls.shape[2:]).to(self.device), self.size)
        shrunk = shrunk.reshape(self.num_views, len(frames), *shrunk.shape[1:])
        for i, f in enumerate(frames):
            self._pixels[f] = shrunk[:, i]

    def _render(self, first: int, last: int, views: Sequence[int] | None = None) -> torch.Tensor:
        """FlashDreams' ``RigRenderFn``: ``[V*T, h, w, 3]``, camera-outer, frame-inner."""
        missing = [f for f in range(first, last) if f not in self._pixels]
        if missing:
            raise RuntimeError(f"the rollout asked for the controls of frames {missing[0]}..{missing[-1]}, "
                               "which were not handed over")
        frames = torch.stack([self._pixels.pop(f) for f in range(first, last)], dim=1)  # [V,T,h,w,3]
        if views is not None:
            frames = frames[list(views)]
        return frames.reshape(-1, *frames.shape[2:])

    def _build(self) -> None:
        from cosmos3_nano_sim_auto.impl.conditioning.control_provider import RenderedControls
        from cosmos3_nano_sim_auto.impl.decode import make_multiview_chunk_decoder
        from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout
        from cosmos3_nano_sim_auto.inputs import make_control_encode_streams

        network = self.network
        dtype = network.config.dtype
        controls = RenderedControls(self._render,
                                    make_control_encode_streams(self.num_views, dtype=dtype, device=self.device),
                                    geometry=self.geometry, latent_patch_size=self._latent_patch_size)
        self._rollout = ChunkRollout(network, geometry=self.geometry, control_tokens=controls,
                                     text_ids=self.prompt.text_ids, view_text_tokens=self.prompt.view_text_tokens,
                                     **self._rollout_args)
        self._decode = make_multiview_chunk_decoder(num_views=self.num_views, dtype=dtype, device=self.device)

    # -- stepping ---------------------------------------------------------------

    @torch.no_grad()
    def step(self, camera_controls: torch.Tensor | None, lidar_controls: torch.Tensor | None = None,
             next_camera_controls: torch.Tensor | None = None) -> Chunk:
        """Generate chunk :attr:`next_chunk`.

        ``camera_controls``: [V,F,H,W,3] uint8 WSM control frames of the chunk's
        camera frames (F = 1 for chunk 0, else 12), at any size of the generated
        aspect ratio (the renderer's are 1280x720); ``None`` at chunk 1, whose
        controls came with chunk 0. ``next_camera_controls``: the next chunk's,
        at chunk 0 only. ``lidar_controls`` is ignored (no LiDAR).
        """
        k = self.next_chunk
        frames = chunk_camera_frames(k)
        if (k == 0) != (next_camera_controls is not None):
            raise ValueError("chunk 0, and only chunk 0, takes next_camera_controls (FlashDreamsStream.lookahead)")
        timing: dict[str, float] = {}
        sync = torch.cuda.synchronize
        t0 = time.perf_counter()
        if k == 0:
            self._put(frames, camera_controls)
            self._put(chunk_camera_frames(1), next_camera_controls)
        elif k > 1:
            self._put(frames, camera_controls)
        # chunk 1's came with chunk 0 and went into the prefill: its camera_controls may be None
        if self._rollout is None:
            self._build()  # the prefill encodes chunks 0 and 1's controls
        sync()
        timing["encode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        trace, _ = self._rollout.step()
        sync()
        timing["generate"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        from cosmos3_nano_sim_auto.impl.decode import decoded_views

        latent = self._rollout.latent_for(self._decoded_to, trace.end)
        self._decoded_to = trace.end
        pixels = decoded_views(latent, self._decode, parallel=None)  # [V,T,3,h,w] in [-1,1]
        rgb = ((pixels.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 1, 3, 4, 2).cpu()
        sync()
        timing["decode"] = time.perf_counter() - t0
        if rgb.shape[1] != len(frames):
            raise RuntimeError(f"chunk {k} decoded {rgb.shape[1]} frames, not {len(frames)}")
        self.next_chunk += 1
        sweeps = chunk_lidar_sweeps(k)
        empty = torch.zeros((len(sweeps), 0, 0))
        return Chunk(k, frames, sweeps, rgb, empty, empty, timing)

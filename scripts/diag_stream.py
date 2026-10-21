"""Where the stream's inputs differ from the CLI's batch inputs for the same request.

    diag_stream.py <checkpoint> <reference request.json> <out dir>

Builds the CLI's joint data for the reference request (framework code end to end)
and a Cosmos3Stream fed the same control frames, then compares the control
latents per chunk, the scheduled text tokens and the sequence plans.
"""

import json
import sys
from dataclasses import fields
from pathlib import Path

import torch


def main() -> None:
    checkpoint, request_path, out = sys.argv[1:4]
    request = json.loads(Path(request_path).read_text())
    view = request["multiview"]["views"][0]
    caption = json.loads(Path(view["caption_path"]).read_text())
    caption = next(iter(caption["caption_structured"].values()))["caption"]

    from cosmos3_canvas.stream import Cosmos3Stream, chunk_camera_frames, chunk_lidar_sweeps

    stream = Cosmos3Stream(checkpoint, [view["camera_key"]], [caption], horizon_seconds=2.8, workdir=Path(out) / "work")
    model, pipe = stream.model, stream.pipe

    # The CLI's data for the reference request (generate_rolling_sample + the joint entry).
    from cosmos_framework.inference.multiview_causal import load_caption_chunks
    from cosmos_framework.data.generator.sequence_packing import build_sequence_plans_from_data_batch
    from cosmos_framework.inference.multiview import _get_multiview_control_pixel_frames
    from cosmos_framework.inference.multiview_causal import (
        per_view_prompt_data,
        prepare_joint_batch,
        prepare_rolling_prompt_schedule,
    )
    from cosmos_framework.model.generator.omni_mot_causal_model import _per_view_caption_groups

    setup = stream.sample.model_copy(deep=True)
    overrides = pipe.setup_args.get_sample_overrides_cls().from_files([Path(request_path)],
                                                                      overrides=pipe.setup_args.sample_overrides)[0]
    overrides.output_dir = Path(out) / "ref"
    sample = overrides.build_sample(model_config=pipe.model_config)
    chunks = load_caption_chunks(sample.multiview.views, first_only=sample.multiview.use_first_chunk_only,
                                 first_caption_paragraph_only=sample.multiview.first_caption_paragraph_only)
    prompts = chunks[0].view_prompts if pipe.per_view_captions and chunks[0].view_prompts else [chunks[0].prompt]
    print("ref prompts", prompts, "stream captions", [caption], "per_view", pipe.per_view_captions)
    with torch.no_grad():
        controls = _get_multiview_control_pixel_frames(sample, height=stream.size[0], width=stream.size[1])
        data = prepare_joint_batch(model, sample, controls, [None], prompts)
        data.update(per_view_prompt_data(sample, model, prompts))
        schedule, meta = prepare_rolling_prompt_schedule(sample, model, chunks, per_view_captions=pipe.per_view_captions,
                                                         system_prompt=data["system_prompt"])
        plans = build_sequence_plans_from_data_batch(data_batch=data, input_video_key=model.input_video_key,
                                                     input_image_key=model.input_image_key)
        groups = _per_view_caption_groups(data[model.input_caption_key])
        model._apply_inference_caption_plan(plans, groups)
        clean = model.get_data_and_condition(data, vision_condition_indexes=[plans[0].condition_frame_indexes_vision],
                                             retain_raw_state_vision=False)

    print("schedule tokens equal:", schedule.views == stream._schedule.views,
          "fps", schedule.fps, stream._schedule.fps, "frames", schedule.num_frames, stream._schedule.num_frames)
    if schedule.views != stream._schedule.views:
        a, b = schedule.views[0][0].token_ids, stream._schedule.views[0][0].token_ids
        print("  lengths", len(a), len(b), "first diff", next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
        print("  ref meta prompt:", meta[0]["prompt"][:400])
    for f in fields(plans[0]):
        x, y = getattr(plans[0], f.name), getattr(stream._plans[0], f.name)
        if x != y:
            print(f"plan.{f.name}: ref={str(x)[:200]} stream={str(y)[:200]}")
    for f in fields(clean):
        if f.name.startswith(("x0_tokens", "raw_state")):
            continue
        x, y = getattr(clean, f.name), getattr(stream.data, f.name)
        same = torch.equal(x, y) if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape else x == y
        if not (same if isinstance(same, bool) else bool(same)):
            print(f"data.{f.name}: ref={str(x)[:200]} stream={str(y)[:200]}")
    print("ref latents", [tuple(t.shape) for t in clean.x0_tokens_vision], [tuple(t.shape) for t in clean.x0_tokens_lidar])

    # The stream's control latents for the same frames, chunk by chunk.
    from torchcodec.decoders import VideoDecoder
    from safetensors.torch import load_file

    decoder = VideoDecoder(view["control_path"], device="cpu", dimension_order="NHWC")
    frames = decoder.get_frames_in_range(0, len(decoder)).data
    lidar = load_file(request["lidar"]["control_path"])["frames"]
    ref_v, ref_l = clean.x0_tokens_vision[0], clean.x0_tokens_lidar[0]
    for k in range(4):
        fr, sw = chunk_camera_frames(k), chunk_lidar_sweeps(k)
        v = stream._encode_camera(0, frames[fr.start:fr.stop])
        lat = range(0, 1) if k == 0 else range(3 * k - 2, 3 * k + 1)
        rv = ref_v[:, :, lat.start:lat.stop].float()
        l = stream._encode_lidar(lidar[:, sw.start:sw.stop])
        rl = ref_l[:, :, sw.start:sw.stop].float()
        print(f"chunk {k}: camera latent rel err {float((v - rv).norm() / rv.norm()):.4f}, "
              f"lidar latent rel err {float((l - rl).norm() / rl.norm()):.4f}")
    print("target latents nonzero (ref):", float(clean.x0_tokens_vision[1].abs().max()), float(clean.x0_tokens_lidar[1].abs().max()))
    print("sample seed/guidance/steps/shift", sample.seed, sample.guidance, sample.num_steps, sample.shift,
          "| stream", stream._seed, stream.sample.guidance, stream.sample.num_steps, stream.sample.shift)


if __name__ == "__main__":
    main()

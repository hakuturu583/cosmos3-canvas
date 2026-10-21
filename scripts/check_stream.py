"""Stream a pre-rendered scene through Cosmos3Stream, chunk by chunk, and compare with the CLI's batch output.

    check_stream.py <checkpoint> <control.mp4> <lidar_control.safetensors> <caption> <reference dir> <out dir>

The reference is the framework CLI run on the same controls with num_frames = 85
(8 chunks). Its last chunk has 3 sweeps (round(85/3) = 28), the stream's 4, so the
first 7 chunks (frames 0..72, sweeps 0..24) are compared.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cosmos3_canvas.stream import Cosmos3Stream, chunk_camera_frames, chunk_lidar_sweeps


def main() -> None:
    checkpoint, video, lidar_path, caption, reference, out = sys.argv[1:7]
    out = Path(out)
    from torchvision.io import read_video, write_video
    from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

    load_torchcodec()

    from torchcodec.decoders import VideoDecoder

    # The framework's control loader decodes with torchcodec; so does this check.
    decoder = VideoDecoder(video, device="cpu", dimension_order="NHWC")
    controls = decoder.get_frames_in_range(0, len(decoder)).data  # [F,720,1280,3] uint8
    lidar = load_file(lidar_path)["frames"]  # [3,S,128,1800]
    import os

    extra = os.environ.get("COSMOS3_SETUP_ARGS", "").split()
    stream = Cosmos3Stream(checkpoint, ["camera_front_wide_120fov"], [caption], horizon_seconds=2.8,
                           workdir=out / "work", extra_setup_args=extra)
    rgb, ranges, timings = [], [], []
    for k in range(stream.num_chunks):
        frames, sweeps = chunk_camera_frames(k), chunk_lidar_sweeps(k)
        chunk = stream.step(controls[frames.start:frames.stop].unsqueeze(0), lidar[:, sweeps.start:sweeps.stop])
        rgb.append(chunk.rgb[0])
        ranges.append(chunk.range_m)
        timings.append(chunk.seconds)
        print(k, {key: round(value, 2) for key, value in chunk.seconds.items()}, flush=True)
    rgb = torch.cat(rgb)
    ranges = torch.cat(ranges)
    write_video(str(out / "stream.mp4"), rgb, 30)
    save_file({"range_m": ranges.contiguous()}, str(out / "stream_lidar.safetensors"))

    ref_rgb = read_video(str(next(Path(reference).glob("vision_view00_*.mp4"))), pts_unit="sec",
                         output_format="THWC")[0]
    ref_range = load_file(str(Path(reference) / "lidar.safetensors"))["range_m"]
    n, s = 73, 25
    mse = ((rgb[:n].float() - ref_rgb[:n].float()) ** 2).mean(dim=(1, 2, 3))
    psnr = (10 * torch.log10(255**2 / mse.clamp(min=1e-6))).tolist()
    both = (ranges[:s] > 0) & (ref_range[:s] > 0)
    report = {
        "psnr_first_frames": [round(p, 1) for p in psnr[:13]],
        "psnr_every_12": [round(p, 1) for p in psnr[::12]],
        "lidar_valid_agreement": float(((ranges[:s] > 0) == (ref_range[:s] > 0)).float().mean()),
        "lidar_mean_abs_range_diff_m": float((ranges[:s] - ref_range[:s]).abs()[both].mean()),
        "seconds_per_chunk": timings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({key: value for key, value in report.items() if key != "seconds_per_chunk"}, indent=1))


if __name__ == "__main__":
    main()

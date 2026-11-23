"""Writes what a rollout was conditioned on and what it generated, frame by frame.

``renderer.options.record_dir`` turns it on. Each rollout gets a directory
``<record_dir>/rollout_<n>``, where two kinds of JPEG land for every stream
frame and configured camera:

* ``wsm_<camera>_<frame>.jpg``: the WSM control frame the model was given,
  shrunk to the generated size;
* ``rgb_<camera>_<frame>.jpg``: the frame the model generated, which the
  CARLA camera standing for it receives.

and, once per rollout and tick, ``meta.json`` (the map, each camera's lens) and
``poses.jsonl`` (the ego, the actors and the cameras, in CARLA's frame): what
``tools/carla_replay.py`` needs to render the same drive with CARLA itself,
through the same lenses (``carla_<camera>_<frame>.jpg``).

``python -m cosmos3_canvas.record <rollout dir> <out.mp4>`` lays them out as
camera grids side by side: the generated frames, and the WSM drawn over them
(``--panels`` picks from generated, overlay, wsm, and, once
``tools/carla_replay.py`` has rendered the drive with CARLA, carla and
carla_overlay).

The files are encoded on a background thread so that recording costs the
rollout little time.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class Recorder:
    """Writes one rollout's frames under ``root/rollout_<n>`` (see the module docstring)."""

    def __init__(self, root: str | Path, cameras: Sequence[str], size_hw: tuple[int, int]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cameras = list(cameras)
        self.size_hw = size_hw
        self.rollouts = 0
        self.dir = self.root / "rollout_000"
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._run, name="cosmos3-record", daemon=True)
        self._thread.start()

    def new_rollout(self, meta: dict | None = None) -> None:
        """Start ``rollout_<n>``; ``meta`` (the map, the cameras' lenses) goes to its ``meta.json``."""
        import json

        self.dir = self.root / f"rollout_{self.rollouts:03d}"
        self.rollouts += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def poses(self, t: float, record: dict) -> None:
        """One tick's poses (CARLA frame), ``t`` seconds after the rollout's start, to ``poses.jsonl``."""
        import json

        with open(self.dir / "poses.jsonl", "a") as f:
            f.write(json.dumps({"t": round(t, 6), **record}) + "\n")

    def put(self, kind: str, frames: range, images: np.ndarray) -> None:
        """``images``: ``[V, F, H, W, 3]`` uint8 for the configured cameras at stream ``frames``."""
        self._queue.put((self.dir, kind, frames, images))

    def flush(self) -> None:
        """Wait until everything put so far is written."""
        self._queue.join()

    def _run(self) -> None:
        from PIL import Image

        height, width = self.size_hw
        while True:
            directory, kind, frames, images = self._queue.get()
            try:
                directory.mkdir(parents=True, exist_ok=True)
                for v, camera in enumerate(self.cameras):
                    for i, f in enumerate(frames):
                        image = Image.fromarray(np.ascontiguousarray(images[v, i, ..., :3]))
                        if image.size != (width, height):
                            image = image.resize((width, height), Image.BILINEAR)
                        image.save(directory / f"{kind}_{camera}_{f:05d}.jpg", quality=90)
            finally:
                self._queue.task_done()


#: grid of the camera tiles in the video, by Hyperion key (None: empty)
LAYOUT = (
    (None, "camera_front_tele_30fov", None),
    ("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"),
    ("camera_rear_left_70fov", "camera_rear_tele_30fov", "camera_rear_right_70fov"),
)


def overlay(rgb: np.ndarray, wsm: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """The WSM's drawn pixels (anything not black) over the generated frame."""
    drawn = wsm.max(axis=-1, keepdims=True) > 24
    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * wsm.astype(np.float32)
    return np.where(drawn, np.rint(blended), rgb).astype(np.uint8)


#: the panels a video frame can show side by side, each a grid of the cameras
PANELS = {"generated": "generated", "overlay": "WSM over generated", "wsm": "WSM control",
          "carla": "CARLA", "carla_overlay": "WSM over CARLA"}


def _panel(panel: str, rgb: np.ndarray | None, wsm: np.ndarray | None,
           real: np.ndarray | None) -> np.ndarray | None:
    base = {"generated": rgb, "overlay": rgb, "wsm": wsm, "carla": real, "carla_overlay": real}[panel]
    if base is None or panel not in ("overlay", "carla_overlay"):
        return base
    return None if wsm is None else overlay(base, wsm)


def compose(directory: Path, frame: int, cameras: Sequence[str],
            panels: Sequence[str] = ("generated", "overlay")) -> np.ndarray | None:
    """One video frame: a grid of the cameras (:data:`LAYOUT`) per panel, side by side."""
    from PIL import Image, ImageDraw, ImageFont

    def load(kind: str, camera: str) -> np.ndarray | None:
        path = directory / f"{kind}_{camera}_{frame:05d}.jpg"
        return np.asarray(Image.open(path).convert("RGB")) if path.exists() else None

    rgb = {c: load("rgb", c) for c in cameras}
    wsm = {c: load("wsm", c) for c in cameras}
    real = {c: load("carla", c) for c in cameras} if {"carla", "carla_overlay"} & set(panels) else {}
    shape = next((im.shape for im in rgb.values() if im is not None), None)
    if shape is None:
        return None
    font = ImageFont.load_default(size=max(12, shape[0] // 16))
    rows_used = [row for row in LAYOUT if any(c in cameras for c in row if c)]
    grids = []
    for panel in panels:
        rows = []
        for row in rows_used:
            tiles = []
            for camera in row:
                image = None
                if camera in cameras:
                    image = _panel(panel, rgb[camera], wsm[camera], real.get(camera))
                if image is None:
                    tiles.append(np.zeros(shape, np.uint8))
                    continue
                tile = Image.fromarray(image)
                ImageDraw.Draw(tile).text((10, 6), camera.removeprefix("camera_"), font=font, fill=(255, 255, 0),
                                          stroke_width=2, stroke_fill=(0, 0, 0))
                tiles.append(np.asarray(tile))
            rows.append(np.concatenate(tiles, axis=1))
        grid = Image.fromarray(np.concatenate(rows, axis=0))
        ImageDraw.Draw(grid).text((10, 6 + shape[0] // 12), PANELS[panel], font=font, fill=(0, 255, 255),
                                  stroke_width=2, stroke_fill=(0, 0, 0))
        grids.append(np.asarray(grid))
    gap = np.full((grids[0].shape[0], 8, 3), 255, np.uint8)
    return np.concatenate([g for pair in zip(grids, [gap] * len(grids)) for g in pair][:-1], axis=1)


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import re

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rollout", type=Path, help="a rollout_<n> directory under record_dir")
    parser.add_argument("output", type=Path, help="the video to write (.mp4)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    parser.add_argument("--width", type=int, default=2496, help="scale the video to this width")
    parser.add_argument("--panels", default="generated,overlay",
                        help=f"comma-separated, side by side: {', '.join(PANELS)}")
    args = parser.parse_args(argv)

    names = [p.name for p in args.rollout.glob("rgb_*.jpg")]
    pattern = re.compile(r"rgb_(camera_.+)_(\d{5})\.jpg")
    cameras = sorted({m.group(1) for n in names if (m := pattern.fullmatch(n))})
    frames = sorted({int(m.group(2)) for n in names if (m := pattern.fullmatch(n))})
    if not frames:
        raise SystemExit(f"no recorded frames in {args.rollout}")
    from PIL import Image

    with imageio.get_writer(args.output, fps=args.fps / args.every, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for frame in frames[::args.every]:
            image = compose(args.rollout, frame, cameras, args.panels.split(","))
            if image is None:
                continue
            scale = args.width / image.shape[1]
            size = (args.width, int(round(image.shape[0] * scale / 8)) * 8)
            writer.append_data(np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR)))
    print(f"{args.output}: {len(frames[::args.every])} frames, cameras {cameras}")


if __name__ == "__main__":
    main()

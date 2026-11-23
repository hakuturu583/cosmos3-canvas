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
carla_overlay). ``--trajectory`` draws planned paths (a driver's) on every
camera, ``--phase`` frames the video by phase.

The files are encoded on a background thread so that recording costs the
rollout little time.
"""

from __future__ import annotations

import json
import math
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


# -- poses and trajectories ---------------------------------------------------------------


def _lerp_pose(a: list[float], b: list[float], w: float) -> list[float]:
    """x, y, z, pitch, yaw, roll: positions linearly, angles the short way round."""
    out = [a[i] + w * (b[i] - a[i]) for i in range(3)]
    for i in range(3, 6):
        out.append(a[i] + w * (((b[i] - a[i]) + 180.0) % 360.0 - 180.0))
    return out


class Poses:
    """``poses.jsonl``: each tracked thing's pose at a time, interpolated between ticks."""

    def __init__(self, path: Path) -> None:
        self.ticks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not self.ticks:
            raise ValueError(f"{path}: no ticks")
        self.times = np.array([tick["t"] for tick in self.ticks])

    def _get(self, tick: dict, kind: str, key: str) -> list[float] | None:
        if kind == "ego":
            return tick["ego"]["pose"]
        if kind == "camera":
            return tick["cameras"].get(key)
        actor = tick["actors"].get(key)
        return None if actor is None else actor["pose"]

    def at(self, t: float, kind: str, key: str = "") -> list[float] | None:
        i = int(np.searchsorted(self.times, t))
        if i == 0 or i == len(self.ticks):
            return self._get(self.ticks[min(i, len(self.ticks) - 1)], kind, key)
        a, b = self.ticks[i - 1], self.ticks[i]
        pa, pb = self._get(a, kind, key), self._get(b, kind, key)
        if pa is None or pb is None:
            return pa or pb
        return _lerp_pose(pa, pb, (t - a["t"]) / (b["t"] - a["t"]))

    def actors(self) -> dict[str, str]:
        """Every actor seen: id -> type_id."""
        seen: dict[str, str] = {}
        for tick in self.ticks:
            for actor_id, actor in tick["actors"].items():
                seen.setdefault(actor_id, actor["type_id"])
        return seen


def ue_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """3x3 whose columns are a CARLA actor's forward, right and up axes in the world (Unreal's)."""
    p, y, r = (math.radians(a) for a in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return np.array([forward, right, up], dtype=float).T


def project(points: np.ndarray, camera_pose: Sequence[float], camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """CARLA-world ``points`` [N, 3] -> (pixels [N, 2], in view [N]) on a recorded camera.

    ``camera_pose``: x, y, z, pitch, yaw, roll (poses.jsonl); ``camera``: its meta.json
    entry, whose F-theta lens is the generated frame's.
    """
    lens = camera["lens"]
    local = (np.asarray(points, np.float64) - np.asarray(camera_pose[:3])) @ ue_rotation(*camera_pose[3:])
    rays = np.stack([local[:, 1], -local[:, 2], local[:, 0]], axis=1)  # optical: right, down, forward
    xy = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(xy, rays[:, 2])
    radius = np.polynomial.polynomial.polyval(theta, lens["angle_to_pixeldist_poly"])
    unit = rays[:, :2] / np.maximum(xy, 1e-12)[:, None]
    c, d, e = lens["linear_cde"]
    offset = radius[:, None] * unit
    u = c * offset[:, 0] + d * offset[:, 1] + lens["principal_point"][0]
    v = e * offset[:, 0] + offset[:, 1] + lens["principal_point"][1]
    pixels = np.stack([u, v], axis=1)
    margin = 0.5 * max(camera["image_width"], camera["image_height"])
    inside = ((u > -margin) & (u < camera["image_width"] + margin)
              & (v > -margin) & (v < camera["image_height"] + margin))
    return pixels, (theta < float(lens["max_angle"])) & (rays[:, 2] > 0.1) & inside


class Trajectories:
    """Planned paths over time: a JSONL of ``{"t": s from the rollout's start, "points": [[x, y, z], ...]}``
    (CARLA's world frame). :meth:`at` is the latest path planned by a time."""

    def __init__(self, path: Path) -> None:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: row["t"])
        self.times = np.array([row["t"] for row in rows])
        self.paths = [np.asarray(row["points"], np.float64).reshape(-1, 3) for row in rows]

    def at(self, t: float) -> np.ndarray | None:
        i = int(np.searchsorted(self.times, t + 1e-9, side="right")) - 1
        return None if i < 0 else self.paths[i]


def draw_path(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray,
              color: tuple[int, int, int] = (255, 0, 255), width: int = 4) -> np.ndarray:
    """The path's in-view runs as polylines (a dark outline under ``color``), a dot at each point."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    runs, run = [], []
    for point, ok in zip(pixels, valid):
        if ok:
            run.append((float(point[0]), float(point[1])))
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    for run in runs:
        if len(run) > 1:
            draw.line(run, fill=(0, 0, 0), width=width + 3, joint="curve")
            draw.line(run, fill=color, width=width, joint="curve")
        for x, y in run:
            draw.ellipse((x - width, y - width, x + width, y + width), fill=color, outline=(0, 0, 0))
    return np.asarray(canvas)


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
            panels: Sequence[str] = ("generated", "overlay"), mark=None) -> np.ndarray | None:
    """One video frame: a grid of the cameras (:data:`LAYOUT`) per panel, side by side.

    ``mark(camera, image) -> image``, if given, draws on every panel's tile of each camera.
    """
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
                if mark is not None:
                    image = mark(camera, image)
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


#: frame colours of the phases (``--phase``), in order
PHASE_COLORS = ((255, 150, 0), (0, 200, 90), (60, 140, 255), (230, 60, 200))


def parse_phases(specs: Sequence[str]) -> list[tuple[float, str]]:
    """``["0:warm-up", "6.95:policy"]`` -> phases by start time (s from the rollout's start)."""
    phases = []
    for spec in specs:
        start, _, label = spec.partition(":")
        phases.append((float(start), label))
    return sorted(phases)


def frame_phase(image: np.ndarray, t: float, phases: Sequence[tuple[float, str]],
                border: int = 14) -> np.ndarray:
    """``image`` framed in the colour of the phase ``t`` falls in, its label on a banner at the top."""
    from PIL import Image, ImageDraw, ImageFont

    current = [i for i, (start, _) in enumerate(phases) if start <= t + 1e-9]
    if not current:
        return image
    index = current[-1]
    color, label = PHASE_COLORS[index % len(PHASE_COLORS)], phases[index][1]
    out = image.copy()
    out[:border], out[-border:], out[:, :border], out[:, -border:] = color, color, color, color
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(16, image.shape[0] // 24))
    text = f"{label}   t = {t:5.2f} s"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (image.shape[1] - (right - left)) // 2
    draw.rectangle((x - 16, 0, x + right - left + 16, bottom - top + 2 * border), fill=color)
    draw.text((x, border - top // 2), text, font=font, fill=(0, 0, 0))
    return np.asarray(canvas)


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
    parser.add_argument("--trajectory", type=Path, default=None,
                        help="a JSONL of planned paths, {\"t\": s from the rollout's start, \"points\": "
                             "[[x, y, z], ...]} in CARLA's world frame, drawn on every camera")
    parser.add_argument("--phase", action="append", default=[], metavar="START:LABEL",
                        help="frame the video from START seconds (from the rollout's start) on in a colour, "
                             "labelled LABEL; repeat for each phase, e.g. --phase 0:warm-up --phase 6.95:policy")
    args = parser.parse_args(argv)

    names = [p.name for p in args.rollout.glob("rgb_*.jpg")]
    pattern = re.compile(r"rgb_(camera_.+)_(\d{5})\.jpg")
    cameras = sorted({m.group(1) for n in names if (m := pattern.fullmatch(n))})
    frames = sorted({int(m.group(2)) for n in names if (m := pattern.fullmatch(n))})
    if not frames:
        raise SystemExit(f"no recorded frames in {args.rollout}")
    from PIL import Image

    phases = parse_phases(args.phase)
    mark_at = None
    if args.trajectory is not None:
        meta = json.loads((args.rollout / "meta.json").read_text())
        poses, plans = Poses(args.rollout / "poses.jsonl"), Trajectories(args.trajectory)

        def mark_at(t: float):
            path = plans.at(t)
            if path is None:
                return None

            def mark(camera: str, image: np.ndarray) -> np.ndarray:
                pose = poses.at(t, "camera", camera)
                if pose is None or camera not in meta["cameras"]:
                    return image
                pixels, valid = project(path, pose, meta["cameras"][camera])
                return draw_path(image, pixels, valid)
            return mark
    with imageio.get_writer(args.output, fps=args.fps / args.every, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for frame in frames[::args.every]:
            image = compose(args.rollout, frame, cameras, args.panels.split(","),
                            mark=mark_at(frame / args.fps) if mark_at else None)
            if image is None:
                continue
            if phases:
                image = frame_phase(image, frame / args.fps, phases)
            scale = args.width / image.shape[1]
            size = (args.width, int(round(image.shape[0] * scale / 8)) * 8)
            writer.append_data(np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR)))
    print(f"{args.output}: {len(frames[::args.every])} frames, cameras {cameras}")


if __name__ == "__main__":
    main()

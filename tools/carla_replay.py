"""Render a recorded cosmos3-canvas rollout with CARLA itself, through the same lenses.

A rollout recorded with ``renderer.options.record_dir`` (cosmos3_canvas/record.py)
holds the generated frames, the WSM controls, and per tick the poses of the ego,
the actors and the cameras (``poses.jsonl``, CARLA's frame) with each camera's
F-theta lens (``meta.json``). This script puts the same actors at the same
poses in a CARLA server that runs without carla_canvas, and renders each
camera at each recorded frame's time. Poses between ticks are interpolated.

CARLA renders pinholes only, so each camera is a pinhole wide enough to hold
the lens' field of view, at the same pose. The image is then resampled into
the Hyperion camera's F-theta lens, and lands pixel for pixel on the generated
frame and on its WSM: ``carla_<camera>_<frame>.jpg`` next to them. Pixels past
the lens' max angle are black.

Run it with a Python that has the CARLA client (``carla`` or
``typesafe_carla``), numpy and Pillow, against a running CARLA server::

    python tools/carla_replay.py /tmp/cosmos3_record/rollout_000 --port 2000

Then ``python -m cosmos3_canvas.record <rollout> out.mp4 --panels carla,carla_overlay,generated``.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos3_canvas.record import Poses  # noqa: E402  (numpy only: no carla_canvas, no model)



def _carla():
    try:
        import carla
    except ImportError:  # the typed build autoware_carla_scenario uses
        import typesafe_carla.carla as carla
    return carla


# -- poses: cosmos3_canvas.record.Poses -------------------------------------------


def _transform(pose: list[float]):
    carla = _carla()
    x, y, z, pitch, yaw, roll = pose
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


# -- lenses -------------------------------------------------------------------------


class LensMap:
    """Resamples a pinhole CARLA image into one Hyperion camera's F-theta lens."""

    def __init__(self, camera: dict, margin_deg: float = 6.0, max_side: int = 4096) -> None:
        lens = camera["lens"]
        self.width, self.height = int(camera["image_width"]), int(camera["image_height"])
        max_angle = float(lens["max_angle"])
        self.fov = min(170.0, math.degrees(2 * max_angle) + margin_deg)
        center_px_per_rad = float(lens["angle_to_pixeldist_poly"][1])
        # the pinhole's centre at least as fine as the lens' centre
        side = 2 * math.tan(math.radians(self.fov) / 2) * center_px_per_rad
        self.side = int(min(max_side, math.ceil(side / 16) * 16))
        focal = self.side / 2 / math.tan(math.radians(self.fov) / 2)

        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        cx, cy = lens["principal_point"]
        c, d, e = lens["linear_cde"]
        # [[c, d], [e, 1]] @ offset = pixel - principal point
        du, dv = u - cx, v - cy
        det = c - d * e
        ox = (du - d * dv) / det
        oy = (-e * du + c * dv) / det
        radius = np.hypot(ox, oy)
        theta = np.polynomial.polynomial.polyval(radius, lens["pixeldist_to_angle_poly"])
        unit_x = ox / np.maximum(radius, 1e-12)
        unit_y = oy / np.maximum(radius, 1e-12)
        x, y, z = np.sin(theta) * unit_x, np.sin(theta) * unit_y, np.cos(theta)  # optical axes
        self.valid = (theta <= max_angle) & (z > 1e-6)
        z = np.where(self.valid, z, 1.0)
        self.pu = (self.side / 2 - 0.5) + focal * x / z
        self.pv = (self.side / 2 - 0.5) + focal * y / z
        self.valid &= (self.pu >= 0) & (self.pu <= self.side - 1) & (self.pv >= 0) & (self.pv <= self.side - 1)

    def __call__(self, pinhole: np.ndarray) -> np.ndarray:
        """``[side, side, 3]`` uint8 -> ``[height, width, 3]`` uint8, bilinear."""
        pu = np.clip(self.pu, 0, self.side - 1.001)
        pv = np.clip(self.pv, 0, self.side - 1.001)
        u0, v0 = pu.astype(np.int64), pv.astype(np.int64)
        fu, fv = (pu - u0)[..., None], (pv - v0)[..., None]
        img = pinhole.astype(np.float32)
        top = img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu
        bottom = img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu
        out = top * (1 - fv) + bottom * fv
        out[~self.valid] = 0
        return np.rint(out).astype(np.uint8)


# -- rendering ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rollout", type=Path, help="a rollout_<n> directory under record_dir")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--weather", default="", help="a carla.WeatherParameters preset (default: the map's)")
    parser.add_argument("--settle", type=int, default=40, help="ticks for the auto-exposure to settle first")
    parser.add_argument("--with-ego", action="store_true", help="put the ego's body in the scene too")
    parser.add_argument("--every", type=int, default=1, help="render every n-th recorded frame")
    args = parser.parse_args()

    meta = json.loads((args.rollout / "meta.json").read_text())
    poses = Poses(args.rollout / "poses.jsonl")
    pattern = re.compile(r"rgb_(camera_.+)_(\d{5})\.jpg")
    frames = sorted({int(m.group(2)) for p in args.rollout.glob("rgb_*.jpg") if (m := pattern.fullmatch(p.name))})
    frames = frames[::args.every]
    fps = float(meta.get("fps", 30.0))
    lenses = {key: LensMap(camera) for key, camera in meta["cameras"].items()}

    carla = _carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    map_name = meta["map"].rsplit("/", 1)[-1]
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != map_name:
        world = client.load_world(map_name)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    for attempt in range(5):  # right after a map load the server can still refuse it
        try:
            world.apply_settings(settings)
            break
        except RuntimeError as error:
            print(f"apply_settings: {error}; retrying", flush=True)
            time.sleep(5)
            world = client.get_world()
    else:
        raise SystemExit("CARLA refused the synchronous settings")
    if args.weather:
        world.set_weather(getattr(carla.WeatherParameters, args.weather))
    library = world.get_blueprint_library()
    spawned = []
    try:
        # Out of sight, high above the map, one spot each (spawns must not overlap).
        x0, y0 = poses.ticks[0]["ego"]["pose"][:2]

        def parked(i: int):
            return carla.Transform(carla.Location(x=x0 + 20.0 * i, y=y0, z=400.0))
        # The ego's own body is left out unless asked for: the cameras sit at its roof line,
        # where CARLA's body shows up as a smear across the bottom of the wide cameras.
        actors = {"ego": ("ego", "", poses.ticks[0]["ego"]["type_id"])} if args.with_ego else {}
        actors.update({actor_id: ("actor", actor_id, type_id) for actor_id, type_id in poses.actors().items()})
        bodies = {}
        for i, (name, (kind, key, type_id)) in enumerate(actors.items()):
            blueprints = library.filter(type_id)
            if not blueprints:
                print(f"no blueprint {type_id!r} for {name}; left out")
                continue
            blueprint = blueprints[0]
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "replay_" + name)
            actor = world.spawn_actor(blueprint, parked(i))
            actor.set_simulate_physics(False)
            spawned.append(actor)
            bodies[name] = (actor, kind, key, parked(i))
        cameras = {}
        for key, lens in lenses.items():
            blueprint = library.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(lens.side))
            blueprint.set_attribute("image_size_y", str(lens.side))
            blueprint.set_attribute("fov", f"{lens.fov:.4f}")
            # the bodies are teleported every frame: motion blur would smear them (the ego's hood)
            for attribute in ("motion_blur_intensity", "motion_blur_max_distortion"):
                if blueprint.has_attribute(attribute):
                    blueprint.set_attribute(attribute, "0")
            sensor = world.spawn_actor(blueprint, parked(len(actors) + len(cameras)))
            images: queue.Queue = queue.Queue()
            sensor.listen(images.put)
            spawned.append(sensor)
            cameras[key] = (sensor, images)
        print(f"{len(bodies)} bodies, cameras {', '.join(f'{k} {v.side}px {v.fov:.0f} deg' for k, v in lenses.items())}")

        def place(t: float) -> None:
            for actor, kind, key, hidden in bodies.values():
                pose = poses.at(t, kind, key)
                actor.set_transform(_transform(pose) if pose is not None else hidden)
            for key, (sensor, _) in cameras.items():
                pose = poses.at(t, "camera", key)
                if pose is not None:
                    sensor.set_transform(_transform(pose))

        def tick() -> dict[str, np.ndarray]:
            frame = world.tick()
            out = {}
            for key, (_, images) in cameras.items():
                while True:
                    image = images.get(timeout=60.0)
                    if image.frame >= frame:
                        break
                bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
                out[key] = bgra[..., 2::-1]
            return out

        place(frames[0] / fps)
        for _ in range(args.settle):
            tick()
        for n, f in enumerate(frames):
            place(f / fps)
            for key, pinhole in tick().items():
                Image.fromarray(lenses[key](pinhole)).save(args.rollout / f"carla_{key}_{f:05d}.jpg", quality=90)
            if n % 30 == 0:
                print(f"frame {f} ({n + 1}/{len(frames)})", flush=True)
    finally:
        for actor in spawned:
            if hasattr(actor, "stop"):
                try:
                    actor.stop()
                except RuntimeError:
                    pass
            actor.destroy()
        world.apply_settings(original)


if __name__ == "__main__":
    main()

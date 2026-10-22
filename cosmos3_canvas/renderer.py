"""Cosmos3-Nano-Sim-Auto as a carla_canvas renderer: CARLA's cameras and LiDAR from the world model.

carla_canvas (https://github.com/hakuturu583/carla_canvas) swaps CARLA's sensor
rendering behind the CARLA API. This backend answers the ego's RGB cameras and
its ray-cast LiDAR with Cosmos3-Nano-Sim-Auto
(https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto), which generates the
Hyperion 8 cameras and the Pandar128 LiDAR together, 0.4 s at a time:

* the map's OpenDRIVE (LoadWorld) becomes an RDS-HQ clip with roadgen, whose
  static layers the cookbook's renderers draw the controls from (controls.py);
* every 0.4 s of simulation time one chunk is due (12 camera frames, 4 sweeps;
  the first chunk is frame 0 and sweep 0). Its WSM control frames and LiDAR
  HD-map range maps are drawn from the sensors' poses and the actors' boxes at
  the frames' times, the later ones extrapolated from the last ticks
  (trajectory.py), and generated with the rolling K/V cache (stream.py);
* a CARLA camera reads the frame nearest to its tick's time, as generated:
  its Hyperion camera's F-theta (fisheye) image; a LiDAR measurement reads each
  ray from the sweep covering its time (sensors.py).

Which Hyperion camera a CARLA camera is: ``options.camera_map`` (role_name ->
camera key), else the configured camera nearest to its yaw on the vehicle.
Every configured camera needs a CARLA camera on the ego (its pose), and the ego
one LiDAR (else a virtual one 1.94 m above the ego's origin).

The cameras are the model's, not any camera: a CARLA camera must be its Hyperion
camera's generated image exactly -- its size (832x480 at resolution 480) and the
fov the camera is named for (120, 70 or 30) -- and is refused otherwise, as is a
second CARLA camera for one Hyperion camera. ``python -m cosmos3_canvas.camera
<key>`` prints the camera as an autoware_carla_scenario ``driver.cameras`` entry,
with its F-theta lens.

Configuration (``renderer`` section of the carla_canvas file)::

    renderer:
      backend: cosmos3_canvas:Cosmos3Renderer
      startup_timeout: 600     # LoadWorld: the model loads
      timeout: 300             # a tick that generates a chunk
      options:
        checkpoint: ~/workspace/cosmos3/Cosmos3-Nano-Sim-Auto
        cameras: [camera_front_wide_120fov]
        caption: "A daytime drive on a dry, sunlit urban road with clear lane markings."
        horizon_seconds: 60
        seed: 0
        resolution: "480"
        actors: ["vehicle.*", "walker.pedestrian.*"]

Traffic-light states are not drawn (carla_canvas does not send them): lights are
"unknown" in the controls.
"""

from __future__ import annotations

import hashlib
import logging
import math
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from carla_canvas import CameraFrame, CameraView, Renderer, WorldInfo
from carla_canvas.geometry import Transform, to_frame
from carla_canvas.renderer import LidarView, RayHits

from .controls import (HYPERION_CAMERAS, Box, LidarControl, WsmCamera, camera_pose_cv, default_ftheta,
                       lidar_control_frames)
from .sensors import GENERATED_SIZE, GeneratedLens
from .trajectory import Track

log = logging.getLogger("cosmos3_canvas")

FRAME_SECONDS = 1.0 / 30
SWEEP_SECONDS = 0.1
#: the cameras' nominal yaw on the rig (degrees, left positive): the default CARLA -> Hyperion mapping
NOMINAL_YAW = {
    "camera_front_wide_120fov": 0.0, "camera_front_tele_30fov": 0.0, "camera_cross_left_120fov": 60.0,
    "camera_cross_right_120fov": -60.0, "camera_rear_left_70fov": 140.0, "camera_rear_right_70fov": -140.0,
    "camera_rear_tele_30fov": 180.0,
}
VIRTUAL_LIDAR_HEIGHT_M = 1.94


def _box_type(info) -> str:
    if info.type_id.startswith("walker."):
        return "Pedestrian"
    if info.attributes.get("number_of_wheels") == "2":
        return "Cyclist"
    return "Truck" if 2 * info.bounding_box.extent[0] > 6.5 else "Car"


@dataclass
class _Rollout:
    """One episode's generation."""

    t0: float
    last_time: float
    ego_id: int
    sensors: dict[int, Track] = field(default_factory=dict)  # camera / LiDAR sensor-to-world (FLU)
    ego: Track = field(default_factory=lambda: Track(maxlen=4))
    actors: dict[int, tuple[Track, np.ndarray, str]] = field(default_factory=dict)  # box track, lwh, type
    camera_keys: dict[int, str] = field(default_factory=dict)  # CARLA camera id -> Hyperion key
    lidar_id: int | None = None
    wsm: dict[str, WsmCamera] = field(default_factory=dict)
    frames: dict[int, list[np.ndarray]] = field(default_factory=dict)  # stream frame -> per configured camera
    sweeps: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # stream sweep -> range, intensity
    checked: set[int] = field(default_factory=set)  # CARLA cameras found to be their Hyperion camera


class Cosmos3Renderer(Renderer):
    name = "cosmos3"
    version = "0.1.0"
    camera_outputs = frozenset({"rgb"})
    coordinate_frame = "right_handed"  # OpenDRIVE's, roadgen's RDS-HQ export's: x forward, y left, z up
    semantic_lidar = False

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        o = self.options
        self.cameras = list(o.get("cameras", ["camera_front_wide_120fov"]))
        for key in self.cameras:
            if key not in HYPERION_CAMERAS:
                raise ValueError(f"unknown Hyperion 8 camera {key!r}; one of {HYPERION_CAMERAS}")
        self.caption = str(o.get("caption", "A daytime drive on a dry, sunlit urban road with clear lane markings."))
        self.captions = [str(c) for c in o.get("captions", [self.caption] * len(self.cameras))]
        self.camera_map = dict(o.get("camera_map", {}))
        for role, key in self.camera_map.items():
            if key not in self.cameras:
                raise ValueError(f"camera_map maps {role!r} to {key!r}, which is not one of the cameras "
                                 f"generated ({self.cameras})")
        self.horizon = float(o.get("horizon_seconds", 60.0))
        self.seed = int(o.get("seed", 0))
        self.resolution = str(o.get("resolution", "480"))
        if self.resolution not in GENERATED_SIZE:
            raise ValueError(f"resolution {self.resolution!r}: one of {sorted(GENERATED_SIZE)}")
        self.lenses = {key: GeneratedLens.of(default_ftheta(key), key, GENERATED_SIZE[self.resolution])
                       for key in self.cameras}
        self.actor_patterns = list(o.get("actors", ["vehicle.*", "walker.pedestrian.*"]))
        self.reset_gap = float(o.get("reset_after_seconds", 1.0))
        self.setup_args = [str(a) for a in o.get("setup_args", [])]
        self._rdshq: tuple[Path, str] | None = None
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._map_key: str | None = None
        self._stream = None
        self._lidar_control: LidarControl | None = None
        self._lidar_grid = None
        self._rollout: _Rollout | None = None
        self._lock = threading.RLock()
        self.timings: list[dict[str, float]] = []

    # -- carla_canvas -----------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        with self._lock:
            self._rollout = None
            if not world.opendrive:
                raise RuntimeError("cosmos3 needs the map's OpenDRIVE (carla_canvas protocol 2)")
            key = "xodr:" + hashlib.sha256(world.opendrive.encode()).hexdigest()
            if key != self._map_key:
                self._rdshq = self._export_rdshq(world.opendrive)
                self._lidar_control = LidarControl(*self._rdshq)
                self._map_key = key
            if self._stream is None:
                self._stream = self._load_stream(world)
                if tuple(self._stream.size) != GENERATED_SIZE[self.resolution]:
                    raise RuntimeError(f"the model generates {self._stream.size} frames at resolution "
                                       f"{self.resolution}, not {GENERATED_SIZE[self.resolution]}")
            else:
                self._stream.reset()
            if self._lidar_grid is None:
                from .sensors import LidarGrid

                self._lidar_grid = LidarGrid()

    def render_camera(self, view: CameraView) -> CameraFrame:
        with self._lock:
            rollout = self._advance(view.scene, view.sensor_id)
            key = rollout.camera_keys.get(view.sensor_id)
            cam = view.camera
            if view.sensor_id not in rollout.checked:
                # Before anything is generated for it: a camera that is not its
                # Hyperion camera's image is a configuration error, not a frame.
                role = view.attributes.get("role_name", "") if view.attributes else ""
                self.lenses[key].check_camera(cam.width, cam.height, cam.fov,
                                              f"CARLA camera {role or view.sensor_id} ({key})")
                rollout.checked.add(view.sensor_id)
            frame_index = max(0, round((view.scene.elapsed_seconds - rollout.t0) / FRAME_SECONDS))
            self._ensure(rollout, self._chunk_of_frame(frame_index))
            frames = rollout.frames.get(frame_index)
            generated = frames[self.cameras.index(key)] if frames is not None else None
            if generated is None:
                return CameraFrame(rgb=np.zeros((cam.height, cam.width, 3), np.uint8))
            return CameraFrame(rgb=np.ascontiguousarray(generated))

    def cast_lidar_rays(self, view: LidarView) -> RayHits:
        with self._lock:
            rollout = self._advance(view.scene, view.sensor_id)
            if view.sensor_id != rollout.lidar_id:
                return RayHits(distance=np.full(len(view.directions), np.inf))
            sweep = max(0, math.floor((view.scene.elapsed_seconds - rollout.t0) / SWEEP_SECONDS + 1e-6))
            self._ensure(rollout, self._chunk_of_sweep(sweep))
            maps = rollout.sweeps.get(sweep)
            if maps is None:
                return RayHits(distance=np.full(len(view.directions), np.inf))
            distance, intensity = self._lidar_grid.read(*maps, view.body_directions(), view.max_distance)
            return RayHits(distance=distance, intensity=intensity)

    def close(self) -> None:
        with self._lock:
            self._rollout = None
            self._stream = None
            if self._tmp is not None:
                self._tmp.cleanup()

    # -- setup (overridable for tests) ----------------------------------------------

    def _export_rdshq(self, xodr: str) -> tuple[Path, str]:
        import roadgen

        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix="cosmos3_canvas_")
        path = Path(self._tmp.name) / "map.xodr"
        path.write_text(xodr)
        road_map = roadgen.read_opendrive(str(path), sampling=0.5)
        for note in road_map.rdshq_warnings():
            log.info("RDS-HQ: %s", note)
        root = Path(self._tmp.name) / "rds_hq"
        clip = road_map.export_rdshq(str(root), trace=False)
        log.info("map exported to RDS-HQ clip %s", clip)
        return root, clip

    def _load_stream(self, world: WorldInfo):
        from .stream import Cosmos3Stream

        checkpoint = world.resolve(self.options["checkpoint"])
        log.info("loading Cosmos3 from %s (cameras %s)", checkpoint, self.cameras)
        workdir = Path(self.options.get("workdir", Path(tempfile.gettempdir()) / "cosmos3_canvas"))
        return Cosmos3Stream(checkpoint, self.cameras, self.captions, horizon_seconds=self.horizon,
                             workdir=world.resolve(str(workdir)), seed=self.seed, resolution=self.resolution,
                             extra_setup_args=self.setup_args)

    # -- the episode's tracks -----------------------------------------------------

    def _advance(self, scene, sensor_id: int) -> _Rollout:
        if self._stream is None:
            raise RuntimeError("load_world has not been called")
        t = float(scene.elapsed_seconds)
        ego_id = scene.actors[sensor_id].parent_id if sensor_id in scene.actors else 0
        rollout = self._rollout
        if rollout is not None and (t < rollout.last_time - 1e-6 or t - rollout.last_time > self.reset_gap
                                    or ego_id != rollout.ego_id):
            log.info("time jumped %.3f -> %.3f s (or a new ego): new rollout", rollout.last_time, t)
            rollout = None
        if rollout is None:
            self._stream.reset()
            rollout = self._rollout = _Rollout(t0=t, last_time=t, ego_id=ego_id)
            rollout.wsm = {key: WsmCamera(*self._rdshq, key) for key in self.cameras}
        rollout.last_time = t
        self._track(rollout, scene, t)
        return rollout

    def _track(self, rollout: _Rollout, scene, t: float) -> None:
        frame = self.coordinate_frame
        if rollout.ego_id and rollout.ego_id in scene.states:
            rollout.ego.add(t, scene.transform(rollout.ego_id).matrix(frame))
        for actor_id, info in scene.actors.items():
            if info.parent_id != rollout.ego_id or not rollout.ego_id or actor_id not in scene.states:
                continue
            if info.type_id == "sensor.camera.rgb" or info.type_id.startswith("sensor.lidar.ray_cast"):
                rollout.sensors.setdefault(actor_id, Track(maxlen=4)).add(t, scene.transform(actor_id).matrix(frame))
            if info.type_id == "sensor.camera.rgb" and actor_id not in rollout.camera_keys:
                rollout.camera_keys[actor_id] = self._camera_key(info, rollout, actor_id)
            if info.type_id == "sensor.lidar.ray_cast" and rollout.lidar_id is None:
                rollout.lidar_id = actor_id
        present = set()
        for info, transform in scene.bodies(exclude={rollout.ego_id}):
            if info.parent_id == rollout.ego_id or not any(info.matches(p) for p in self.actor_patterns):
                continue
            b = info.bounding_box
            box_to_world = to_frame(transform.matrix() @ Transform(b.location, b.rotation).matrix(), frame)
            entry = rollout.actors.get(info.id)
            if entry is None:
                entry = rollout.actors[info.id] = (Track(maxlen=4), 2 * np.asarray(b.extent, np.float64), _box_type(info))
            entry[0].add(t, box_to_world)
            present.add(info.id)
        for gone in set(rollout.actors) - present:
            del rollout.actors[gone]

    def _camera_key(self, info, rollout: _Rollout, actor_id: int) -> str:
        role = info.attributes.get("role_name", "")
        if role in self.camera_map:
            return str(self.camera_map[role])
        sensor = rollout.sensors[actor_id].at(rollout.last_time)
        ego = rollout.ego.at(rollout.last_time)
        mount = np.linalg.inv(ego) @ sensor
        yaw = math.degrees(math.atan2(mount[1, 0], mount[0, 0]))
        taken = set(rollout.camera_keys.values())
        free = [k for k in self.cameras if k not in taken]
        if not free:
            raise ValueError(f"CARLA camera {role or actor_id} on the ego: every generated camera "
                             f"({self.cameras}) has one already; add its Hyperion camera to "
                             f"renderer.options.cameras, or remove the camera")
        return min(free, key=lambda k: abs((NOMINAL_YAW[k] - yaw + 180) % 360 - 180))

    # -- chunks -------------------------------------------------------------------

    @staticmethod
    def _chunk_of_frame(frame: int) -> int:
        return 0 if frame == 0 else math.ceil(frame / 12)

    @staticmethod
    def _chunk_of_sweep(sweep: int) -> int:
        return 0 if sweep == 0 else math.ceil(sweep / 4)

    def _boxes(self, rollout: _Rollout, t: float) -> list[Box]:
        return [Box(str(actor_id), kind, lwh, track.at(t)) for actor_id, (track, lwh, kind) in rollout.actors.items()]

    def _camera_pose(self, rollout: _Rollout, key: str, t: float) -> np.ndarray:
        ids = [i for i, k in rollout.camera_keys.items() if k == key]
        if not ids:
            raise RuntimeError(f"no CARLA camera on the ego is {key} (renderer.options.camera_map / cameras)")
        return camera_pose_cv(rollout.sensors[ids[0]].at(t))

    def _lidar_pose(self, rollout: _Rollout, t: float) -> np.ndarray:
        if rollout.lidar_id is not None:
            return rollout.sensors[rollout.lidar_id].at(t)
        pose = rollout.ego.at(t)
        pose[:3, 3] += pose[:3, 2] * VIRTUAL_LIDAR_HEIGHT_M
        return pose

    def _ensure(self, rollout: _Rollout, chunk: int) -> None:
        stream = self._stream
        if chunk >= stream.num_chunks:
            if stream.next_chunk >= stream.num_chunks and not getattr(rollout, "_warned", False):
                log.error("past the %.1f s horizon (renderer.options.horizon_seconds): sensors go dark",
                          self.horizon)
                rollout._warned = True  # type: ignore[attr-defined]
            chunk = stream.num_chunks - 1
        while stream.next_chunk <= chunk:
            self._generate(rollout, stream.next_chunk)

    def _generate(self, rollout: _Rollout, k: int) -> None:
        import torch

        from .stream import chunk_camera_frames, chunk_lidar_sweeps

        t_start = time.perf_counter()
        frames, sweeps = chunk_camera_frames(k), chunk_lidar_sweeps(k)
        camera_controls = np.stack([
            np.stack([rollout.wsm[key].render(self._camera_pose(rollout, key, rollout.t0 + f * FRAME_SECONDS),
                                              self._boxes(rollout, rollout.t0 + f * FRAME_SECONDS))
                      for f in frames])
            for key in self.cameras])  # [V,F,720,1280,3]
        ranges = np.stack([
            self._lidar_control.render(self._lidar_pose(rollout, rollout.t0 + s * SWEEP_SECONDS),
                                       self._lidar_pose(rollout, rollout.t0 + (s + 1) * SWEEP_SECONDS),
                                       self._boxes(rollout, rollout.t0 + s * SWEEP_SECONDS))
            for s in sweeps])  # [S,128,1800]
        controls_seconds = time.perf_counter() - t_start
        chunk = self._stream.step(torch.from_numpy(camera_controls), torch.from_numpy(lidar_control_frames(ranges)))
        rgb = chunk.rgb.numpy()
        for i, f in enumerate(frames):
            rollout.frames[f] = [rgb[v, i] for v in range(len(self.cameras))]
        for i, s in enumerate(sweeps):
            rollout.sweeps[s] = (chunk.range_m[i].numpy(), chunk.intensity[i].numpy())
        for old in [f for f in rollout.frames if f < frames.start - 24]:
            del rollout.frames[old]
        for old in [s for s in rollout.sweeps if s < sweeps.start - 8]:
            del rollout.sweeps[old]
        timing = {"chunk": k, "controls": controls_seconds, **chunk.seconds}
        self.timings.append(timing)
        log.info("chunk %d (%.1f-%.1f s): %s", k, frames.start * FRAME_SECONDS, frames.stop * FRAME_SECONDS,
                 {key: round(value, 2) for key, value in timing.items() if key != "chunk"})

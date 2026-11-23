"""Which Hyperion camera a CARLA camera on the ego stands for."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cosmos3_canvas.renderer import Cosmos3Renderer
from cosmos3_canvas.trajectory import Track


def _rollout() -> SimpleNamespace:
    ego = Track()
    ego.add(0.0, np.eye(4))
    return SimpleNamespace(ego=ego, sensors={}, camera_keys={}, last_time=0.0)


def _forward_camera(rollout: SimpleNamespace, actor_id: int, fov: str) -> SimpleNamespace:
    mount = Track()
    mount.add(0.0, np.eye(4))  # straight ahead
    rollout.sensors[actor_id] = mount
    return SimpleNamespace(attributes={"role_name": "", "fov": fov})


def test_two_forward_cameras_are_told_apart_by_their_fov() -> None:
    renderer = Cosmos3Renderer({"cameras": ["camera_front_wide_120fov", "camera_front_tele_30fov"]})
    rollout = _rollout()
    tele = _forward_camera(rollout, 1, "30.0")  # the first one seen is the tele
    wide = _forward_camera(rollout, 2, "120.0")
    rollout.camera_keys[1] = renderer._camera_key(tele, rollout, 1)
    rollout.camera_keys[2] = renderer._camera_key(wide, rollout, 2)
    assert rollout.camera_keys == {1: "camera_front_tele_30fov", 2: "camera_front_wide_120fov"}


def test_a_camera_without_a_known_fov_goes_by_its_yaw() -> None:
    renderer = Cosmos3Renderer({"cameras": ["camera_rear_tele_30fov", "camera_front_wide_120fov"]})
    rollout = _rollout()
    camera = _forward_camera(rollout, 1, "")
    assert renderer._camera_key(camera, rollout, 1) == "camera_front_wide_120fov"

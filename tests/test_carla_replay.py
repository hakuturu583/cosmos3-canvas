"""tools/carla_replay.py: interpolating poses, and resampling a pinhole into the F-theta lens."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("carla_replay", Path(__file__).parents[1] / "tools" / "carla_replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def test_poses_interpolate_between_ticks_the_short_way_round(tmp_path) -> None:
    ticks = [{"t": 0.0, "ego": {"pose": [0, 0, 0, 0, 170, 0]}, "actors": {}, "cameras": {"c": [1, 0, 0, 0, 0, 0]}},
             {"t": 0.1, "ego": {"pose": [2, 0, 0, 0, -170, 0]}, "actors": {}, "cameras": {"c": [3, 0, 0, 0, 0, 0]}}]
    path = tmp_path / "poses.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in ticks))
    poses = replay.Poses(path)
    np.testing.assert_allclose(poses.at(0.05, "ego"), [1, 0, 0, 0, 180, 0])
    np.testing.assert_allclose(poses.at(0.025, "camera", "c")[0], 1.5)
    assert poses.at(1.0, "ego") == ticks[-1]["ego"]["pose"]  # held past the last tick


def test_the_lens_centre_samples_the_pinhole_centre() -> None:
    camera = {"image_width": 832, "image_height": 480,
              "lens": {"principal_point": [416.0, 240.0], "pixeldist_to_angle_poly": [0.0, 1 / 400.0],
                       "angle_to_pixeldist_poly": [0.0, 400.0], "max_angle": 1.2,
                       "linear_cde": [1.0, 0.0, 0.0]}}
    lens = replay.LensMap(camera)
    pinhole = np.zeros((lens.side, lens.side, 3), np.uint8)
    c = lens.side // 2
    pinhole[c - 4:c + 4, c - 4:c + 4] = 255
    out = lens(pinhole)
    assert out.shape == (480, 832, 3)
    assert out[240, 416, 0] > 200 and out[240, 300, 0] == 0
    assert out[0, 0, 0] == 0 and not lens.valid[0, 0]  # the corner is past 1.2 rad

"""A sweep drawn from one LiDAR pose, read from where the LiDAR turns out to be."""

from __future__ import annotations

import math

import numpy as np
import pytest

carla_canvas = pytest.importorskip("carla_canvas")

from cosmos3_canvas.renderer import _from_drawn_pose  # noqa: E402


def _pose(yaw_deg: float = 0.0, xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = xyz
    return pose


def test_a_sweep_read_from_where_it_was_drawn_is_read_as_it_is() -> None:
    rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])
    directions, offset = _from_drawn_pose(_pose(30.0, (5, 2, 1)), _pose(30.0, (5, 2, 1)), rays)
    np.testing.assert_allclose(directions, rays, atol=1e-12)
    np.testing.assert_allclose(offset, 0.0, atol=1e-12)


def test_a_lidar_that_turned_since_reads_the_sweep_turned_back() -> None:
    """Drawn facing +x, the LiDAR now faces 10 degrees left: its forward ray is the
    drawn sweep's ray 10 degrees left of forward."""
    (direction,), _ = _from_drawn_pose(_pose(10.0), _pose(0.0), np.array([[1.0, 0.0, 0.0]]))
    assert math.degrees(math.atan2(direction[1], direction[0])) == pytest.approx(10.0)


def test_a_lidar_that_moved_on_adds_the_drawn_origins_offset_along_the_ray() -> None:
    """Drawn 0.5 m behind where the LiDAR now is: a range read ahead is 0.5 m
    further than the distance from the LiDAR, one read behind 0.5 m shorter."""
    rays = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    _, offset = _from_drawn_pose(_pose(0.0, (0.5, 0, 0)), _pose(0.0, (0.0, 0, 0)), rays)
    np.testing.assert_allclose(offset, [-0.5, 0.5])

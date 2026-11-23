"""Poses ahead of the last tick."""

from __future__ import annotations

import numpy as np

from cosmos3_canvas.trajectory import Track


def _at(x: float, y: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = (x, y, 0.0)
    return pose


def test_one_tick_without_a_velocity_holds_the_pose() -> None:
    track = Track()
    track.add(1.0, _at(3.0))
    np.testing.assert_allclose(track.at(1.4), _at(3.0))


def test_one_tick_moves_on_at_its_reported_velocity() -> None:
    track = Track()
    track.add(1.0, _at(3.0), np.array([8.0, -1.0, 0.0]))
    np.testing.assert_allclose(track.at(1.5)[:3, 3], [7.0, -0.5, 0.0])
    np.testing.assert_allclose(track.at(1.0), _at(3.0))  # not before it


def test_two_ticks_extrapolate_their_own_motion_not_the_velocity() -> None:
    track = Track()
    track.add(0.0, _at(0.0), np.array([100.0, 0.0, 0.0]))
    track.add(0.1, _at(1.0), np.array([100.0, 0.0, 0.0]))
    np.testing.assert_allclose(track.at(0.3)[:3, 3], [3.0, 0.0, 0.0], atol=1e-9)

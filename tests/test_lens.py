"""GeneratedLens: the F-theta lens of the generated frames, and the cameras it refuses."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from cosmos3_canvas.controls import WSM_SIZE
from cosmos3_canvas.sensors import GeneratedLens

#: An equidistant lens (radius = 400 px * angle) at 1277x720, as the cookbook's are rescaled.
FTHETA = SimpleNamespace(
    width=1277, height=720, _center=np.array([640.0, 362.0]), linear_cde=np.array([1.0, 0.0, 0.0]),
    _fw_poly=Polynomial([0.0, 400.0]), _bw_poly=Polynomial([0.0, 1 / 400.0]), _max_ray_angle=1.2)
SIZE = (480, 832)


def lens() -> GeneratedLens:
    return GeneratedLens.of(FTHETA, "camera_front_wide_120fov", SIZE)


def test_the_lens_is_the_control_lens_resized_and_cropped() -> None:
    rays = np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
                     for t in (0.0, 0.3, 1.1) for p in (0.0, 1.0, -2.5)])
    # The lens' pixels, through the 1280x720 control, to the 832x480 crop (pixel centres).
    theta = np.arccos(rays[:, 2])
    unit = rays[:, :2] / np.maximum(np.linalg.norm(rays[:, :2], axis=1), 1e-12)[:, None]
    lens_px = FTHETA._center + 400.0 * theta[:, None] * unit
    s = max(SIZE[0] / WSM_SIZE[1], SIZE[1] / WSM_SIZE[0])
    x = lens_px[:, 0] * WSM_SIZE[0] / FTHETA.width * s - (WSM_SIZE[0] * s - SIZE[1]) / 2 - 0.5
    y = lens_px[:, 1] * WSM_SIZE[1] / FTHETA.height * s - (WSM_SIZE[1] * s - SIZE[0]) / 2 - 0.5
    np.testing.assert_allclose(lens().ray_to_pixel(rays), np.stack([x, y], 1), atol=1e-9)


def test_the_two_polynomials_invert_each_other() -> None:
    lens_ = lens()
    angles = np.linspace(0, 1.2, 50)
    radii = np.polynomial.polynomial.polyval(angles, lens_.angle_to_pixeldist_poly)
    np.testing.assert_allclose(np.polynomial.polynomial.polyval(radii, lens_.pixeldist_to_angle_poly), angles,
                               atol=1e-12)


def test_only_the_generated_image_is_a_camera() -> None:
    lens().check_camera(832, 480, 120.0)
    for width, height, fov in [(960, 540, 95.0), (832, 480, 90.0), (960, 604, 120.0)]:
        with pytest.raises(ValueError, match="image_size_x=832, image_size_y=480, fov=120"):
            lens().check_camera(width, height, fov)


def test_the_driver_camera_entry_carries_the_lens() -> None:
    entry = lens().as_driver_camera()
    assert (entry["image_width"], entry["image_height"], entry["fov"]) == (832, 480, 120.0)
    assert entry["lens"]["model"] == "ftheta"
    assert entry["lens"]["max_angle"] == pytest.approx(1.2)
    assert entry["lens"]["linear_cde"][0] == pytest.approx(1280 / 1277)

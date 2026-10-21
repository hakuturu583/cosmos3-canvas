"""From the model's outputs to what CARLA's sensors measure.

Cameras: the model draws each Hyperion 8 camera through its F-theta lens, as
its 1280x720 WSM control was drawn and then resized and center-cropped to the
generation size (e.g. 832x480, the framework's ``_resize_and_center_crop``).
A CARLA camera (a pinhole at the same pose) samples that image along its own
pixel rays; rays outside the lens' field of view are black.

LiDARs: the model's range maps are the Pandar128's 128 rings x 1800 columns
(``cosmos_auto.lidar``: ring 0 on top, column 0 looking backwards, azimuth
decreasing with the column). Each CARLA ray reads the cell it points at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .controls import WSM_SIZE

LIDAR_COLUMNS = 1800


@dataclass
class PinholeFromFtheta:
    """A cv2.remap of generated frames onto one pinhole camera."""

    map_x: np.ndarray  # (H, W) float32, generated-image columns
    map_y: np.ndarray
    valid: np.ndarray  # (H, W) bool

    @staticmethod
    def build(ftheta, generated_size: tuple[int, int], width: int, height: int, fov_deg: float) -> "PinholeFromFtheta":
        """``ftheta``: the camera model the control was drawn with (its pixel grid scaled to
        WSM_SIZE); ``generated_size``: (H, W) of the model's frames."""
        gh, gw = generated_size
        focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        rays = np.stack([(u - width / 2) / focal, (v - height / 2) / focal, np.ones_like(u)], -1).reshape(-1, 3)
        pixels = ftheta.ray2pixel_np(rays)  # (N, 2) at the lens model's resolution
        angle = np.arccos(rays[:, 2] / np.linalg.norm(rays, axis=1))
        # Lens model pixels -> the 1280x720 control -> the resized and center-cropped generation.
        x = pixels[:, 0] * (WSM_SIZE[0] / ftheta.width)
        y = pixels[:, 1] * (WSM_SIZE[1] / ftheta.height)
        scale = max(gh / WSM_SIZE[1], gw / WSM_SIZE[0])
        x = x * scale - (WSM_SIZE[0] * scale - gw) / 2
        y = y * scale - (WSM_SIZE[1] * scale - gh) / 2
        valid = (angle <= float(ftheta._max_ray_angle)) & (x >= 0) & (x <= gw - 1) & (y >= 0) & (y <= gh - 1)
        shape = (height, width)
        return PinholeFromFtheta((x - 0.5).reshape(shape).astype(np.float32),
                                 (y - 0.5).reshape(shape).astype(np.float32), valid.reshape(shape))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        out = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[~self.valid] = 0
        return out


class LidarGrid:
    """Cells of the Pandar128 range map that sensor-frame rays point at."""

    def __init__(self) -> None:
        from cosmos_auto.lidar.sensor_model import SpinningLidarModel

        self.elevations = np.asarray(SpinningLidarModel.default().row_elevations_rad, np.float64)  # descending

    def cells(self, directions_flu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, columns, inside) of (N, 3) unit directions in FLU sensor axes."""
        d = np.asarray(directions_flu, np.float64)
        elevation = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        azimuth = np.arctan2(d[:, 1], d[:, 0])
        order = np.argsort(self.elevations)
        ascending = self.elevations[order]
        position = np.clip(np.searchsorted(ascending, elevation), 1, len(ascending) - 1)
        lower, upper = ascending[position - 1], ascending[position]
        nearest = np.where(elevation - lower < upper - elevation, position - 1, position)
        rows = order[nearest]
        margin = np.radians(0.1)
        inside = (elevation >= ascending[0] - margin) & (elevation <= ascending[-1] + margin)
        columns = np.floor((np.pi - azimuth) / (2 * np.pi) * LIDAR_COLUMNS).astype(np.int64) % LIDAR_COLUMNS
        return rows, columns, inside

    def read(self, range_m: np.ndarray, intensity: np.ndarray, directions_flu: np.ndarray,
             max_distance: float) -> tuple[np.ndarray, np.ndarray]:
        """Distances (inf: none) and intensities of the rays, from one sweep's maps."""
        rows, columns, inside = self.cells(directions_flu)
        distance = np.where(inside, range_m[rows, columns], 0.0)
        hit = (distance > 0) & (distance <= max_distance)
        return np.where(hit, distance, np.inf), np.where(hit, intensity[rows, columns], 0.0)

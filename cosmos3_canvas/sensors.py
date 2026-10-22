"""From the model's outputs to what CARLA's sensors measure.

Cameras: the model draws each Hyperion 8 camera through its F-theta lens, as
its 1280x720 WSM control was drawn and then resized and center-cropped to the
generation size (e.g. 832x480, the framework's ``_resize_and_center_crop``).
A CARLA camera standing for one gets that fisheye frame as it is, so it must
be exactly that image (GeneratedLens.check_camera): its size, and the fov the
camera is named for.

LiDARs: the model's range maps are the Pandar128's 128 rings x 1800 columns
(``cosmos_auto.lidar``: ring 0 on top, column 0 looking backwards, azimuth
decreasing with the column). Each CARLA ray reads the cell it points at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controls import WSM_SIZE

LIDAR_COLUMNS = 1800


#: The generated frames' size (H, W) at each ``resolution``, 16:9 (cosmos_framework's
#: ``RESOLUTION_RATIO_DICT``); checked against the loaded model's at LoadWorld.
GENERATED_SIZE = {"256": (192, 320), "480": (480, 832), "720": (720, 1280), "768": (768, 1360)}

#: The horizontal field of view a Hyperion camera is named for (degrees): what a CARLA
#: camera standing for it is to say as its ``fov``.
NOMINAL_FOV = {
    "camera_front_wide_120fov": 120.0, "camera_cross_left_120fov": 120.0, "camera_cross_right_120fov": 120.0,
    "camera_rear_left_70fov": 70.0, "camera_rear_right_70fov": 70.0,
    "camera_rear_tele_30fov": 30.0, "camera_front_tele_30fov": 30.0,
}


@dataclass(frozen=True)
class GeneratedLens:
    """A Hyperion camera's F-theta lens on the generated frames.

    The control was drawn through the lens at 1280x720 (WSM_SIZE) and the frames are
    that image resized and center-cropped to the generation size, so the frames are
    the same F-theta lens, scaled and shifted: the angle -> pixel-distance polynomial
    scaled by the rows' scale, the principal point moved with the crop, and the
    columns' slightly different scale folded into ``linear_cde`` (``[[c, d], [e, 1]]``
    maps the radial offset to pixels, as in NVIDIA's F-theta model).

    Pixel coordinates are OpenCV's: the centre of the top-left pixel is (0, 0).
    """

    camera_key: str
    width: int
    height: int
    principal_point: tuple[float, float]
    pixeldist_to_angle_poly: tuple[float, ...]  # radius (px) -> angle from the optical axis (rad)
    angle_to_pixeldist_poly: tuple[float, ...]  # angle (rad) -> radius (px)
    max_angle: float  # rad: the lens' (rays beyond it are outside the model)
    linear_cde: tuple[float, float, float]

    @staticmethod
    def of(ftheta, camera_key: str, generated_size: tuple[int, int]) -> "GeneratedLens":
        """``ftheta``: the camera model the control was drawn with (controls.default_ftheta);
        ``generated_size``: (H, W) of the model's frames."""
        gh, gw = generated_size
        scale = max(gh / WSM_SIZE[1], gw / WSM_SIZE[0])
        # lens pixels -> the 1280x720 control -> the resized, center-cropped frame
        sx = WSM_SIZE[0] / ftheta.width * scale
        sy = WSM_SIZE[1] / ftheta.height * scale
        ox = -(WSM_SIZE[0] * scale - gw) / 2 - 0.5  # continuous -> pixel-centre coordinates
        oy = -(WSM_SIZE[1] * scale - gh) / 2 - 0.5
        cx, cy = (float(v) for v in ftheta._center)
        c, d, e = (float(v) for v in np.asarray(ftheta.linear_cde).ravel())
        fw = np.asarray(ftheta._fw_poly.coef, np.float64) * sy
        bw = np.asarray(ftheta._bw_poly.coef, np.float64) / sy ** np.arange(len(ftheta._bw_poly.coef))
        return GeneratedLens(
            camera_key=camera_key, width=gw, height=gh,
            principal_point=(sx * cx + ox, sy * cy + oy),
            pixeldist_to_angle_poly=tuple(float(v) for v in bw),
            angle_to_pixeldist_poly=tuple(float(v) for v in fw),
            max_angle=float(ftheta._max_ray_angle),
            linear_cde=(c * sx / sy, d * sx / sy, e),
        )

    @property
    def fov(self) -> float:
        """The ``fov`` a CARLA camera standing for this lens says (degrees)."""
        return NOMINAL_FOV[self.camera_key]

    def ray_to_pixel(self, rays: np.ndarray) -> np.ndarray:
        """(N, 2) pixels of (N, 3) rays in the camera's optical axes (x right, y down, z forward)."""
        rays = np.asarray(rays, np.float64).reshape(-1, 3)
        xy = np.linalg.norm(rays[:, :2], axis=1)
        theta = np.arctan2(xy, rays[:, 2])
        radius = np.polynomial.polynomial.polyval(theta, self.angle_to_pixeldist_poly)
        unit = rays[:, :2] / np.maximum(xy, 1e-12)[:, None]
        c, d, e = self.linear_cde
        offset = radius[:, None] * unit
        u = c * offset[:, 0] + d * offset[:, 1] + self.principal_point[0]
        v = e * offset[:, 0] + offset[:, 1] + self.principal_point[1]
        return np.stack([u, v], 1)

    def check_camera(self, width: int, height: int, fov: float, sensor: str = "a CARLA camera") -> None:
        """Refuse a camera that is not this lens: cosmos3 draws only the Hyperion camera, as
        it was trained, so a CARLA camera standing for it must be exactly that image."""
        if (int(width), int(height)) != (self.width, self.height) or abs(float(fov) - self.fov) > 1e-3:
            raise ValueError(
                f"{sensor} is {int(width)}x{int(height)} with fov {float(fov):g}, but cosmos3 draws "
                f"{self.camera_key} as a {self.width}x{self.height} F-theta (fisheye) image only; set "
                f"image_size_x={self.width}, image_size_y={self.height}, fov={self.fov:g} (and the lens: "
                f"`python -m cosmos3_canvas.camera {self.camera_key}` prints the driver.cameras entry)")

    def as_driver_camera(self) -> dict:
        """The camera as an autoware_carla_scenario ``driver.cameras`` entry, lens included."""
        return {
            "logical_id": self.camera_key, "image_width": self.width, "image_height": self.height,
            "fov": self.fov,
            "lens": {
                "model": "ftheta",
                "principal_point": [round(v, 6) for v in self.principal_point],
                "pixeldist_to_angle_poly": [float(f"{v:.9g}") for v in self.pixeldist_to_angle_poly],
                "angle_to_pixeldist_poly": [float(f"{v:.9g}") for v in self.angle_to_pixeldist_poly],
                "max_angle": round(self.max_angle, 9),
                "linear_cde": [float(f"{v:.9g}") for v in self.linear_cde],
            },
        }


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

"""Print a Hyperion camera as cosmos3 generates it: an autoware_carla_scenario ``driver.cameras`` entry.

    python -m cosmos3_canvas.camera camera_front_wide_120fov [--resolution 480] [--format yaml|json]

The entry is the generated image's size, the fov the camera is named for and its
F-theta ``lens``: what a CARLA camera standing for the camera must say (the renderer
refuses any other), and what a driver policy and a T4 dataset are to be told the
frames are. Mount it on the ego with ``position_*`` / ``roll`` / ``pitch`` / ``yaw``.
Needs the cookbook's control renderers (``cosmos_auto``), not the model.
"""

from __future__ import annotations

import argparse
import json

from .controls import HYPERION_CAMERAS, default_ftheta
from .sensors import GENERATED_SIZE, GeneratedLens


def lens(camera_key: str, resolution: str = "480") -> GeneratedLens:
    """``camera_key``'s lens on the frames generated at ``resolution``."""
    if camera_key not in HYPERION_CAMERAS:
        raise ValueError(f"unknown Hyperion 8 camera {camera_key!r}; one of {HYPERION_CAMERAS}")
    if resolution not in GENERATED_SIZE:
        raise ValueError(f"resolution {resolution!r}: one of {sorted(GENERATED_SIZE)}")
    return GeneratedLens.of(default_ftheta(camera_key), camera_key, GENERATED_SIZE[resolution])


def _yaml(entry: dict) -> str:
    def value(v) -> str:
        return json.dumps(v) if not isinstance(v, dict) else ""

    lines = []
    for i, (k, v) in enumerate(entry.items()):
        prefix = "- " if i == 0 else "  "
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.extend(f"    {k2}: {value(v2)}" for k2, v2 in v.items())
        else:
            lines.append(f"{prefix}{k}: {value(v)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("camera", choices=HYPERION_CAMERAS)
    parser.add_argument("--resolution", default="480", choices=sorted(GENERATED_SIZE))
    parser.add_argument("--format", default="yaml", choices=["yaml", "json"])
    args = parser.parse_args(argv)
    entry = lens(args.camera, args.resolution).as_driver_camera()
    print(_yaml(entry) if args.format == "yaml" else json.dumps(entry))


if __name__ == "__main__":
    main()

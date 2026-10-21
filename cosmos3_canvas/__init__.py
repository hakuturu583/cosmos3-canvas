"""Cosmos3-Nano-Sim-Auto as a carla_canvas backend (``renderer.backend: cosmos3_canvas:Cosmos3Renderer``).

The renderer imports carla_canvas; ``cosmos3_canvas.stream`` (the model alone) does not.
"""

__all__ = ["Cosmos3Renderer"]


def __getattr__(name: str):
    if name == "Cosmos3Renderer":
        from .renderer import Cosmos3Renderer

        return Cosmos3Renderer
    raise AttributeError(name)

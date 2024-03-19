from dataclasses import dataclass


@dataclass
class BoundingBox:
    name: str
    x_top_left: int
    y_top_left: int
    x_bottom_right: int
    y_bottom_right: int
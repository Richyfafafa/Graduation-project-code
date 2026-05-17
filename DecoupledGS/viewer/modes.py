from enum import IntEnum


class RenderMode(IntEnum):
    COLOR = 0
    GAUSSIAN = 1
    ELLIPSOID = 2
    STRENGTH = 3
    BASE_COLOR = 4
    REFL_COLOR = 5
    NORMAL = 6


MODE_ORDER = [
    RenderMode.COLOR,
    RenderMode.GAUSSIAN,
    RenderMode.ELLIPSOID,
    RenderMode.STRENGTH,
    RenderMode.BASE_COLOR,
    RenderMode.REFL_COLOR,
    RenderMode.NORMAL,
]


MODE_NAME = {
    RenderMode.COLOR: "color(final)",
    RenderMode.GAUSSIAN: "gaussian(c3)",
    RenderMode.ELLIPSOID: "ellipsoid(debug)",
    RenderMode.STRENGTH: "refl_strength",
    RenderMode.BASE_COLOR: "base_color",
    RenderMode.REFL_COLOR: "refl_color",
    RenderMode.NORMAL: "normal",
}


KEY_TO_MODE = {
    ord("1"): RenderMode.COLOR,
    ord("2"): RenderMode.GAUSSIAN,
    ord("3"): RenderMode.ELLIPSOID,
    ord("4"): RenderMode.STRENGTH,
    ord("5"): RenderMode.BASE_COLOR,
    ord("6"): RenderMode.REFL_COLOR,
    ord("7"): RenderMode.NORMAL,
}

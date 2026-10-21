import inspect
import math
from dataclasses import dataclass

import cv2
import numpy as np


class ParamError(ValueError):
    """Raised when the parameters of an augmentation are not valid."""


@dataclass
class Augmentation:
    name: str
    func: object
    converters: tuple
    required: int


REGISTRY = {}


def register(name, *converters):
    def decorator(func):
        params = list(inspect.signature(func).parameters.values())[1:]
        required = sum(1 for p in params if p.default is p.empty)
        REGISTRY[name.lower()] = Augmentation(name, func, converters, required)
        return func
    return decorator


def lookup(name):
    return REGISTRY.get(name.lower())


def convert_params(aug, tokens):
    low, high = aug.required, len(aug.converters)
    if not low <= len(tokens) <= high:
        expected = str(low) if low == high else f"{low} to {high}"
        raise ParamError(f"{aug.name} expects {expected} parameter(s), got {len(tokens)}")
    args = []
    for conv, tok in zip(aug.converters, tokens):
        try:
            args.append(conv(tok))
        except ValueError as e:
            raise ParamError(f"{aug.name}: invalid parameter {tok!r} ({e})")
    return args


def finite_float(s):
    v = float(s)
    if not math.isfinite(v):
        raise ValueError("must be a finite number")
    return v


def positive_float(s):
    v = finite_float(s)
    if v <= 0:
        raise ValueError("must be > 0")
    return v


def positive_int(s):
    v = int(s)
    if v <= 0:
        raise ValueError("must be > 0")
    return v


def odd_int(s):
    v = int(s)
    if v < 1 or v % 2 == 0:
        raise ValueError("must be a positive odd integer")
    return v


def choice(*options):
    def convert(s):
        s = s.lower()
        if s not in options:
            raise ValueError("must be one of: " + ", ".join(options))
        return s
    return convert


@register("Dummy", float)
def dummy(img, value):
    """Returns the same image as the input. The parameter is required but
    ignored; it only tests that parameters are parsed from the config."""
    return img.copy()


# ---------------------------------------------------------------------------
# Pixel level / color  -  LOW LEVEL
# ---------------------------------------------------------------------------
@register("Brightness", int)
def brightness(img, b):
    table = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        new = v + b
        if new < 0:
            new = 0
        elif new > 255:
            new = 255
        table[v] = new
    return table[img]


@register("Contrast", positive_float)
def contrast(img, a):
    table = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        new = int(v * a + 0.5)
        if new > 255:
            new = 255
        table[v] = new
    return table[img]


@register("Tint", choice("blue", "green", "red"), int)
def tint(img, color, amount):
    channel = {"blue": 0, "green": 1, "red": 2}[color]
    table = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        new = v + amount
        if new < 0:
            new = 0
        elif new > 255:
            new = 255
        table[v] = new
    out = img.copy()
    out[:, :, channel] = table[img[:, :, channel]]
    return out


# ---------------------------------------------------------------------------
# Filtering  -  OpenCV
# ---------------------------------------------------------------------------
@register("Median", odd_int)
def median(img, k=3):
    return cv2.medianBlur(img, k)


@register("Blur", odd_int)
def blur(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)


# ---------------------------------------------------------------------------
# Geometrical / morphological  -  LOW LEVEL
# ---------------------------------------------------------------------------
@register("Flip", choice("h", "v", "hv"))
def flip(img, mode="h"):
    if mode == "h":
        return img[:, ::-1].copy()
    if mode == "v":
        return img[::-1, :].copy()
    return img[::-1, ::-1].copy()


@register("Rotation", finite_float)
def rotation(img, angle):
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = np.radians(angle)
    cos, sin = np.cos(t), np.sin(t)

    dx = np.arange(w) - cx
    dy = (np.arange(h) - cy)[:, None]

    sx = np.rint(cos * dx - sin * dy + cx).astype(np.intp) + 1
    sy = np.rint(sin * dx + cos * dy + cy).astype(np.intp) + 1
    np.clip(sx, 0, w + 1, out=sx)
    np.clip(sy, 0, h + 1, out=sy)

    pad = ((1, 1), (1, 1)) + ((0, 0),) * (img.ndim - 2)
    flat = np.pad(img, pad).reshape((h + 2) * (w + 2), *img.shape[2:])
    return flat.take(sy * (w + 2) + sx, axis=0)


# ---------------------------------------------------------------------------
# Geometrical / morphological  -  OpenCV
# ---------------------------------------------------------------------------
@register("Resize", positive_int, positive_int)
def resize(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


@register("Erode", odd_int)
def erode(img, k=3):
    return cv2.erode(img, np.ones((k, k), np.uint8))


@register("Dilate", odd_int)
def dilate(img, k=3):
    return cv2.dilate(img, np.ones((k, k), np.uint8))

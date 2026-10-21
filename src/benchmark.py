import math
import os
import sys
import time

import cv2
import numpy as np

import augmentations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------
def brightness_loop_pixels(img, b):
    h, w, c = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                v = int(img[y, x, ch]) + b
                out[y, x, ch] = 0 if v < 0 else 255 if v > 255 else v
    return out


def brightness_loop_rows(img, b):
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        out[y] = np.clip(img[y].astype(np.int32) + b, 0, 255).astype(np.uint8)
    return out


def brightness_opencv(img, b):
    return cv2.convertScaleAbs(img, alpha=1, beta=b)


# ---------------------------------------------------------------------------
# Contrast  g = a * f
# ---------------------------------------------------------------------------
def contrast_loop_pixels(img, a):
    h, w, c = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                v = int(int(img[y, x, ch]) * a + 0.5)
                out[y, x, ch] = 255 if v > 255 else v
    return out


def contrast_loop_rows(img, a):
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        out[y] = np.clip(img[y].astype(np.float64) * a + 0.5, 0, 255).astype(np.uint8)
    return out


def contrast_opencv(img, a):
    return cv2.convertScaleAbs(img, alpha=a, beta=0)


# ---------------------------------------------------------------------------
# Tint
# ---------------------------------------------------------------------------
_CHANNEL = {"blue": 0, "green": 1, "red": 2}


def tint_loop_pixels(img, color, amount):
    h, w = img.shape[:2]
    ch = _CHANNEL[color]
    out = img.copy()
    for y in range(h):
        for x in range(w):
            v = int(img[y, x, ch]) + amount
            out[y, x, ch] = 0 if v < 0 else 255 if v > 255 else v
    return out


def tint_loop_rows(img, color, amount):
    ch = _CHANNEL[color]
    out = img.copy()
    for y in range(img.shape[0]):
        out[y, :, ch] = np.clip(img[y, :, ch].astype(np.int32) + amount, 0, 255)
    return out


def tint_opencv(img, color, amount):
    scalar = [0, 0, 0, 0]
    scalar[_CHANNEL[color]] = amount
    return cv2.add(img, tuple(scalar))


# ---------------------------------------------------------------------------
# Flip horizontal
# ---------------------------------------------------------------------------
def flip_loop_pixels(img):
    h, w = img.shape[:2]
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = img[y, w - 1 - x]
    return out


def flip_loop_rows(img):
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        out[y] = img[y, ::-1]
    return out


def flip_opencv(img):
    return cv2.flip(img, 1)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
def rotation_loop_pixels(img, angle):
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = math.radians(angle)
    cos, sin = math.cos(t), math.sin(t)
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            sx = round(cos * dx - sin * dy + cx)
            sy = round(sin * dx + cos * dy + cy)
            if 0 <= sx < w and 0 <= sy < h:
                out[y, x] = img[sy, sx]
    return out


def rotation_loop_rows(img, angle):
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = np.radians(angle)
    cos, sin = np.cos(t), np.sin(t)
    xs = np.arange(w)
    out = np.zeros_like(img)
    for y in range(h):
        dx, dy = xs - cx, y - cy
        sx = np.rint(cos * dx - sin * dy + cx).astype(np.int64)
        sy = np.rint(sin * dx + cos * dy + cy).astype(np.int64)
        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        out[y, valid] = img[sy[valid], sx[valid]]
    return out


def rotation_opencv(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST)


def timeit(func, *args, repeat=1):
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def report(title, reference, runs):
    print(f"\n{title}")
    print(f"{'method':<28}{'time [ms]':>12}{'speed-up':>11}{'max |diff|':>12}")
    base_time = None
    for label, func, args, repeat in runs:
        ms, res = timeit(func, *args, repeat=repeat)
        base_time = base_time or ms
        diff = int(np.abs(res.astype(int) - reference.astype(int)).max())
        print(f"{label:<28}{ms:>12.2f}{base_time / ms:>10.0f}x{diff:>12}")


def load_image():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        test_dir = os.path.join(BASE_DIR, "test")
        if os.path.isdir(test_dir):
            jpgs = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(".jpg"))
            if jpgs:
                path = os.path.join(test_dir, jpgs[0])
    img = cv2.imread(path) if path else None
    if img is None:
        print("No image found - using a random 640x480 image")
        img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    else:
        print(f"Image: {path}  shape={img.shape}")
    return img


def main():
    img = load_image()
    b, a, color, amount, angle = 40, 0.75, "blue", 50, 15

    ref = augmentations.brightness(img, b)
    report(f"Brightness {b}", ref, [
        ("loops over pixels", brightness_loop_pixels, (img, b), 1),
        ("loop over rows + numpy", brightness_loop_rows, (img, b), 5),
        ("lookup table (used)", augmentations.brightness, (img, b), 20),
        ("OpenCV convertScaleAbs", brightness_opencv, (img, b), 20),
    ])

    ref = augmentations.contrast(img, a)
    report(f"Contrast {a}", ref, [
        ("loops over pixels", contrast_loop_pixels, (img, a), 1),
        ("loop over rows + numpy", contrast_loop_rows, (img, a), 5),
        ("lookup table (used)", augmentations.contrast, (img, a), 20),
        ("OpenCV convertScaleAbs", contrast_opencv, (img, a), 20),
    ])

    ref = augmentations.tint(img, color, amount)
    report(f"Tint {color} {amount}", ref, [
        ("loops over pixels", tint_loop_pixels, (img, color, amount), 1),
        ("loop over rows + numpy", tint_loop_rows, (img, color, amount), 5),
        ("lookup table (used)", augmentations.tint, (img, color, amount), 20),
        ("OpenCV add", tint_opencv, (img, color, amount), 20),
    ])

    ref = augmentations.flip(img, "h")
    report("Flip H", ref, [
        ("loops over pixels", flip_loop_pixels, (img,), 1),
        ("loop over rows + numpy", flip_loop_rows, (img,), 5),
        ("numpy vectors (used)", augmentations.flip, (img, "h"), 20),
        ("OpenCV flip", flip_opencv, (img,), 20),
    ])

    ref = augmentations.rotation(img, angle)
    report(f"Rotation {angle}", ref, [
        ("loops over pixels", rotation_loop_pixels, (img, angle), 1),
        ("loop over rows + numpy", rotation_loop_rows, (img, angle), 20),
        ("numpy vectors (used)", augmentations.rotation, (img, angle), 20),
        ("OpenCV warpAffine", rotation_opencv, (img, angle), 20),
    ])
    print("\nmax |diff| is measured against the version used in augmentations.py "
          "(0 = identical result).")


if __name__ == "__main__":
    main()

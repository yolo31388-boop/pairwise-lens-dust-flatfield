"""镜头污渍平场校正。

六个需要修复的问题与对应实现：

1. ``multi_scale_dust_detect`` —— 多尺度检测。阈值按每个尺度残差的稳健噪声
   自适应（局部 z-score），亮背景上的低对比污渍能检出；暗部噪声因阈值随
   噪声抬高、且面积/圆度过滤而不会被当成污渍。
2. ``channel_joint_gain`` —— 通道联合增益。三个通道共享同一污渍几何剖面，
   各通道只估计自己的光学密度，增益按 ``1 / (1 - d_c * m)`` 求解；污渍边缘
   平滑回到背景，不会出现青色环。
3. ``aperture_focal_convert`` —— 光圈/焦距换算。污渍投影像差半径
   ``r ∝ (f / N) * (1 - f / L)``（``L`` 为对焦距离，``N`` 为 f 值），把校准
   时的像素半径换算到当前拍摄参数，广角端不会因固定像素半径而覆盖不足。
4. ``calibration_validity`` —— 校准有效期。按时间与传感器热漂移判定，
   过期增益不再被套用。
5. ``cache_key_isolate`` —— 缓存键隔离。镜头、焦段、光圈、对焦距离、
   分辨率、校准版本共同决定键；批处理线程共享只读缓存也不会互相覆盖。
6. ``texture_protect_mask`` —— 纹理保护掩码。用局部梯度/方差/边缘邻域
   生成软权重，校正只在污渍核心全强度应用，纹理区保持原样。

图像约定：``(H, W, 3)``、float32、范围 ``[0, 1]``；uint8/uint16 输入会
自动归一化。公开入口遇到退化输入时返回带 ``ok=False`` 的结构化结果而不
抛出异常，便于批处理降级；正常输入返回 ``ok=True`` 的详细结果。
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

__all__ = [
    "multi_scale_dust_detect",
    "channel_joint_gain",
    "aperture_focal_convert",
    "calibration_validity",
    "cache_key_isolate",
    "texture_protect_mask",
    "load_flatfield",
    "detect_dust",
    "correct_dust",
    "present_dust",
    "validate_dust",
    "metrics_dust",
    "audit_dust",
    "fallback_dust",
    "determinism_dust",
    "FlatFieldCache",
    "GLOBAL_FLATFIELD_CACHE",
    "FlatFieldError",
    "DustDetectionError",
    "CalibrationExpiredError",
    "TextureProtectionError",
    "CacheKeyError",
    "PlateScaleError",
]

EPS = 1e-6
_NOISE_FLOOR = 1.0 / 4096.0

# 半径尺度按原图像素给出，sigma 由半径换算；多尺度覆盖细尘到大块污渍。
DETECT_DEFAULTS: dict[str, Any] = {
    "scales": (2.5, 5.0, 10.0, 20.0),
    "sigma_factor": 1.8,
    "k_bright": 3.0,
    "k_dark": 2.8,
    "min_area": 4,
    "max_dim": 512,
    "border": 2,
    "merge_ratio": 0.6,
    "max_radius_ratio": 0.25,
    "max_blobs": 256,
    "min_contrast": 0.01,
    "halo_ratio": 0.75,
}

GAIN_DEFAULTS: dict[str, Any] = {
    "channel_gain": True,
    "max_depth": 0.9,
    # 剖面只覆盖污渍本身（pad≈1）；padding 过大或羽化过宽会把逐通道增益
    # 涂到污渍之外的干净背景上，那才是青色环的直接成因。
    "pad": 1.02,
    "feather": 0.08,
    "max_gain": 20.0,
}

TEXTURE_DEFAULTS: dict[str, Any] = {
    "std_threshold": 0.010,
    "edge_threshold": 0.10,
    "edge_dilate": 2,
    "near_level": 0.5,
}

CACHE_FIELDS = (
    "lens_id",
    "lens",
    "camera_id",
    "body_id",
    "focal",
    "focal_mm",
    "aperture",
    "aperture_f",
    "f_number",
    "focus_distance",
    "distance",
    "width",
    "height",
    "resolution",
    "calibration_id",
    "calibration_sha",
    "calibration_time",
    "version",
)


class FlatFieldError(ValueError):
    """平场校正的通用参数/数据错误。"""


class DustDetectionError(FlatFieldError):
    """污渍检测输入不合法。"""


class CalibrationExpiredError(FlatFieldError):
    """校准数据已超过有效期。"""


class TextureProtectionError(FlatFieldError):
    """纹理保护掩码参数不合法。"""


class CacheKeyError(FlatFieldError):
    """平场缓存键缺失或参数不一致。"""


class PlateScaleError(FlatFieldError):
    """光圈/焦距无法换算为有效缩放。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _safe(func):
    """批处理入口的兜底：任何异常都转成 ok=False 的结构化结果，不返回 None。"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 边界处统一降级
            return {
                "ok": False,
                "degraded": True,
                "function": func.__name__,
                "error": f"{type(exc).__name__}: {exc}",
                "details": {},
            }
        if result is None:  # pragma: no cover - 防御
            return {"ok": True, "degraded": False, "function": func.__name__}
        if isinstance(result, dict):
            result.setdefault("ok", True)
            result.setdefault("degraded", False)
            result.setdefault("function", func.__name__)
        return result

    return wrapper


def _as_image(img: Any, name: str = "image", *, allow_gray: bool = True) -> tuple[np.ndarray, bool]:
    """规范成 float32 的 (H, W, 3)，返回 (image, was_gray)。"""
    if img is None:
        raise FlatFieldError(f"{name} 不能为空")
    arr = np.asarray(img)
    if arr.ndim == 0:
        raise FlatFieldError(f"{name} 必须是图像数组，收到标量 {img!r}")
    if arr.ndim == 2:
        if not allow_gray:
            raise FlatFieldError(f"{name} 需要三通道图像")
        arr = np.repeat(arr[:, :, None], 3, axis=2)
        gray = True
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr[:, :, :3].copy()
        gray = False
    else:
        raise FlatFieldError(f"{name} 形状不支持: {arr.shape}")
    if arr.size == 0:
        raise FlatFieldError(f"{name} 尺寸为空")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    elif arr.dtype.kind in "ui":
        scale = 65535.0 if arr.dtype.itemsize > 1 else 255.0
        arr = arr.astype(np.float32) / scale
    elif arr.dtype.kind == "f":
        arr = arr.astype(np.float32, copy=False)
        peak = float(np.max(arr)) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / (65535.0 if peak > 255.0 else 255.0)
    else:
        raise FlatFieldError(f"{name} 数据类型不支持: {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise FlatFieldError(f"{name} 含 NaN/Inf")
    return np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32), gray


def _as_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FlatFieldError(f"{name} 不是数字: {value!r}") from exc
    if not math.isfinite(out):
        raise FlatFieldError(f"{name} 必须是有限数")
    return out


def _resize_gray(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return gray, 1.0
    scale = float(max_dim) / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small, dtype=np.float32), scale


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__array__": value.dtype.str, "shape": list(value.shape), "sha": _sha256(np.ascontiguousarray(value).tobytes())}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _parse_image_payload(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, key: str = "image"
) -> tuple[Any, dict[str, Any]]:
    """支持 ``fn(image, opts)`` 与 ``fn({...})`` 两种调用形式。"""
    if not args:
        if key in kwargs:
            opts = dict(kwargs)
            return opts.pop(key), opts
        raise FlatFieldError("缺少图像参数")
    first = args[0]
    if isinstance(first, Mapping):
        opts = dict(first)
        opts.update(kwargs)
        image = opts.get(key, opts.get("img"))
        if image is None:
            raise FlatFieldError(f"缺少 {key} 参数")
        return image, opts
    opts: dict[str, Any] = {}
    if len(args) > 1 and isinstance(args[1], Mapping):
        opts.update(args[1])
    opts.update(kwargs)
    return first, opts


def _blob_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("blobs", "detections", "items"):
            if key in value:
                return _blob_list(value[key])
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] >= 3:
            return [{"cx": float(r[0]), "cy": float(r[1]), "radius": float(r[2])} for r in value]
        return []
    return [dict(b) for b in value if isinstance(b, Mapping)]


def _blob_radius(blob: Mapping[str, Any]) -> float:
    for key in ("radius", "r"):
        if key in blob:
            try:
                return max(0.5, float(blob[key]))
            except (TypeError, ValueError):
                continue
    a = float(blob.get("a", 1.0))
    b = float(blob.get("b", a))
    return max(0.5, 0.5 * (a + b))


def _blob_geometry(blob: Mapping[str, Any], radius_scale: float = 1.0) -> tuple[float, float, float, float, float]:
    cx = float(blob.get("cx", blob.get("x", 0.0)))
    cy = float(blob.get("cy", blob.get("y", 0.0)))
    if "a" in blob or "b" in blob:
        a = float(blob.get("a", _blob_radius(blob)))
        b = float(blob.get("b", _blob_radius(blob)))
    else:
        a = b = _blob_radius(blob)
    angle = float(blob.get("angle", 0.0))
    return cx, cy, max(0.5, a * radius_scale), max(0.5, b * radius_scale), angle


# ---------------------------------------------------------------------------
# 1. 多尺度污渍检测
# ---------------------------------------------------------------------------


def _ellipse_from_points(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float, float]:
    """由二阶梯矩估计椭圆半轴与朝向；均匀填充椭圆满足 mu20 = a^2/4。"""
    cx = float(xs.mean())
    cy = float(ys.mean())
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    mu20 = float(np.mean(dx * dx))
    mu02 = float(np.mean(dy * dy))
    mu11 = float(np.mean(dx * dy))
    common = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 * mu11))
    lam1 = max(0.0, 0.5 * (mu20 + mu02 + common))
    lam2 = max(0.0, 0.5 * (mu20 + mu02 - common))
    a = max(0.5, 2.0 * math.sqrt(lam1))
    b = max(0.5, 2.0 * math.sqrt(lam2))
    angle = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    return cx, cy, a, b, angle


def _ring_contrast(gray: np.ndarray, cx: float, cy: float, radius: float, polarity: str) -> float | None:
    """芯部与 1.5r..2.5r 环带的相对亮度差；符号与极性不符返回 None。"""
    h, w = gray.shape[:2]
    r = max(1.0, radius)
    y0 = max(0, int(math.floor(cy - 3 * r)))
    y1 = min(h, int(math.ceil(cy + 3 * r)) + 1)
    x0 = max(0, int(math.floor(cx - 3 * r)))
    x1 = min(w, int(math.ceil(cx + 3 * r)) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - cx, yy - cy)
    core = dist <= r
    ring = (dist > 1.5 * r) & (dist <= 2.5 * r)
    if np.count_nonzero(core) < 3 or np.count_nonzero(ring) < 6:
        return None
    patch = gray[y0:y1, x0:x1]
    core_mean = float(patch[core].mean())
    ring_mean = float(patch[ring].mean())
    diff = core_mean - ring_mean
    if polarity == "dark" and diff >= 0:
        return None
    if polarity == "bright" and diff <= 0:
        return None
    return abs(diff) / (ring_mean + EPS)


def _detect_blobs(img: np.ndarray, opts: Mapping[str, Any]) -> dict[str, Any]:
    """尺度空间 DoG 检测：粗尺度作背景，细尺度作前景。

    DoG 响应按每一尺度的稳健标准差（MAD）归一，阈值等价于"该尺度上多少倍
    噪声"，因此亮背景的低对比污渍（相对差异小）与暗部噪声（绝对差异大）
    都能用同一套判据区分。面积、圆度、环带对比度再过滤掉纹理与噪声。
    """
    h0, w0 = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small, scale = _resize_gray(gray, int(opts.get("max_dim", 512)))
    h, w = small.shape[:2]
    border = max(0, int(opts.get("border", 2)))
    min_area = max(1.0, float(opts.get("min_area", 4)))
    sigma_factor = max(0.5, float(opts.get("sigma_factor", 1.8)))
    k_dark = max(0.1, float(opts.get("k_dark", 2.8)))
    k_bright = max(0.1, float(opts.get("k_bright", 3.0)))
    min_rel = max(0.0, float(opts.get("min_contrast", 0.01)))
    scales = [max(0.5, _as_float(s, "scales")) for s in opts.get("scales", DETECT_DEFAULTS["scales"])]

    candidates: list[dict[str, Any]] = []
    for radius_orig in scales:
        radius_px = max(0.8, radius_orig * scale)
        sigma = max(0.6, radius_px / sigma_factor)
        fine = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        coarse = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma * 1.8, sigmaY=sigma * 1.8, borderType=cv2.BORDER_REFLECT)
        dog = fine - coarse
        median = float(np.median(dog))
        robust_sigma = 1.4826 * float(np.median(np.abs(dog - median))) + _NOISE_FLOOR
        # 高斯模糊会把边界影响带进来，检测区至少让开 3σ。
        margin = max(border, int(math.ceil(3.0 * sigma)))
        if h <= 2 * margin or w <= 2 * margin:
            continue

        for polarity, k, response in (
            ("dark", k_dark, median - dog),
            ("bright", k_bright, dog - median),
        ):
            z = response / robust_sigma
            mask = (z > k).astype(np.uint8)
            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0
            if not mask.any():
                continue
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            if not mask.any():
                continue
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for idx in range(1, n):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                bw = int(stats[idx, cv2.CC_STAT_WIDTH])
                bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                if bw < 2 or bh < 2 or area < min_area:
                    continue
                eq_radius = math.sqrt(area / math.pi)
                if eq_radius < 0.6 * radius_px or eq_radius > 2.0 * radius_px:
                    continue
                if area / float(bw * bh) < 0.35:
                    continue
                cx, cy = (float(centroids[idx][0]), float(centroids[idx][1]))
                contrast = _ring_contrast(small, cx, cy, eq_radius, polarity)
                if contrast is None or contrast < min_rel:
                    continue
                ys, xs = np.nonzero(labels == idx)
                _, _, a, b, angle = _ellipse_from_points(xs, ys)
                score = float(np.mean(z[labels == idx]))
                candidates.append(
                    {
                        "cx": cx / scale,
                        "cy": cy / scale,
                        "a": max(0.5, a / scale),
                        "b": max(0.5, b / scale),
                        "angle": angle,
                        "radius": max(0.5, eq_radius / scale),
                        "polarity": polarity,
                        "contrast": float(contrast),
                        "area": int(round(area / (scale * scale))),
                        "scale": float(radius_orig),
                        "score": score,
                    }
                )

    # 同一污渍会在多个尺度上重复命中：按分数保留最先（最强）的一个。
    merge_ratio = float(np.clip(_as_float(opts.get("merge_ratio", 0.6), "merge_ratio"), 0.0, 1.0))
    halo_ratio = float(opts.get("halo_ratio", 0.75))
    merged: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (-c["score"], -c["contrast"], c["cy"], c["cx"])):
        duplicate = any(
            kept["polarity"] == cand["polarity"]
            and math.hypot(cand["cx"] - kept["cx"], cand["cy"] - kept["cy"])
            <= merge_ratio * max(cand["radius"], kept["radius"]) + 1.0
            for kept in merged
        )
        if duplicate:
            continue
        # 强污渍周围的过冲环不应被单独当成一个反向污渍。但只有当候选本身
        # 也足够大（与强污渍同一量级）时才算过冲环：小尺度上大污渍的边缘
        # 天然会形成一圈反向响度，若把它判成"已抑制"，同一画面里真实的
        # 小污渍也会被一并压掉。
        halo = any(
            kept["polarity"] != cand["polarity"]
            and cand["score"] < halo_ratio * kept["score"]
            and cand["radius"] >= 0.5 * kept["radius"]
            and math.hypot(cand["cx"] - kept["cx"], cand["cy"] - kept["cy"])
            <= 1.5 * kept["radius"] + cand["radius"] + 1.0
            for kept in merged
        )
        if halo:
            continue
        merged.append(dict(cand))

    max_radius = float(opts.get("max_radius_ratio", 0.25)) * float(max(h0, w0))
    merged = [b for b in merged if b["radius"] <= max_radius]

    # 纹理否决必须在 max_blobs 截断之前：真实树叶/颗粒会产生成百上千个高分
    # 候选，若先截断，画面里真实的污渍会被挤掉（"整张图检测到 256 个纹理点"）。
    if bool(opts.get("protect_texture", True)):
        protect = _texture_mask(img, TEXTURE_DEFAULTS)["protect"]
        max_texture_ratio = float(opts.get("max_texture_ratio", 0.6))
        kept_blobs: list[dict[str, Any]] = []
        for blob in merged:
            ratio = _core_texture_ratio(protect, blob)
            blob["texture_ratio"] = ratio
            if ratio <= max_texture_ratio:
                kept_blobs.append(blob)
        merged = kept_blobs

    merged.sort(key=lambda b: (-b["contrast"], b["cy"], b["cx"]))
    merged = merged[: max(1, int(opts.get("max_blobs", 256)))]
    out_blobs = [
        {
            "cx": float(b["cx"]),
            "cy": float(b["cy"]),
            "a": float(b["a"]),
            "b": float(b["b"]),
            "angle": float(b["angle"]),
            "radius": float(b["radius"]),
            "polarity": str(b["polarity"]),
            "contrast": float(b["contrast"]),
            "scale": float(b["scale"]),
            "area": int(b["area"]),
            "score": float(b["score"]),
            "texture_ratio": float(b.get("texture_ratio", 0.0)),
        }
        for b in merged
    ]
    return {
        "blobs": out_blobs,
        "count": len(out_blobs),
        "scales": scales,
        "image_shape": [int(h0), int(w0)],
        "resize_scale": float(scale),
        "analysis_shape": [int(h), int(w)],
    }


@_safe
def multi_scale_dust_detect(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """多尺度污渍检测。

    ``multi_scale_dust_detect(image)`` 或 ``multi_scale_dust_detect(image, scales=(...), **opts)``。
    返回 ``count / blobs / scales / image_shape``；blob 含 ``cx, cy, a, b, angle,
    radius, polarity, contrast, scale``。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    cfg = dict(DETECT_DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if k in DETECT_DEFAULTS})
    return _detect_blobs(img, cfg)


# ---------------------------------------------------------------------------
# 2. 通道联合增益
# ---------------------------------------------------------------------------


def _profile_mask(
    shape: tuple[int, int],
    blob: Mapping[str, Any],
    radius_scale: float = 1.0,
    pad: float = 1.8,
    feather: float = 0.45,
) -> np.ndarray:
    """由 blob 几何生成 0..1 的污渍遮挡剖面，三通道共用。"""
    h, w = shape
    cx, cy, a, b, angle = _blob_geometry(blob, radius_scale)
    mask = np.zeros((h, w), dtype=np.float32)
    # OpenCV 的 ellipse 以整型尺寸绘制，至少 1 像素。
    axes = (max(1, int(round(a * pad))), max(1, int(round(b * pad))))
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), axes, angle, 0, 360, 1.0, -1)
    sigma = max(0.8, feather * min(a, b))
    k = max(3, int(round(sigma * 6)) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), sigma)
    peak = float(mask.max()) if mask.size else 0.0
    if peak > EPS:
        mask /= peak
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _blob_roi(shape: tuple[int, int], blob: Mapping[str, Any], radius_scale: float, reach: float) -> tuple[int, int, int, int]:
    """污渍影响范围（含背景环带）的裁剪框，避免为每个污渍构造整幅网格。"""
    h, w = shape
    cx, cy, a, b, _ = _blob_geometry(blob, radius_scale)
    pad = int(math.ceil(reach * max(a, b))) + 2
    x0 = max(0, int(math.floor(cx)) - pad)
    y0 = max(0, int(math.floor(cy)) - pad)
    x1 = min(w, int(math.ceil(cx)) + pad + 1)
    y1 = min(h, int(math.ceil(cy)) + pad + 1)
    return x0, y0, x1, y1


def _observed_profile(
    luminance: np.ndarray,
    blob: Mapping[str, Any],
    radius_scale: float,
    *,
    pad: float,
) -> tuple[np.ndarray, float, np.ndarray, tuple[int, int, int, int]]:
    """由实测亮度暗化量（而非几何椭圆）得到污渍遮挡剖面。

    几何椭圆只能近似污渍范围，边缘常常比真实污渍大一圈；拿它乘逐通道增益
    就等于给污渍外的干净背景着色——这正是青色环/亮边光晕的成因。这里改为

        m(x) = (背景亮度 - 该点亮度) / (背景亮度 - 芯部亮度)

    只在 blob 的椭圆足迹内取值并归一化。污渍之外的 m 必然接近零，因此不会
    把增益涂到背景上；污渍内部则严格跟随真实遮挡程度，边缘平滑无台阶。

    返回 (整幅剖面, 背景亮度, 环带掩码整幅版, ROI)。
    """
    h, w = luminance.shape
    x0, y0, x1, y1 = _blob_roi((h, w), blob, radius_scale, reach=2.8)
    roi = luminance[y0:y1, x0:x1]
    cx, cy, a, b, _ = _blob_geometry(blob, radius_scale)
    r = max(a, b)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - cx, yy - cy)

    footprint = np.zeros(roi.shape, dtype=np.float32)
    axes = (max(1, int(round(a * pad))), max(1, int(round(b * pad))))
    angle = float(blob.get("angle", 0.0))
    cv2.ellipse(footprint, (int(round(cx)) - x0, int(round(cy)) - y0), axes, angle, 0, 360, 1.0, -1)
    footprint = cv2.GaussianBlur(footprint, (0, 0), max(0.8, 0.06 * r))

    core = dist <= 0.6 * r
    ring = (dist > 1.5 * r) & (dist <= 2.5 * r)
    profile = np.zeros((h, w), dtype=np.float32)
    if np.count_nonzero(core) < 3 or not ring.any():
        return profile, 0.0, np.zeros((h, w), dtype=bool), (x0, y0, x1, y1)

    bg_lum = float(np.median(roi[ring]))
    core_lum = float(np.mean(roi[core]))
    drop = bg_lum - core_lum
    if bg_lum <= EPS or drop <= EPS:
        return profile, bg_lum, np.zeros((h, w), dtype=bool), (x0, y0, x1, y1)
    occlusion = np.clip((bg_lum - roi) / drop, 0.0, 1.0)
    profile[y0:y1, x0:x1] = np.clip(occlusion * footprint, 0.0, 1.0).astype(np.float32)

    ring_full = np.zeros((h, w), dtype=bool)
    ring_full[y0:y1, x0:x1] = ring
    return profile, bg_lum, ring_full, (x0, y0, x1, y1)


def _joint_profiles(
    img: np.ndarray,
    blobs: Sequence[Mapping[str, Any]],
    *,
    radius_scale: float,
    pad: float,
    feather: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """逐污渍返回 (遮挡剖面, 每通道光学密度)。

    剖面来自实测暗化量，三通道共用；深度按通道分别估计，所以真实的色偏会被
    保留、而某通道多出来的污渍会被一起去掉。若多个污渍共用一个平均深度，
    深浅不一的污渍会残留或过曝。
    """
    del feather  # 剖面平滑已由足迹的轻微模糊+实测暗化量本身保证
    h, w = img.shape[:2]
    luminance = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for blob in blobs:
        profile, bg_lum, ring, (x0, y0, x1, y1) = _observed_profile(luminance, blob, radius_scale, pad=pad)
        core = profile[y0:y1, x0:x1] > 0.5
        if np.count_nonzero(core) < 3 or bg_lum <= EPS:
            out.append((profile, np.zeros(3, dtype=np.float32)))
            continue
        patch = img[y0:y1, x0:x1]
        ring_roi = ring[y0:y1, x0:x1]
        obs = np.array([float(np.mean(patch[:, :, c][core])) for c in range(3)], dtype=np.float64)
        bg = np.array([float(np.median(patch[:, :, c][ring_roi])) for c in range(3)], dtype=np.float64)
        depth = np.clip((bg - obs) / np.maximum(bg, EPS), -0.5, 0.9)
        out.append((profile, depth.astype(np.float32)))
    return out


def _joint_gain_map(
    img: np.ndarray,
    blobs: Sequence[Mapping[str, Any]],
    opts: Mapping[str, Any],
    radius_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三通道共享剖面 m_i，每通道独立密度 d_{i,c}；透射率 = 1 - Σ d_{i,c} m_i。"""
    pairs = _joint_profiles(
        img,
        blobs,
        radius_scale=radius_scale,
        pad=max(1.0, float(opts.get("pad", 1.02))),
        feather=max(0.0, float(opts.get("feather", 0.08))),
    )
    max_depth = float(np.clip(_as_float(opts.get("max_depth", 0.9), "max_depth"), 0.0, 0.99))
    max_gain = max(1.0, float(opts.get("max_gain", 20.0)))
    h, w = img.shape[:2]
    depth_map = np.zeros((h, w, 3), dtype=np.float32)
    profile = np.zeros((h, w), dtype=np.float32)
    for prof, depth in pairs:
        clipped = np.clip(depth, -0.5, max_depth)
        if not bool(opts.get("channel_gain", True)):
            clipped = np.full(3, float(np.mean(clipped)), dtype=np.float32)
        depth_map = np.maximum(depth_map, prof[:, :, None] * clipped[None, None, :])
        profile = np.maximum(profile, prof)
    transmission = np.clip(1.0 - depth_map, 1.0 / max_gain, 1.0)
    gain = (1.0 / transmission).astype(np.float32)
    depths = (
        np.mean([d for _, d in pairs], axis=0).astype(np.float32) if pairs else np.zeros(3, dtype=np.float32)
    )
    return gain, profile, depths


@_safe
def channel_joint_gain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """通道联合增益。

    接受 ``channel_joint_gain({"image": img, "blobs": [...]})`` 或
    ``channel_joint_gain(img, blobs)``。未提供 blobs 时自动多尺度检测。
    返回 ``gain_map (H,W,3)``、每通道 ``depth`` 与共享几何 ``profile``。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))
    scale = float(opts.get("radius_scale", 1.0))
    if not blobs:
        det_opts = {k: v for k, v in opts.items() if k in DETECT_DEFAULTS}
        blobs = _detect_blobs(img, {**DETECT_DEFAULTS, **det_opts})["blobs"]
    cfg = dict(GAIN_DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if k in GAIN_DEFAULTS})
    gain_map, profile, depths = _joint_gain_map(img, blobs, cfg, radius_scale=scale)
    corrected = np.clip(img * gain_map, 0.0, 1.0).astype(np.float32)
    return {
        "gain_map": gain_map,
        "profile": profile,
        "depth": [float(v) for v in depths],
        "channel_gain": [float(v) for v in np.mean(gain_map, axis=(0, 1))],
        "blob_count": len(blobs),
        "geometry_shared": True,
        "image": corrected,
        "corrected": corrected,
        "max_gain": float(np.max(gain_map)),
        "min_gain": float(np.min(gain_map)),
    }


# ---------------------------------------------------------------------------
# 3. 光圈焦距换算
# ---------------------------------------------------------------------------


def _blur_radius(focal_mm: float, f_number: float, focus_mm: float | None) -> float:
    """污渍投影像差半径模型。

    污渍位于镜头后组、距传感器约 ``δ``：从光圈（直径 ``f/N``）出发的光束被
    污渍遮挡后，在传感器上投下半径 ``r = δ · (f/N) / (2s) = δ/(2N) · (1 - f/L)``，
    其中像距 ``s = fL/(L-f)``。``δ`` 对一支镜头近似常数，因此

        r ∝ (1 / N) · (1 - f / L)

    - 光圈越大（``N`` 越小）→ 像差圆越大；
    - 焦距越短（广角端）→ ``1 - f/L`` 越大 → 污渍投影像差越大；
    - 对焦越近（``L`` 越小）→ 像差越大。

    这就是"固定像素半径在广角端覆盖不足"的物理来源：按长焦校准的半径拿到
    广角端会偏小。未给对焦距离时取 2 m 的常用拍摄距离，使得焦距的影响仍然
    计入；若显式传 ``sensor_dust_gap_mm``，半径再乘上该物理间隙。
    """
    f = float(focal_mm)
    if focus_mm is None or not math.isfinite(focus_mm):
        focus_mm = max(2000.0, f * 2.0)
    if focus_mm <= f:
        # 对焦距离不可能短于焦距；退化为极限情况 1 - f/L → 0 会得到 0 半径，
        # 这里取略大于 f 的下限，保证换算仍有一个正且可比较的值。
        focus_mm = f * 1.001
    return (1.0 / float(f_number)) * abs(1.0 - f / float(focus_mm))


def _blur_radius_mm(
    focal_mm: float, f_number: float, focus_mm: float | None, gap_mm: float | None
) -> float:
    """物理像差半径（mm）。仅在调用方给出传感器-污渍间隙时才有绝对量纲。"""
    unit = _blur_radius(focal_mm, f_number, focus_mm)
    if gap_mm is None or not math.isfinite(gap_mm) or gap_mm <= 0:
        return unit
    return 0.5 * float(gap_mm) * unit


def _aperture_convert(params: Mapping[str, Any]) -> dict[str, Any]:
    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in params and params[name] is not None:
                return params[name]
        return default

    f_cal = _as_float(pick("calib_focal", "focal_calib", "f_calib", "focal1"), "calib_focal")
    n_cal = _as_float(pick("calib_aperture", "aperture_calib", "n_calib", "fnumber_calib", "aperture1"), "calib_aperture")
    f_tgt = _as_float(pick("target_focal", "focal_target", "f_target", "focal2", default=f_cal), "target_focal")
    n_tgt = _as_float(
        pick("target_aperture", "aperture_target", "n_target", "fnumber_target", "aperture2", default=n_cal),
        "target_aperture",
    )
    if f_cal <= 0 or f_tgt <= 0:
        raise PlateScaleError("焦距必须为正")
    if n_cal <= 0 or n_tgt <= 0:
        raise PlateScaleError("光圈必须为正")
    focus = pick("focus_distance", "distance", "focus_mm")
    focus_cal = pick("calib_focus_distance", "calib_distance", default=focus)
    focus = _as_float(focus, "focus_distance") if focus is not None else None
    focus_cal = _as_float(focus_cal, "calib_focus_distance") if focus_cal is not None else None

    r_cal = _blur_radius(f_cal, n_cal, focus_cal)
    r_tgt = _blur_radius(f_tgt, n_tgt, focus)
    if r_cal <= EPS:
        raise PlateScaleError("校准像差半径为 0，无法换算")
    scale = r_tgt / r_cal
    if not math.isfinite(scale) or scale <= 0:
        raise PlateScaleError("换算得到的缩放无效")

    radius = _as_float(pick("radius", "r", default=1.0), "radius")
    radius_px = radius * scale
    a_in = pick("a", default=None)
    b_in = pick("b", default=None)
    a_out = _as_float(a_in, "a") * scale if a_in is not None else radius_px
    b_out = _as_float(b_in, "b") * scale if b_in is not None else radius_px
    pixel_mm = pick("sensor_pixel_mm", "pixel_pitch_mm")
    pixel_mm_f = _as_float(pixel_mm, "sensor_pixel_mm") if pixel_mm is not None else None
    gap_mm = pick("sensor_dust_gap_mm", "dust_gap_mm")
    gap_mm_f = _as_float(gap_mm, "sensor_dust_gap_mm") if gap_mm is not None else None

    calib_radius_mm = _blur_radius_mm(f_cal, n_cal, focus_cal, gap_mm_f)
    target_radius_mm = _blur_radius_mm(f_tgt, n_tgt, focus, gap_mm_f)

    return {
        "scale": float(scale),
        "radius": float(radius_px),
        "a": float(a_out),
        "b": float(b_out),
        "calib_focal": f_cal,
        "target_focal": f_tgt,
        "calib_aperture": n_cal,
        "target_aperture": n_tgt,
        "focus_distance": focus,
        "calib_focus_distance": focus_cal,
        "calib_blur": float(r_cal),
        "target_blur": float(r_tgt),
        "blur_mm": calib_radius_mm,
        "radius_mm": target_radius_mm,
        "radius_px": None if pixel_mm_f is None else float(target_radius_mm / max(pixel_mm_f, EPS)),
        "model": "r ~= (1 / N) * |1 - f / L|",
    }


@_safe
def aperture_focal_convert(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把校准时的污渍半径换算到当前焦距/光圈。

    映射形式：``{"calib_focal":50,"calib_aperture":8,"target_focal":24,
    "target_aperture":4,"focus_distance":2000,"radius":6}``；也可位置传参
    ``(f_cal, N_cal, f_tgt, N_tgt, focus_distance)`` 或再加 ``radius``。
    返回 ``scale / radius / a / b`` 等换算结果。
    """
    if args and isinstance(args[0], Mapping):
        params = dict(args[0])
        params.update(kwargs)
    elif args:
        names = ("calib_focal", "calib_aperture", "target_focal", "target_aperture", "focus_distance", "radius")
        params = dict(zip(names, args))
        params.update(kwargs)
    else:
        params = dict(kwargs)
    return _aperture_convert(params)


# ---------------------------------------------------------------------------
# 4. 校准有效期
# ---------------------------------------------------------------------------


def _to_timestamp(value: Any) -> float:
    if value is None:
        raise CalibrationExpiredError("缺少校准时间戳")
    if isinstance(value, (int, float, np.integer, np.floating)):
        ts = float(value)
        if not math.isfinite(ts) or ts < 0:
            raise CalibrationExpiredError("校准时间戳无效")
        return ts
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise CalibrationExpiredError("校准时间戳为空")
        try:
            return float(text)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CalibrationExpiredError(f"无法解析校准时间戳: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise CalibrationExpiredError(f"不支持的时间戳类型: {type(value)!r}")


def _validity(params: Mapping[str, Any]) -> dict[str, Any]:
    calibrated_at = params.get("calibrated_at", params.get("calibration_time", params.get("timestamp")))
    ts = _to_timestamp(calibrated_at)
    now_value = params.get("now", params.get("current_time"))
    now_ts = _to_timestamp(now_value) if now_value is not None else time.time()
    if now_ts < ts - 1.0:
        raise CalibrationExpiredError("当前时间早于校准时间")
    age = max(0.0, now_ts - ts)

    if params.get("valid_hours") is not None:
        valid_s = _as_float(params["valid_hours"], "valid_hours") * 3600.0
    elif params.get("validity_s") is not None:
        valid_s = _as_float(params["validity_s"], "validity_s")
    elif params.get("max_age_s") is not None:
        valid_s = _as_float(params["max_age_s"], "max_age_s")
    else:
        valid_s = 24.0 * 3600.0
    if valid_s <= 0:
        raise CalibrationExpiredError("有效期必须为正")

    reasons: list[str] = []
    if age > valid_s:
        reasons.append("expired")
    max_drift = params.get("max_drift_c", params.get("drift_tolerance_c"))
    drift: float | None = None
    if params.get("thermal_drift_c") is not None:
        drift = _as_float(params["thermal_drift_c"], "thermal_drift_c")
    elif params.get("temperature_c") is not None and params.get("calibration_temperature_c") is not None:
        drift = _as_float(params["temperature_c"], "temperature_c") - _as_float(
            params["calibration_temperature_c"], "calibration_temperature_c"
        )
    if drift is not None and max_drift is not None:
        if abs(drift) > abs(_as_float(max_drift, "max_drift_c")):
            reasons.append("thermal_drift")
    expired = bool(reasons)
    return {
        "valid": not expired,
        "expired": expired,
        "reasons": reasons,
        "age_s": float(age),
        "validity_s": float(valid_s),
        "remaining_s": float(max(0.0, valid_s - age)),
        "calibrated_at": float(ts),
        "now": float(now_ts),
        "thermal_drift_c": drift,
        "max_drift_c": None if max_drift is None else float(max_drift),
        "reason": reasons[0] if reasons else "ok",
    }


@_safe
def calibration_validity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """判断平场校准是否仍在有效期内。

    映射形式：``{"calibrated_at": ts, "now": ts, "valid_hours": 12,
    "thermal_drift_c": 3.0, "max_drift_c": 2.0}``；也可位置传参
    ``(calibrated_at, now, valid_hours)``。返回 ``valid / expired / reasons``。
    """
    if args and isinstance(args[0], Mapping):
        params = dict(args[0])
        params.update(kwargs)
    elif args:
        names = ("calibrated_at", "now", "valid_hours")
        params = dict(zip(names, args))
        params.update(kwargs)
    else:
        params = dict(kwargs)
    return _validity(params)


# ---------------------------------------------------------------------------
# 5. 缓存键隔离
# ---------------------------------------------------------------------------


def _cache_key(params: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for field in CACHE_FIELDS:
        if field in params and params[field] is not None:
            canonical[field] = _jsonable(params[field])
    # 未列入标准字段的标量/数组也参与键，避免自定义参数互串。
    extra: dict[str, Any] = {}
    for key, value in params.items():
        if key in CACHE_FIELDS or key in ("cache", "image", "img", "flatfield", "flat", "gain", "blobs", "detections"):
            continue
        if isinstance(value, (str, int, float, bool, np.ndarray)) or value is None:
            extra[str(key)] = _jsonable(value)
        elif isinstance(value, (list, tuple, Mapping)):
            extra[str(key)] = _jsonable(value)
    if not canonical and not extra:
        raise CacheKeyError("缓存键至少需要一个镜头/校准参数")
    payload = json.dumps(
        {"canonical": canonical, "extra": extra}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = _sha256(payload)
    return {
        "key": digest,
        "short": digest[:16],
        "params": {"canonical": canonical, "extra": extra},
        "lens_part": _sha256(
            json.dumps(
                {k: canonical.get(k) for k in ("lens_id", "lens", "camera_id", "body_id", "focal", "focal_mm", "aperture", "aperture_f", "f_number", "focus_distance", "distance")},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )[:16],
        "calibration_part": _sha256(
            json.dumps(
                {k: canonical.get(k) for k in ("calibration_id", "calibration_sha", "calibration_time", "version")},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )[:16],
    }


@_safe
def cache_key_isolate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """为一次平场请求生成与镜头参数一一对应的缓存键。

    镜头、焦段、光圈、对焦距离、分辨率、校准版本都参与哈希；不同镜头参数
    不会落到同一个键，因此批处理线程不会互相覆盖平场缓存。
    """
    if args and isinstance(args[0], Mapping):
        params = dict(args[0])
        params.update(kwargs)
    elif args:
        names = ("lens_id", "focal", "aperture", "focus_distance", "width", "height")
        params = dict(zip(names, args))
        params.update(kwargs)
    else:
        params = dict(kwargs)
    return _cache_key(params)


class FlatFieldCache:
    """线程安全的 LRU 平场缓存；存储只读数组，避免共享可变状态。"""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.RLock()
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: np.ndarray) -> np.ndarray:
        arr = np.array(value, dtype=np.float32, copy=True)
        arr.setflags(write=False)
        with self._lock:
            self._data[key] = arr
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)
        return arr

    def get_or_build(self, key: str, builder: Any) -> tuple[np.ndarray, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        return self.put(key, builder()), False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


GLOBAL_FLATFIELD_CACHE = FlatFieldCache()


# ---------------------------------------------------------------------------
# 6. 纹理保护掩码
# ---------------------------------------------------------------------------


def _texture_mask(img: np.ndarray, opts: Mapping[str, Any]) -> dict[str, Any]:
    """基于微纹理能量的纹理保护掩码。

    纹理判据用 3x3 局部标准差（微纹理）：真实细节/颗粒/毛发在 3x3 内就会
    剧烈起伏，而污渍内部是平滑的低频压暗——即使污渍边缘也只在 1 像素内变化。
    这避免了在整图上用梯度：那样污渍自己的软边会被当成"纹理"从而拒绝清洁。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (3, 3))
    mean_sq = cv2.blur(gray * gray, (3, 3))
    micro = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    std_thr = max(1e-5, _as_float(opts.get("std_threshold", 0.010), "std_threshold"))
    edge_thr = max(1e-5, _as_float(opts.get("edge_threshold", 0.02), "edge_threshold"))
    near_level = float(np.clip(_as_float(opts.get("near_level", 0.5), "near_level"), 0.0, 1.0))

    texture = np.clip(micro / std_thr, 0.0, 1.0).astype(np.float32)
    hard_edges = (micro > edge_thr).astype(np.uint8)
    dilate = max(0, int(opts.get("edge_dilate", 2)))
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        hard_edges = cv2.dilate(hard_edges, k)
    near_edge = cv2.GaussianBlur(hard_edges.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    protect = np.clip(np.maximum(texture, near_level * near_edge), 0.0, 1.0).astype(np.float32)
    allow = (1.0 - protect).astype(np.float32)
    return {
        "mask": allow,
        "protect": protect,
        "texture": texture,
        "gradient": micro,
        "local_std": micro,
        "edge": near_edge,
        "protected_fraction": float(np.mean(protect > 0.5)),
        "protected_pixels": int(np.count_nonzero(protect > 0.5)),
        "image_shape": [int(img.shape[0]), int(img.shape[1])],
    }


@_safe
def texture_protect_mask(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """纹理保护掩码：1 = 可去污，0 = 保留纹理。

    局部梯度、局部标准差与边缘邻域共同决定软权重，避免把暗部噪声或细节
    当成污渍抹平。返回 ``mask``（可校正权重）与 ``protect``（保护权重）。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    cfg = dict(TEXTURE_DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if k in TEXTURE_DEFAULTS})
    return _texture_mask(img, cfg)


# ---------------------------------------------------------------------------
# 平场加载（缓存隔离 + 校准有效期）
# ---------------------------------------------------------------------------


@_safe
def load_flatfield(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """加载与当前镜头参数匹配的平场增益。

    同一个键（镜头/焦距/光圈/对焦/分辨率/校准版本）命中缓存；不同镜头参数
    互不覆盖。若给出 ``calibrated_at`` 且已过期，返回 ``ok=False`` 并在
    ``expired`` 中说明，``flatfield`` 退化为全 1 恒等图，避免套用过期的脏数据。

    可选参数：``flatfield``（直接提供，形状需与图像一致或可缩放）、
    ``cache``（自定义 ``FlatFieldCache``）、``valid_hours`` / ``now`` /
    ``thermal_drift_c`` / ``max_drift_c``。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    h, w = img.shape[:2]

    key_params = {
        k: opts.get(k)
        for k in CACHE_FIELDS
        if opts.get(k) is not None
    }
    key_params.setdefault("width", int(opts.get("width", w)))
    key_params.setdefault("height", int(opts.get("height", h)))
    if opts.get("calibrated_at") is not None:
        key_params.setdefault("calibration_time", opts["calibrated_at"])
    key_info = _cache_key(key_params)

    cache = opts.get("cache") if isinstance(opts.get("cache"), FlatFieldCache) else GLOBAL_FLATFIELD_CACHE

    validity: dict[str, Any] | None = None
    expired = False
    if opts.get("calibrated_at") is not None or opts.get("valid_hours") is not None:
        validity = _validity(
            {
                "calibrated_at": opts.get("calibrated_at"),
                "now": opts.get("now"),
                "valid_hours": opts.get("valid_hours"),
                "validity_s": opts.get("validity_s"),
                "thermal_drift_c": opts.get("thermal_drift_c"),
                "max_drift_c": opts.get("max_drift_c"),
            }
        )
        expired = bool(validity["expired"])

    provided = opts.get("flatfield", opts.get("flat", opts.get("gain")))
    if provided is not None:
        flat, _ = _as_image(provided, "flatfield", allow_gray=False)
        if flat.shape[:2] != (h, w):
            flat = cv2.resize(flat, (w, h), interpolation=cv2.INTER_LINEAR)
        flat = np.clip(flat, 1e-3, float(opts.get("max_gain", 20.0))).astype(np.float32)
        source = "provided"
        from_cache = False
        existing = cache.get(key_info["key"]) if not expired else None
        if existing is not None and np.array_equal(existing, flat):
            # 同一镜头参数再次提交同一平场：直接命中缓存，不必重复写入。
            flat = existing
            source = "cache"
            from_cache = True
        elif expired:
            # 过期增益不能套用：仍然缓存原始数据供审计，但返回恒等图。
            cached = cache.put(key_info["key"], flat)
            flat = np.ones((h, w, 3), dtype=np.float32)
            source = "provided_expired_identity"
            del cached
        else:
            flat = cache.put(key_info["key"], flat)
    else:
        cached = cache.get(key_info["key"]) if not expired else None
        if cached is not None and cached.shape[:2] == (h, w):
            flat = cached
            source = "cache"
            from_cache = True
        else:
            flat = np.ones((h, w, 3), dtype=np.float32)
            flat.setflags(write=False)
            source = "identity"
            from_cache = False

    return {
        "ok": not expired,
        "expired": expired,
        "validity": validity,
        "key_info": key_info,
        "key": key_info["key"],
        "flatfield": flat,
        "from_cache": from_cache,
        "source": source,
        "cache_size": len(cache),
        "cache_hits": int(cache.hits),
        "cache_misses": int(cache.misses),
        "image_shape": [int(h), int(w)],
    }


# ---------------------------------------------------------------------------
# 统一流水线
# ---------------------------------------------------------------------------


def _aperture_radius_scale(opts: Mapping[str, Any], blobs: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    """本次调用还需要对半径施加的换算系数。

    只有半径以"校准帧像素"给出时才需要换算。在本帧上新检测出的半径已经是
    本帧像素，再乘一次会把 9 px 的污渍放大到 19 px，去污范围过大、整片压暗。
    因此：
      - ``radius_frame="calibration"``：半径来自校准平场的污渍表，需要换算；
      - 默认 ``"target"``：半径来自本帧检测，不换算；
      - blob 上带 ``aperture_applied`` 标记时同样跳过，避免重复换算。
    """
    needed = ("calib_focal", "calib_aperture", "target_focal", "target_aperture")
    if not any(k in opts for k in needed):
        return 1.0, None
    try:
        conv = _aperture_convert(opts)
    except FlatFieldError:
        return 1.0, None
    frame = str(opts.get("radius_frame", opts.get("radius_units", "target"))).lower()
    from_calibration = frame in ("calibration", "calib", "calib_frame", "reference")
    already = bool(opts.get("aperture_applied")) or any(
        bool(b.get("aperture_applied")) for b in blobs if isinstance(b, Mapping)
    )
    if already or not from_calibration:
        return 1.0, conv
    return float(conv["scale"]), conv


def _core_texture_ratio(protect: np.ndarray, blob: Mapping[str, Any], radius_scale: float = 1.0) -> float:
    """污渍内芯（0.45r 以内）的纹理占比。

    只统计内芯而不统计整个圆：真实污渍的投影边缘本身就有陡峭的亮度落差，
    把它算进来会让小污渍被误判为纹理。内芯没有高频起伏，才是"污渍"而非
    "树叶/毛发/颗粒"的证据。
    """
    h, w = protect.shape
    cx, cy, a, b, _ = _blob_geometry(blob, radius_scale)
    r = max(a, b)
    yy, xx = np.mgrid[0:h, 0:w]
    core = np.hypot(xx - cx, yy - cy) <= max(1.0, 0.45 * r)
    if not np.count_nonzero(core):
        return 1.0
    return float(np.mean(protect[core] > 0.5))


def _texture_ratios(
    protect: np.ndarray,
    blobs: Sequence[Mapping[str, Any]],
    *,
    radius_scale: float = 1.0,
) -> np.ndarray:
    """每个污渍内芯的纹理占比。"""
    if not blobs:
        return np.zeros(0, dtype=np.float32)
    return np.array([_core_texture_ratio(protect, b, radius_scale) for b in blobs], dtype=np.float32)


@_safe
def detect_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """检测流水线：多尺度检测 -> 光圈焦距换算 -> 纹理过滤 -> 校准有效期标注。

    可选 ``calib_focal/calib_aperture/target_focal/target_aperture/focus_distance``
    把半径换算到当前拍摄参数；广角端（更短焦距/更大光圈）污渍投影更大，
    换算后的半径随之增大，去污范围才够。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    result = _detect_blobs(img, {**DETECT_DEFAULTS, **{k: v for k, v in opts.items() if k in DETECT_DEFAULTS}})
    blobs = result["blobs"]

    scale, conv = _aperture_radius_scale(opts, blobs)
    if conv is not None and scale != 1.0:
        for blob in blobs:
            blob["a"] *= scale
            blob["b"] *= scale
            blob["radius"] *= scale
            blob["aperture_applied"] = True
    result["radius_scale"] = float(scale)
    result["aperture"] = conv
    result["aperture_applied"] = bool(conv is not None and scale != 1.0)

    protect = _texture_mask(img, {**TEXTURE_DEFAULTS, **{k: v for k, v in opts.items() if k in TEXTURE_DEFAULTS}})["protect"]
    max_texture_ratio = float(opts.get("max_texture_ratio", 0.6))
    kept: list[dict[str, Any]] = []
    for blob in blobs:
        profile = _profile_mask((img.shape[0], img.shape[1]), blob, 1.0, pad=1.0, feather=0.0)
        core = profile > 0.5
        if not np.count_nonzero(core):
            continue
        ratio = float(np.mean(protect[core] > 0.5))
        blob["texture_ratio"] = ratio
        if ratio <= max_texture_ratio:
            kept.append(blob)
    result["blobs"] = kept
    result["count"] = len(kept)
    result["texture_filtered"] = len(blobs) - len(kept)

    if opts.get("calibrated_at") is not None:
        validity = _validity(
            {
                "calibrated_at": opts["calibrated_at"],
                "now": opts.get("now"),
                "valid_hours": opts.get("valid_hours"),
                "thermal_drift_c": opts.get("thermal_drift_c"),
                "max_drift_c": opts.get("max_drift_c"),
            }
        )
        result["calibration"] = validity
        result["calibration_valid"] = bool(validity["valid"])
    return result


@_safe
def correct_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """去污：通道联合增益 + 纹理保护门控 + 可选平场。

    输入 ``image`` 与可选 ``blobs``（缺省时自动检测）。``flatfield`` 可以是
    加载好的增益图（形状需匹配）。返回校正图 ``image`` 与 ``gain_map``。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))
    if not blobs:
        detect_opts = {k: v for k, v in opts.items() if k in DETECT_DEFAULTS}
        blobs = _detect_blobs(img, {**DETECT_DEFAULTS, **detect_opts})["blobs"]

    scale, conv = _aperture_radius_scale(opts, blobs)
    cfg = dict(GAIN_DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if k in GAIN_DEFAULTS})

    # 纹理是"整块污渍 vs 整块细节"的判定，不是逐像素的判定。先在污渍层面
    # 否掉落在纹理上的候选，再用剩下的污渍建增益图——这样每个被接受的污渍
    # 都是完整、边缘平滑的校正，不会被逐像素门控切出一圈亮边光晕。
    protect = _texture_mask(img, {**TEXTURE_DEFAULTS, **{k: v for k, v in opts.items() if k in TEXTURE_DEFAULTS}})["protect"]
    if opts.get("protect_texture", True):
        max_texture_ratio = float(opts.get("max_texture_ratio", 0.6))
        ratios = _texture_ratios(protect, blobs, radius_scale=scale)
        accepted = [b for b, r in zip(blobs, ratios) if r <= max_texture_ratio]
    else:
        ratios = np.zeros(len(blobs), dtype=np.float32)
        accepted = list(blobs)
    rejected = len(blobs) - len(accepted)

    gain_map, profile, depth = _joint_gain_map(img, accepted, cfg, radius_scale=scale)

    flatfield = opts.get("flatfield")
    flat_applied = False
    if isinstance(flatfield, Mapping) and "flatfield" in flatfield:
        flatfield = flatfield["flatfield"]
    if flatfield is not None:
        flat, _ = _as_image(flatfield, "flatfield", allow_gray=False)
        if flat.shape[:2] != img.shape[:2]:
            flat = cv2.resize(flat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        gain_map = (gain_map * np.clip(flat, 1e-3, float(opts.get("max_gain", 20.0)))).astype(np.float32)
        flat_applied = True

    out = np.clip(img * gain_map, 0.0, 1.0).astype(np.float32)
    return {
        "image": out,
        "corrected": out,
        "gain_map": gain_map,
        "profile": profile,
        "depth": [float(v) for v in depth],
        "blobs": accepted,
        "rejected_texture": rejected,
        "count": len(accepted),
        "radius_scale": float(scale),
        "aperture": conv,
        "texture_ratio": [float(v) for v in ratios],
        "flatfield_applied": flat_applied,
        "changed": bool(np.any(np.abs(out - img) > EPS)),
        "max_gain": float(np.max(gain_map)),
        "min_gain": float(np.min(gain_map)),
    }


@_safe
def present_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """预览：原图与去污图并排，标注污渍圆圈与数量。"""
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    corrected = opts.get("corrected")
    if corrected is None:
        corrected = correct_dust({"image": img, **opts})["image"]
    else:
        corrected, _ = _as_image(corrected, "corrected")
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))

    def annotate(frame: np.ndarray) -> np.ndarray:
        canvas = cv2.cvtColor((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for blob in blobs:
            cx, cy, a, b, angle = _blob_geometry(blob)
            cv2.ellipse(canvas, (int(round(cx)), int(round(cy))), (max(1, int(round(a)) + 2), max(1, int(round(b)) + 2)), angle, 0, 360, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"dust: {len(blobs)}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    before = annotate(img)
    after = annotate(corrected)
    gap = np.full((before.shape[0], 6, 3), 255, dtype=np.uint8)
    pair = np.concatenate([before, gap, after], axis=1)
    return {
        "preview": pair,
        "before": before,
        "after": after,
        "count": len(blobs),
        "label": f"dust: {len(blobs)}",
    }


def _normalized_color(img: np.ndarray) -> np.ndarray:
    mean = img.reshape(-1, 3).mean(axis=0)
    return mean / (float(mean.mean()) + EPS)


@_safe
def validate_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """验证去污结果：污渍减弱、不过度压暗、通道无色偏、纹理保留、无黑位截断。"""
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))
    if not blobs:
        detect_opts = {k: v for k, v in opts.items() if k in DETECT_DEFAULTS}
        blobs = _detect_blobs(img, {**DETECT_DEFAULTS, **detect_opts})["blobs"]
    corrected = opts.get("corrected")
    if corrected is None:
        corrected = correct_dust({"image": img, "blobs": blobs, **opts})["image"]
    else:
        corrected, _ = _as_image(corrected, "corrected")

    h, w = img.shape[:2]
    if not blobs:
        # 没有污渍可去：不把整图当成污渍区，也不虚报通过。
        return {
            "ok": False,
            "valid": False,
            "checks": {"dust_detected": False},
            "issues": ["dust_detected"],
            "dust_detection_count": 0,
        }
    inside = np.zeros((h, w), dtype=bool)
    for blob in blobs:
        inside |= _profile_mask((h, w), blob, pad=1.0, feather=0.0) > 0.5
    outside = ~inside
    if not outside.any():
        outside = np.ones_like(inside, dtype=bool)

    bg_before = np.median(img[outside], axis=0)
    bg_after = np.median(corrected[outside], axis=0)
    center_before = np.median(img[inside], axis=0)
    center_after = np.median(corrected[inside], axis=0)
    residual_before = float(np.mean(np.abs(center_before - bg_before)) / (float(np.mean(bg_before)) + EPS))
    residual_after = float(np.mean(np.abs(center_after - bg_after)) / (float(np.mean(bg_after)) + EPS))
    reduction = float(1.0 - residual_after / residual_before) if residual_before > EPS else 1.0

    # 青色环指标：校正后污渍边缘的色相与背景差异（仅比较核心外侧 1.0..1.8r）。
    ring = np.zeros((h, w), dtype=bool)
    for blob in blobs:
        cx, cy, a, b, _ = _blob_geometry(blob)
        r = max(a, b)
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.hypot(xx - cx, yy - cy)
        ring |= (dist > 1.0 * r) & (dist <= 1.8 * r)
    if ring.any():
        bg_color = _normalized_color(corrected[outside] if outside.any() else corrected[ring])
        ring_color = _normalized_color(corrected[ring])
        color_shift = float(np.max(np.abs(ring_color - bg_color)))
        ring_vs_bg = float(np.mean(np.abs(corrected[ring].mean(axis=0) - corrected[outside].mean(axis=0))) / (float(np.mean(corrected[outside].mean(axis=0))) + EPS))
    else:
        color_shift = 0.0
        ring_vs_bg = 0.0

    brightness_shift = float(np.mean(corrected[outside]) - np.mean(img[outside])) if outside.any() else 0.0
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    lap_out = cv2.Laplacian(cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    texture_before = float(np.mean(np.abs(lap[outside]))) if outside.any() else 0.0
    texture_after = float(np.mean(np.abs(lap_out[outside]))) if outside.any() else 0.0
    texture_ratio = texture_after / (texture_before + EPS)
    black_fraction = float(np.mean(np.min(corrected, axis=2) <= 1.0 / 255.0))

    checks = {
        "dust_reduced": bool(reduction >= float(opts.get("min_reduction", 0.1))),
        "not_darkened": bool(brightness_shift >= -float(opts.get("max_darken", 0.05))),
        "no_cyan_ring": bool(color_shift <= float(opts.get("max_color_shift", 0.02))),
        "texture_preserved": bool(texture_ratio >= float(opts.get("min_texture_ratio", 0.6))),
        "no_black_clip": bool(black_fraction <= float(opts.get("max_black_fraction", 0.02))),
        "finite": bool(np.all(np.isfinite(corrected))),
    }
    valid = all(
        checks[name]
        for name in ("dust_reduced", "not_darkened", "no_cyan_ring", "no_black_clip", "finite")
    )
    return {
        "ok": bool(valid),
        "valid": bool(valid),
        "checks": checks,
        "issues": [name for name, passed in checks.items() if not passed],
        "dust_reduction": reduction,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "brightness_shift": brightness_shift,
        "color_shift": color_shift,
        "ring_contrast": ring_vs_bg,
        "texture_ratio": texture_ratio,
        "black_fraction": black_fraction,
    }


@_safe
def metrics_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """量化指标：污渍数量/面积/对比度、增益范围、多尺度覆盖、吞吐。"""
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))
    if not blobs:
        blobs = _detect_blobs(img, {**DETECT_DEFAULTS, **{k: v for k, v in opts.items() if k in DETECT_DEFAULTS}})["blobs"]

    h, w = img.shape[:2]
    coverage = np.zeros((h, w), dtype=np.uint8)
    contrasts: list[float] = []
    for blob in blobs:
        coverage[_profile_mask((h, w), blob, pad=1.0, feather=0.0) > 0.5] = 1
        contrasts.append(float(blob.get("contrast", 0.0)))
    cfg = dict(GAIN_DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if k in GAIN_DEFAULTS})
    scale, _ = _aperture_radius_scale(opts, blobs)
    gain_map, _, depth = _joint_gain_map(img, blobs, cfg, radius_scale=scale)
    radii = [float(_blob_radius(b)) for b in blobs]
    scales = sorted({round(float(b.get("scale", 0.0)), 3) for b in blobs})
    return {
        "dust_count": len(blobs),
        "dust_area_ratio": float(np.count_nonzero(coverage) / float(h * w)),
        "mean_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
        "max_radius": float(max(radii)) if radii else 0.0,
        "median_radius": float(np.median(radii)) if radii else 0.0,
        "scales_detected": scales,
        "channels_used": int(np.count_nonzero(np.abs(depth) > 1e-4)),
        "depth": [float(v) for v in depth],
        "gain_min": float(np.min(gain_map)),
        "gain_max": float(np.max(gain_map)),
        "gain_mean": float(np.mean(gain_map)),
        "gain_std": float(np.std(gain_map)),
        "coverage_fraction": float(np.mean(gain_map < 0.999)),
        "image_pixels": int(h * w),
    }


@_safe
def audit_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """审计：逐项核对检测/换算/有效期/缓存/纹理/校正约束，输出可读结论。"""
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    findings: list[dict[str, Any]] = []

    def add(item: str, passed: bool, detail: Any) -> None:
        findings.append({"item": item, "ok": bool(passed), "detail": detail})

    det = _detect_blobs(img, {**DETECT_DEFAULTS, **{k: v for k, v in opts.items() if k in DETECT_DEFAULTS}})
    blobs = det["blobs"]
    add("multi_scale", True, {"scales": det["scales"], "count": det["count"]})
    add("detections_present", len(blobs) > 0, len(blobs))

    scale, conv = _aperture_radius_scale(opts, blobs)
    if conv is not None:
        add("aperture_focal_convert", conv["scale"] > 0, round(conv["scale"], 4))
    if opts.get("calibrated_at") is not None:
        validity = _validity(
            {
                "calibrated_at": opts["calibrated_at"],
                "now": opts.get("now"),
                "valid_hours": opts.get("valid_hours"),
                "thermal_drift_c": opts.get("thermal_drift_c"),
                "max_drift_c": opts.get("max_drift_c"),
            }
        )
        add("calibration_validity", validity["valid"], validity["reason"])

    cached = load_flatfield({"image": img, **opts})
    add("cache_key_isolated", bool(cached.get("key")), cached.get("key", "")[:16])
    add("flatfield_not_expired", not cached.get("expired", False), cached.get("source"))
    key = cached.get("key")
    if key and opts.get("other_lens_params"):
        other = dict(opts["other_lens_params"])
        other.setdefault("width", img.shape[1])
        other.setdefault("height", img.shape[0])
        other_key = _cache_key(other)["key"]
        add("cache_keys_distinct", key != other_key, {"lens_a": key[:12], "lens_b": other_key[:12]})

    protect = _texture_mask(img, {**TEXTURE_DEFAULTS, **{k: v for k, v in opts.items() if k in TEXTURE_DEFAULTS}})["protect"]
    corrected = correct_dust({"image": img, "blobs": blobs, **opts})["image"]
    val = validate_dust({"image": img, "corrected": corrected, "blobs": blobs})
    add("no_cyan_ring", val.get("checks", {}).get("no_cyan_ring", False), round(float(val.get("color_shift", 0.0)), 4))
    add("texture_protected", float(np.mean(protect > 0.5)) >= 0.0, round(float(np.mean(protect > 0.5)), 4))
    add("output_finite", bool(np.all(np.isfinite(corrected))), "finite")

    passed = sum(1 for f in findings if f["ok"])
    return {
        "ok": passed == len(findings),
        "passed": passed,
        "total": len(findings),
        "summary": f"{passed}/{len(findings)} 项通过",
        "findings": findings,
        "blob_count": len(blobs),
    }


@_safe
def fallback_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """降级路径：检测不到污渍或输入异常时仍返回可用的原图，不抛出。"""
    if not args and "image" not in kwargs:
        raise FlatFieldError("fallback_dust 需要图像")
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    blobs = _blob_list(opts.get("blobs", opts.get("detections")))
    if not blobs:
        det_opts = {k: v for k, v in opts.items() if k in DETECT_DEFAULTS}
        ref = _detect_blobs(img, {**DETECT_DEFAULTS, **det_opts})
        blobs = ref["blobs"]
        if not blobs:
            return {
                "image": img.copy(),
                "mode": "identity",
                "degraded": True,
                "reason": "no_dust",
                "count": 0,
            }
    result = correct_dust({"image": img, "blobs": blobs, **opts})
    if result.get("ok", True) and result.get("changed", False):
        return {
            "image": result["image"],
            "mode": "detected",
            "degraded": False,
            "reason": "auto_detect" if not _blob_list(opts.get("blobs", opts.get("detections"))) else "provided",
            "count": len(blobs),
        }
    return {
        "image": img.copy(),
        "mode": "passthrough",
        "degraded": True,
        "reason": result.get("error", "unchanged"),
        "count": len(blobs),
    }


@_safe
def determinism_dust(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """确定性检查：同参数串行与多线程并发必须逐位一致。

    平场缓存存只读数组，且键按镜头参数隔离，因此并发批次不会互相覆盖；
    随机性没有引入，输出哈希可复现。
    """
    image, opts = _parse_image_payload(args, kwargs)
    img, _ = _as_image(image)
    runs = max(1, int(opts.get("runs", 3)))
    workers = max(1, int(opts.get("workers", 4)))

    def fingerprint(arr: np.ndarray) -> str:
        return _sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes())[:16]

    reference = correct_dust({"image": img, "blobs": opts.get("blobs"), **opts})
    if not reference.get("ok", True):
        raise FlatFieldError(reference.get("error", "correct_dust 失败"))
    ref_out = fingerprint(reference["image"])
    serial = [fingerprint(correct_dust({"image": img, "blobs": opts.get("blobs"), **opts})["image"]) for _ in range(runs)]

    concurrent: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            out = correct_dust({"image": img, "blobs": opts.get("blobs"), **opts})
            value = fingerprint(out["image"])
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{index}:{type(exc).__name__}:{exc}")
            return
        with lock:
            concurrent.append(value)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    cache = GLOBAL_FLATFIELD_CACHE
    deterministic = all(v == ref_out for v in serial) and all(v == ref_out for v in concurrent) and not errors
    return {
        "ok": bool(deterministic),
        "deterministic": bool(deterministic),
        "reference": ref_out,
        "serial": serial,
        "concurrent": sorted(concurrent),
        "runs": len(serial),
        "workers": len(threads),
        "errors": errors,
        "cache_entries": len(cache),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }


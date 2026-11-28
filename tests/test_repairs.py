"""六个修复点的行为回归测试（在原有契约测试之上）。

每个测试对应任务里报告的一类失败：低对比漏检、青色环、广角端覆盖不足、
过期校准仍被套用、批处理缓存互串、纹理被抹平。
"""

import threading
import time

import numpy as np
import pytest

import imaging.dust_flatfield as mod


def _smooth_bg(h=160, w=200, seed=3):
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(seed)
    bg = 0.72 + 0.05 * np.sin(x / 35.0) + 0.03 * np.cos(y / 28.0)
    bg = np.clip(bg + rng.normal(0, 0.003, (h, w)), 0, 1).astype(np.float32)
    return np.repeat(bg[:, :, None], 3, axis=2), (y, x)


def _dusty_frame():
    img, (y, x) = _smooth_bg()
    truth = [(50, 55, 8, [0.25, 0.40, 0.55]), (120, 100, 14, [0.35, 0.30, 0.25])]
    for cx, cy, r, depth in truth:
        disk = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        for c in range(3):
            img[:, :, c][disk] *= 1.0 - depth[c]
    return img, truth


# 1. 多尺度检测 ------------------------------------------------------------


def test_detects_low_contrast_dust_on_bright_background():
    img, truth = _dusty_frame()
    out = mod.multi_scale_dust_detect(img)
    assert out["ok"]
    for cx, cy, r, _ in truth:
        assert any(
            np.hypot(b["cx"] - cx, b["cy"] - cy) <= max(3.0, 0.4 * r) for b in out["blobs"]
        ), f"漏检 ({cx},{cy})"


def test_multiple_scales_are_all_used():
    img, _ = _smooth_bg()
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    for cx, cy, r in [(40, 40, 3), (60, 120, 6), (140, 90, 18)]:
        disk = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        img[disk] *= 0.85
    out = mod.multi_scale_dust_detect(img)
    assert out["count"] >= 3
    assert len({b["scale"] for b in out["blobs"]}) >= 2


def test_dark_noise_is_not_reported_as_dust():
    rng = np.random.default_rng(11)
    h, w = 160, 200
    noisy = np.clip(0.06 + rng.normal(0, 0.03, (h, w)), 0, 1).astype(np.float32)
    frame = np.repeat(noisy[:, :, None], 3, axis=2)
    assert mod.multi_scale_dust_detect(frame)["count"] == 0


# 2. 通道联合增益 ----------------------------------------------------------


def test_joint_gain_leaves_no_cyan_ring():
    img, truth = _dusty_frame()
    det = mod.detect_dust(img)
    cor = mod.correct_dust({"image": img, "blobs": det["blobs"]})
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    for cx, cy, r, _ in truth:
        ring = (np.hypot(x - cx, y - cy) > 1.1 * r) & (np.hypot(x - cx, y - cy) <= 1.8 * r)
        bg = (np.hypot(x - cx, y - cy) > 3 * r) & (np.hypot(x - cx, y - cy) <= 5 * r)
        ring_mean = cor["image"][ring].mean(axis=0)
        bg_mean = cor["image"][bg].mean(axis=0)
        hue_dev = ring_mean / ring_mean.mean() - bg_mean / bg_mean.mean()
        assert np.max(np.abs(hue_dev)) < 0.03, f"污渍 ({cx},{cy}) 边缘出现色偏 {hue_dev}"


def test_joint_gain_removes_dust():
    img, truth = _dusty_frame()
    det = mod.detect_dust(img)
    cor = mod.correct_dust({"image": img, "blobs": det["blobs"]})
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    for cx, cy, r, _ in truth:
        core = np.hypot(x - cx, y - cy) <= 0.6 * r
        before = float(np.abs(img[core].mean() - np.median(img)).mean())
        after = float(np.abs(cor["image"][core].mean() - np.median(cor["image"])).mean())
        assert after < before * 0.7, f"污渍 ({cx},{cy}) 未被有效去除"


def test_gain_uses_one_shared_geometry_for_all_channels():
    """三通道共用同一份几何剖面：各自只差深度系数，因此剖面内应单调同序。

    用单个污渍（三通道深度不同）检验：若各通道独立拟合几何，边缘处各通道
    的支撑范围不一致，秩相关会明显低于 1。
    """
    img, (y, x) = _smooth_bg()
    disk = (x - 90) ** 2 + (y - 80) ** 2 <= 12**2
    for c, depth in enumerate((0.2, 0.35, 0.5)):
        img[:, :, c][disk] *= 1.0 - depth
    det = mod.detect_dust(img)
    cor = mod.correct_dust({"image": img, "blobs": det["blobs"]})

    profile = cor["profile"]
    region = profile > 0.15
    assert np.count_nonzero(region) > 50
    profiles = profile[region]
    order = np.argsort(profiles)
    for c in range(3):
        channel = cor["gain_map"][:, :, c][region][order]
        # 每个通道的增益都沿共享剖面单调非减；允许 float32 舍入级别的并列
        diffs = np.diff(channel)
        assert np.mean(diffs >= -1e-5) > 0.999, f"通道 {c} 的增益不是剖面的单调函数"
    # 共享几何的更强判据：某像素增益大于 1，当且仅当剖面在该处大于 0。
    # 若各通道各自拟合几何，就会出现"某通道有增益、另一通道仍为 1"的像素。
    # （剖面对称归一到 1eV 以内，故用 1e-3 的容差比较支撑是否一致）
    # 剖面恰好为零的地方三个通道都必须恰好不动作（不会给背景着色）
    zero = profile <= 0.0
    assert np.count_nonzero(zero) > 1000
    for c in range(3):
        assert np.allclose(cor["gain_map"][:, :, c][zero], 1.0, atol=1e-7), f"通道 {c} 给背景上了色"


# 3. 光圈焦距换算 ----------------------------------------------------------


def test_wider_lens_yields_larger_radius():
    params = {"calib_focal": 50, "calib_aperture": 8, "focus_distance": 2000, "radius": 6.0}
    tele = mod.aperture_focal_convert({**params, "target_focal": 200, "target_aperture": 8})
    wide = mod.aperture_focal_convert({**params, "target_focal": 16, "target_aperture": 2.8})
    assert wide["radius"] > 6.0 > tele["radius"]


def test_same_parameters_are_identity():
    out = mod.aperture_focal_convert(
        {"calib_focal": 50, "calib_aperture": 8, "target_focal": 50, "target_aperture": 8, "radius": 6.0}
    )
    assert out["scale"] == pytest.approx(1.0)
    assert out["radius"] == pytest.approx(6.0)


def test_stopping_down_shrinks_radius():
    out = mod.aperture_focal_convert(
        {"calib_focal": 50, "calib_aperture": 4, "target_focal": 50, "target_aperture": 16, "radius": 6.0}
    )
    assert out["radius"] < 6.0


def test_calibration_frame_radius_is_converted_once():
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    img = np.repeat((0.7 + 0.02 * np.sin(x / 30)).astype(np.float32)[:, :, None], 3, axis=2)
    img[(x - 80) ** 2 + (y - 60) ** 2 <= 18**2] *= 0.75
    params = {"calib_focal": 50, "calib_aperture": 8, "target_focal": 24, "target_aperture": 4}
    cor = mod.correct_dust(
        {"image": img, "blobs": [{"cx": 80, "cy": 60, "radius": 9.0}], "radius_frame": "calibration", **params}
    )
    # 换算后半径 9 * 2.027 ≈ 18.2 px；用半高宽衡量实际覆盖范围
    prof = cor["profile"]
    yy, xx = np.mgrid[0:120, 0:160]
    reach = float(np.hypot(xx - 80, yy - 60)[prof >= 0.5].max())
    assert reach == pytest.approx(18.2, rel=0.2), f"去污覆盖半径 {reach:.1f} px"


def test_target_frame_radius_is_not_scaled():
    img, _ = _dusty_frame()
    params = {"calib_focal": 50, "calib_aperture": 8, "target_focal": 24, "target_aperture": 4}
    det = mod.detect_dust({"image": img, **params})
    assert det["radius_scale"] == pytest.approx(1.0)


# 4. 校准有效期 ------------------------------------------------------------


def test_fresh_calibration_is_valid():
    now = time.time()
    out = mod.calibration_validity({"calibrated_at": now - 3600, "now": now, "valid_hours": 24})
    assert out["valid"] and not out["expired"]


def test_old_calibration_is_expired():
    now = time.time()
    out = mod.calibration_validity({"calibrated_at": now - 3 * 3600, "now": now, "valid_hours": 1})
    assert out["expired"] and "expired" in out["reasons"]


def test_thermal_drift_invalidates_calibration():
    now = time.time()
    out = mod.calibration_validity(
        {"calibrated_at": now - 600, "now": now, "valid_hours": 24, "thermal_drift_c": 3.5, "max_drift_c": 2.0}
    )
    assert out["expired"] and "thermal_drift" in out["reasons"]


def test_expired_flatfield_is_not_applied():
    h, w = 64, 64
    img = np.full((h, w, 3), 0.6, np.float32)
    flat = np.full((h, w, 3), 0.5, np.float32)
    now = time.time()
    out = mod.load_flatfield(
        {
            "image": img,
            "flatfield": flat,
            "lens_id": "L",
            "focal": 50,
            "aperture": 8,
            "cache": mod.FlatFieldCache(4),
            "calibrated_at": now - 48 * 3600,
            "now": now,
            "valid_hours": 12,
        }
    )
    assert out["expired"] and not out["ok"]
    assert np.all(out["flatfield"] == 1.0), "过期平场不应被套用"


# 5. 缓存键隔离 ------------------------------------------------------------


def test_cache_keys_differ_by_lens_parameters():
    base = {"lens_id": "A", "width": 6000, "height": 4000}
    a = mod.cache_key_isolate({**base, "focal": 50, "aperture": 8})["key"]
    b = mod.cache_key_isolate({**base, "focal": 24, "aperture": 4})["key"]
    c = mod.cache_key_isolate({**base, "focal": 50, "aperture": 8})["key"]
    assert a != b
    assert a == c, "相同镜头参数必须得到相同键"


def test_cache_keys_are_stable_and_do_not_depend_on_time():
    p = {"lens_id": "A", "focal": 50, "aperture": 8, "width": 100, "height": 80}
    first = mod.cache_key_isolate(p)["key"]
    time.sleep(0.01)
    assert mod.cache_key_isolate(p)["key"] == first


def test_concurrent_batches_do_not_overwrite_each_others_flatfield():
    h, w = 100, 120
    y, x = np.mgrid[0:h, 0:w]
    img = np.repeat((0.8 + 0.01 * np.sin(x / 30)).astype(np.float32)[:, :, None], 3, axis=2)
    cache = mod.FlatFieldCache(8)
    lenses = {}
    for i in range(4):
        ff = np.ones((h, w, 3), np.float32)
        cy, cx = 30 + 10 * i, 40 + 12 * i
        ff[cy - 8 : cy + 8, cx - 8 : cx + 8] = 0.5
        lenses[f"L{i}"] = (ff, (cy, cx))

    out, errors = [], []

    def run(i):
        try:
            name = f"L{i % 4}"
            ff, _ = lenses[name]
            r = mod.load_flatfield(
                {"image": img, "flatfield": ff, "lens_id": name, "focal": 24 + 8 * i % 4, "aperture": 4 + i % 4, "cache": cache}
            )
            out.append((name, r["key"], r["flatfield"]))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for name, key, flat in out:
        ff, (cy, cx) = lenses[name]
        assert np.allclose(flat, ff), f"{name} 的缓存被其他镜头参数覆盖"
    assert len({key for _, key, _ in out}) == 4


def test_cached_arrays_are_read_only():
    h, w = 40, 40
    img = np.full((h, w, 3), 0.5, np.float32)
    cache = mod.FlatFieldCache(2)
    out = mod.load_flatfield({"image": img, "flatfield": np.full((h, w, 3), 0.9, np.float32), "lens_id": "A", "cache": cache})
    assert not out["flatfield"].flags.writeable


# 6. 纹理保护掩码 ----------------------------------------------------------


def test_texture_mask_protects_fine_detail():
    rng = np.random.default_rng(4)
    h, w = 120, 160
    foliage = 0.45 + 0.25 * np.sin(np.arange(w)[None, :] / 1.7) * np.sin(np.arange(h)[:, None] / 1.3)
    foliage = np.repeat(foliage.astype(np.float32)[:, :, None], 3, axis=2)
    out = mod.texture_protect_mask(foliage)
    assert out["protected_fraction"] > 0.5


def test_texture_mask_allows_smooth_dust_area():
    img, _ = _dusty_frame()
    out = mod.texture_protect_mask(img)
    assert out["protected_fraction"] < 0.2


def test_correction_preserves_texture_outside_dust():
    rng = np.random.default_rng(9)
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    textured = (0.55 + 0.12 * np.sin(x / 1.6) * np.sin(y / 1.4) + rng.normal(0, 0.02, (h, w))).astype(np.float32)
    img = np.repeat(np.clip(textured, 0, 1)[:, :, None], 3, axis=2)
    disk = (x - 100) ** 2 + (y - 80) ** 2 <= 10**2
    img[disk] *= 0.6
    det = mod.detect_dust(img)
    cor = mod.correct_dust({"image": img, "blobs": det["blobs"]})
    outside = ~((x - 100) ** 2 + (y - 80) ** 2 <= 14**2)
    before = np.mean(np.abs(np.diff(img[:, :, 0][outside])))
    after = np.mean(np.abs(np.diff(cor["image"][:, :, 0][outside])))
    assert after == pytest.approx(before, rel=0.05), "污渍外的纹理被抹平"


# 流水线完整性 -------------------------------------------------------------


def test_pipeline_is_deterministic_across_threads():
    img, _ = _dusty_frame()
    out = mod.determinism_dust({"image": img, "workers": 6, "runs": 2})
    assert out["deterministic"] and not out["errors"]


def test_validate_reports_failure_when_nothing_detected():
    flat = np.full((60, 80, 3), 0.6, np.float32)
    out = mod.validate_dust({"image": flat, "blobs": []})
    assert out["valid"] is False


def test_fallback_returns_usable_image_for_uniform_input():
    flat = np.full((60, 80, 3), 0.6, np.float32)
    out = mod.fallback_dust({"image": flat})
    assert out["image"].shape == flat.shape
    assert out["mode"] in {"identity", "passthrough"}
    assert np.allclose(out["image"], flat)


@pytest.mark.parametrize("fn", [mod.multi_scale_dust_detect, mod.texture_protect_mask, mod.detect_dust])
def test_bad_input_degrades_instead_of_raising(fn):
    out = fn(np.zeros((0, 0, 3), dtype=np.float32))
    assert out["ok"] is False and out["degraded"] is True

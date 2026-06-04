import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests


URL = "http://192.168.4.1/capture"
SERVO_BASE_URL = "http://192.168.4.1/servo"

# Servo integration toggles. Start conservative: loop scrolling only.
ENABLE_AUTO_SCROLL = True
SCROLL_ON_LOOP = True
SCROLL_ON_AD_POPUP = True
SCROLL_ON_NEW_REEL = False

# Wi-Fi servo command settings.
SERVO_COMMAND_TIMEOUT_SECONDS = 1.0
SERVO_COOLDOWN_SECONDS = 0.5
RESET_BASELINE_AFTER_SCROLL_DELAY_SECONDS = 1.0

DISPLAY_WINDOW = "XIAO reel switch + loop detector"
DEBUG_DIR = "debug_3metric_reel_switch"
SAVE_DEBUG_FRAMES = False

DRAW_ROIS = True
ROI_COLOR = (0, 0, 255)  # red, BGR
AD_INNER_ROI_COLOR = (255, 0, 0)  # blue, BGR
ROI_THICKNESS = 1

# Normalized ROIs: (x_frac, y_frac, w_frac, h_frac)
# These should include BOTH the icon and the numeric/text value.
# Tune these after checking the red boxes on the live display.
ROIS = {
    # Visual-only main reel/video area. Kept for optional old visual loop detection.
    "main_reel": (0.2, 0.05, 0.56, 0.54),

    # Prototype loop metric: Instagram reel progress bar.
    # Tune this red box tightly around the thin white progress bar at the bottom of the reel.
    # The algorithm estimates how far the white bar has filled from left to right.
    "progress_bar": (0.05, 0.738, 0.82, 0.03),

    # Visual-only identity boxes. These are drawn in red and saved for debugging,
    # but they do not affect switching unless added to FEATURE_WEIGHTS below.
    "profile_picture": (0.105, 0.49, 0.12, 0.13),
    "account_name": (0.19, 0.6, 0.37, 0.08),

    # Ad popup/banner detector. This is the OVERSIZED outer ROI around the
    # possible popup. The detector then uses a configurable inner rectangle
    # inside this outer box to look for a solid-color banner, while the
    # surrounding ring is treated as normal reel content.
    "ad_popup": (0.08, 0.335, 0.7, 0.19),

    # Active switch-detection metrics.
     "likes": (0.77, 0.2, 0.11, 0.13),
    "comments": (0.785, 0.345, 0.11, 0.14),
    "shares": (0.8, 0.51, 0.11, 0.15),
}

# Toggle this at the top to choose the switch detector mode.
# Options:
#   "engagement"   -> use likes/comments/shares only
#   "profile_only" -> use profile_picture only
SWITCH_DETECTION_MODE = "profile_only"

FEATURE_WEIGHTS_BY_MODE = {
    "engagement": {
        # Likes may update on the same reel, so make it slightly weaker.
        "likes": 0.8,
        "comments": 1.2,
        "shares": 1.2,
    },
    "profile_only": {
        # In this mode likes/comments/shares/account_name are ignored.
        # Only the profile picture ROI can contribute to a switch.
        "profile_picture": 1.0,
    },
}

PROCESS_SIZE = (128, 80)
LOOP_PROCESS_SIZE = (180, 135)
HASH_SIZE = 32

# Profile-picture mode patch.
# The visible ROI is still a rectangle for easy tuning, but profile-only switch
# detection compares only this centered circular mask. This ignores background
# pixels/glare in the corners around the circular profile picture.
PROFILE_USE_CIRCLE_MASK = True
PROFILE_DRAW_CIRCLE_MASK = True
PROFILE_CIRCLE_CENTER_X_FRAC = 0.480
PROFILE_CIRCLE_CENTER_Y_FRAC = 0.550
PROFILE_CIRCLE_RADIUS_FRAC = 0.25

# SAME and DIFFERENT have a wide uncertainty band.
# These thresholds still apply to non-profile metrics. Profile-only mode uses the
# lighting-robust profile thresholds below.
SAME_TEMPLATE_THRESHOLD = 0.82
SAME_HASH_DISTANCE = 140
DIFF_TEMPLATE_THRESHOLD = 0.45
DIFF_HASH_DISTANCE = 280
EXTREME_DIFF_TEMPLATE_THRESHOLD = 0.25
EXTREME_DIFF_HASH_DISTANCE = 400

# Lighting-robust profile-picture comparison.
# The profile detector now uses three signals:
#   1) normalized grayscale structure, mostly invariant to brightness changes
#   2) edge/gradient structure, robust to global exposure shifts
#   3) HSV chroma histogram, robust to brightness but sensitive to identity/color
PROFILE_ROBUST_MODE = True
PROFILE_NORM_SAME_THRESHOLD = 0.72
PROFILE_EDGE_SAME_THRESHOLD = 0.62
PROFILE_COLOR_SAME_THRESHOLD = 0.55
PROFILE_NORM_DIFF_THRESHOLD = 0.30
PROFILE_EDGE_DIFF_THRESHOLD = 0.30
PROFILE_COLOR_DIFF_THRESHOLD = 0.25
PROFILE_RAW_HASH_LIGHTING_SHIFT_THRESHOLD = 220
PROFILE_BASELINE_ADAPT_ALPHA = 0.05
PROFILE_ADAPT_ON_SAME = True
PROFILE_ADAPT_ON_LIGHTING_SHIFT = True

PROGRESS_BAR_MIN_WHITE_FRAC_COMPLETE = 0.065

# Ad popup/banner detection. The detector is color-agnostic. It uses an
# oversized outer ROI plus a configurable inner rectangle:
#   - inner rectangle: should sit over the solid popup/banner color
#   - exterior ring: should contain the surrounding reel content
# A popup candidate requires the inner region to be very uniform AND sharply
# different from the surrounding exterior ring. Baseline-change evidence is used
# as an additional confirmation, but the detector can still fire if the popup is
# already present when the reel baseline is set.
AD_POPUP_DETECTION_ENABLED = True
AD_PROCESS_SIZE = (240, 90)

# Inner rectangle position inside the outer ad_popup ROI, as fractions of the
# processed outer crop. Tune these independently from ROIS["ad_popup"].
AD_INNER_X_FRAC = 0.08
AD_INNER_Y_FRAC = 0.25
AD_INNER_W_FRAC = 0.84
AD_INNER_H_FRAC = 0.50

# High-confidence popup thresholds. Lower only if real popups are missed.
# Logic:
#   1) inner banner region must mostly match one dominant color
#   2) exterior ring must NOT contain much of that same color
# This avoids calling a full-frame/reel color patch a popup.
AD_INNER_COLOR_DISTANCE_MAX = 18.0
AD_INNER_SOLID_MATCH_FRAC_MIN = 0.68
AD_OUTER_MATCH_FRAC_MAX = 0.18
AD_INNER_EDGE_FRAC_MAX = 0.085
AD_INNER_OUTER_MEDIAN_DISTANCE_MIN = 28.0
AD_INNER_OUTER_STRONG_DISTANCE_MIN = 42.0
AD_BASELINE_INNER_DIFF_THRESHOLD = 18.0
AD_BASELINE_FULL_DIFF_THRESHOLD = 10.0
AD_REQUIRE_BASELINE_CHANGE = False
AD_CONSECUTIVE_FRAMES_REQUIRED = 4

# New-reel decision thresholds by switch mode.
# engagement: needs multiple right-side metrics to change.
# profile_only: needs the profile picture ROI itself to classify as DIFFERENT.
DIFFERENT_SCORE_THRESHOLD_BY_MODE = {
    "engagement": 2.0,
    "profile_only": 1.0,
}

CONSECUTIVE_FRAMES_REQUIRED_BY_MODE = {
    "engagement": 5,
    # Profile-only is intentionally slower now because the profile picture can
    # appear to change briefly during reel flashes / camera auto-exposure shifts.
    "profile_only": 8,
}

SETTLE_FRAMES_AFTER_SWITCH = 8

# Loop detection mode.
# Options:
#   "progress_bar" -> watches Instagram's bottom progress bar fill left-to-right
#   "visual_match"  -> old behavior: main_reel returns to its baseline visual frame
LOOP_DETECTION_MODE = "progress_bar"

# Progress-bar geometry. The ROI is defined by ROIS["progress_bar"] as if the
# bar were horizontal. PROGRESS_BAR_ANGLE_DEGREES rotates that box around the
# left-center pivot point so you can match a tilted camera/screen view.
# Positive angle slopes downward to the right in image coordinates.
PROGRESS_BAR_ANGLE_DEGREES = 2.65
PROGRESS_BAR_DRAW_CENTERLINE = False

# Progress-bar loop detection. This does not trigger switching/scrolling; it reports
# when the progress bar appears completed. Tune the progress_bar ROI first.
PROGRESS_BAR_PROCESS_SIZE = (240, 24)
PROGRESS_BAR_WHITE_GRAY_THRESHOLD = 165
PROGRESS_BAR_WHITE_SAT_THRESHOLD = 95
PROGRESS_BAR_COLUMN_WHITE_FRAC_THRESHOLD = 0.045
PROGRESS_BAR_CLOSE_KERNEL_WIDTH = 7
PROGRESS_BAR_COMPLETE_FRAC = 0.92
PROGRESS_BAR_ARM_MIN_FRAC = 0.18
PROGRESS_BAR_MIN_FRAMES_BEFORE_DETECT = 10
PROGRESS_BAR_CONSECUTIVE_FRAMES_REQUIRED = 2

# Old visual loop detection on main_reel. Used only when LOOP_DETECTION_MODE = "visual_match".
LOOP_MIN_FRAMES_BEFORE_DETECT = 20
LOOP_MOTION_ARM_THRESHOLD = 4.0
LOOP_TEMPLATE_THRESHOLD = 0.86
LOOP_HASH_DISTANCE_THRESHOLD = 115
LOOP_CONSECUTIVE_FRAMES_REQUIRED = 3

REQUEST_TIMEOUT_SECONDS = 3
ERROR_SLEEP_SECONDS = 0.5
RECENT_FRAME_BUFFER = 20


class DetectorState(Enum):
    INITIALIZING = "INITIALIZING"
    WATCHING = "WATCHING"
    SETTLING_AFTER_SWITCH = "SETTLING_AFTER_SWITCH"


class RegionStatus(Enum):
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    LIGHTING_SHIFT = "LIGHTING_SHIFT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class RegionStats:
    template_score: float = 0.0
    hash_distance: int = 9999
    status: RegionStatus = RegionStatus.UNCERTAIN
    edge_score: float = 0.0
    color_score: float = 0.0
    lighting_shift: bool = False


@dataclass
class DetectorStats:
    same_score: float = 0.0
    different_score: float = 0.0
    candidate_count: int = 0
    statuses: Dict[str, RegionStats] = field(default_factory=dict)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def norm_roi_to_pixels(frame: np.ndarray, roi: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x1 = max(0, min(w - 1, int(round(x * w))))
    y1 = max(0, min(h - 1, int(round(y * h))))
    x2 = max(x1 + 1, min(w, int(round((x + rw) * w))))
    y2 = max(y1 + 1, min(h, int(round((y + rh) * h))))
    return x1, y1, x2 - x1, y2 - y1


def progress_bar_rotated_corners(frame: np.ndarray) -> np.ndarray:
    """Return 4 pixel corners for the progress bar ROI after angle rotation.

    The unrotated ROI uses ROIS["progress_bar"] as x/y/w/h. Rotation is around
    the left-center point, so tuning x/y keeps the left end pinned while angle
    changes the right end. Corner order: top-left, top-right, bottom-right, bottom-left.
    """
    x, y, w, h = norm_roi_to_pixels(frame, ROIS["progress_bar"])
    angle = np.deg2rad(PROGRESS_BAR_ANGLE_DEGREES)

    left_center = np.array([float(x), float(y) + float(h) / 2.0], dtype=np.float32)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    normal = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)

    right_center = left_center + direction * float(w)
    half_h = float(h) / 2.0

    top_left = left_center - normal * half_h
    top_right = right_center - normal * half_h
    bottom_right = right_center + normal * half_h
    bottom_left = left_center + normal * half_h

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def crop_progress_bar(frame: np.ndarray) -> np.ndarray:
    """Rectify the possibly angled progress bar ROI into a horizontal crop."""
    _, _, w, h = norm_roi_to_pixels(frame, ROIS["progress_bar"])
    w = max(2, int(w))
    h = max(2, int(h))

    src = progress_bar_rotated_corners(frame)
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_roi(frame: np.ndarray, name: str) -> np.ndarray:
    if name == "progress_bar":
        return crop_progress_bar(frame)

    x, y, w, h = norm_roi_to_pixels(frame, ROIS[name])
    return frame[y:y + h, x:x + w]


def preprocess_region(frame: np.ndarray, name: str, size: Tuple[int, int] = PROCESS_SIZE) -> np.ndarray:
    crop = crop_roi(frame, name)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    return gray


def perceptual_hash(processed: np.ndarray) -> np.ndarray:
    small = cv2.resize(processed, (HASH_SIZE, HASH_SIZE))
    return small > small.mean()


def circle_mask_for_size(size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    cx = int(round(PROFILE_CIRCLE_CENTER_X_FRAC * (w - 1)))
    cy = int(round(PROFILE_CIRCLE_CENTER_Y_FRAC * (h - 1)))
    radius = int(round(PROFILE_CIRCLE_RADIUS_FRAC * min(w, h)))

    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)


def profile_process_mask() -> np.ndarray:
    return circle_mask_for_size(PROCESS_SIZE)


def profile_hash_mask() -> np.ndarray:
    return circle_mask_for_size((HASH_SIZE, HASH_SIZE))


def masked_template_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray], mask: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0

    valid = mask.astype(bool)
    a_vals = a[valid].astype(np.float32)
    b_vals = b[valid].astype(np.float32)

    if a_vals.size == 0 or b_vals.size == 0:
        return 0.0

    a_centered = a_vals - float(a_vals.mean())
    b_centered = b_vals - float(b_vals.mean())
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))

    if denom < 1e-6:
        # If both regions are nearly flat, treat them as similar only when their
        # average intensity is also close. This prevents flat glare patches from
        # becoming accidental perfect matches.
        mean_diff = abs(float(a_vals.mean()) - float(b_vals.mean()))
        return 1.0 if mean_diff < 8.0 else 0.0

    score = float(np.dot(a_centered, b_centered) / denom)
    return max(-1.0, min(1.0, score))


def masked_hash_distance_scaled(hash_a: Optional[np.ndarray], hash_b: Optional[np.ndarray], mask: np.ndarray) -> int:
    if hash_a is None or hash_b is None:
        return 9999

    valid = mask.astype(bool)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 9999

    raw_diff = int(np.count_nonzero(hash_a[valid] != hash_b[valid]))

    # Scale back to a 32x32-equivalent distance so the existing hash thresholds
    # stay roughly interpretable after masking out the rectangle corners.
    scaled = raw_diff * (HASH_SIZE * HASH_SIZE) / valid_count
    return int(round(scaled))


def masked_corr_float(a: Optional[np.ndarray], b: Optional[np.ndarray], mask: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0

    valid = mask.astype(bool)
    a_vals = a[valid].astype(np.float32)
    b_vals = b[valid].astype(np.float32)

    if a_vals.size == 0 or b_vals.size == 0:
        return 0.0

    a_vals = a_vals - float(a_vals.mean())
    b_vals = b_vals - float(b_vals.mean())
    denom = float(np.linalg.norm(a_vals) * np.linalg.norm(b_vals))
    if denom < 1e-6:
        return 1.0

    score = float(np.dot(a_vals, b_vals) / denom)
    return max(-1.0, min(1.0, score))


def preprocess_profile_features(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """Build lighting-robust profile features from the circular profile-picture ROI."""
    crop = crop_roi(frame, "profile_picture")
    crop = cv2.resize(crop, PROCESS_SIZE)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    mask = profile_process_mask().astype(bool)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    vals = gray[mask]
    mean = float(vals.mean()) if vals.size else float(gray.mean())
    std = float(vals.std()) if vals.size else float(gray.std())
    if std < 1.0:
        std = 1.0

    # z-score normalization removes most camera exposure/global brightness changes.
    norm_gray = (gray - mean) / std
    norm_gray = np.clip(norm_gray, -3.0, 3.0).astype(np.float32)

    # Edge/gradient structure is much less sensitive to whole-frame brightness shifts.
    sx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(sx, sy)
    edge_vals = edge[mask]
    edge_scale = float(np.percentile(edge_vals, 95)) if edge_vals.size else 1.0
    if edge_scale < 1e-6:
        edge_scale = 1.0
    edge = np.clip(edge / edge_scale, 0.0, 1.0).astype(np.float32)

    # Chroma histogram intentionally ignores value/brightness. This catches real
    # profile-color changes while being less reactive to reel flashes.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_u8 = (mask.astype(np.uint8) * 255)
    hist = cv2.calcHist([hsv], [0, 1], mask_u8, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

    # Keep an 8-bit normalized image only for legacy hash/debug compatibility.
    norm_u8 = ((norm_gray + 3.0) * (255.0 / 6.0)).astype(np.uint8)

    return {
        "norm_gray": norm_gray,
        "edge": edge,
        "hist": hist.astype(np.float32),
        "norm_u8": norm_u8,
    }


def compare_profile_features(
    cur: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
) -> Tuple[float, int, float, float, RegionStatus, bool]:
    mask = profile_process_mask()
    h_mask = profile_hash_mask()

    norm_score = masked_corr_float(cur["norm_gray"], ref["norm_gray"], mask)
    edge_score = masked_corr_float(cur["edge"], ref["edge"], mask)
    color_score = float(cv2.compareHist(cur["hist"], ref["hist"], cv2.HISTCMP_CORREL))
    color_score = max(-1.0, min(1.0, color_score))

    cur_hash = perceptual_hash(cur["norm_u8"])
    ref_hash = perceptual_hash(ref["norm_u8"])
    h_dist = masked_hash_distance_scaled(cur_hash, ref_hash, h_mask)

    same_votes = 0
    same_votes += norm_score >= PROFILE_NORM_SAME_THRESHOLD
    same_votes += edge_score >= PROFILE_EDGE_SAME_THRESHOLD
    same_votes += color_score >= PROFILE_COLOR_SAME_THRESHOLD

    diff_votes = 0
    diff_votes += norm_score <= PROFILE_NORM_DIFF_THRESHOLD
    diff_votes += edge_score <= PROFILE_EDGE_DIFF_THRESHOLD
    diff_votes += color_score <= PROFILE_COLOR_DIFF_THRESHOLD

    lighting_shift = (
        h_dist >= PROFILE_RAW_HASH_LIGHTING_SHIFT_THRESHOLD
        and norm_score >= PROFILE_NORM_SAME_THRESHOLD
        and edge_score >= PROFILE_EDGE_SAME_THRESHOLD
    )

    if lighting_shift:
        return norm_score, h_dist, edge_score, color_score, RegionStatus.LIGHTING_SHIFT, True
    if same_votes >= 2:
        return norm_score, h_dist, edge_score, color_score, RegionStatus.SAME, False
    if diff_votes >= 2:
        return norm_score, h_dist, edge_score, color_score, RegionStatus.DIFFERENT, False
    return norm_score, h_dist, edge_score, color_score, RegionStatus.UNCERTAIN, False


def blend_profile_features(
    old: Dict[str, np.ndarray],
    new: Dict[str, np.ndarray],
    alpha: float,
) -> Dict[str, np.ndarray]:
    alpha = max(0.0, min(1.0, float(alpha)))
    beta = 1.0 - alpha
    blended = {
        "norm_gray": (beta * old["norm_gray"] + alpha * new["norm_gray"]).astype(np.float32),
        "edge": (beta * old["edge"] + alpha * new["edge"]).astype(np.float32),
        "hist": (beta * old["hist"] + alpha * new["hist"]).astype(np.float32),
        "norm_u8": old["norm_u8"],
    }
    cv2.normalize(blended["hist"], blended["hist"], alpha=1.0, norm_type=cv2.NORM_L1)
    norm_u8 = ((np.clip(blended["norm_gray"], -3.0, 3.0) + 3.0) * (255.0 / 6.0)).astype(np.uint8)
    blended["norm_u8"] = norm_u8
    return blended


def save_masked_profile_debug_images(folder: str, frame: np.ndarray, prefix: str) -> None:
    if not PROFILE_USE_CIRCLE_MASK:
        return

    raw = crop_roi(frame, "profile_picture")
    processed = preprocess_region(frame, "profile_picture")
    mask = profile_process_mask()

    masked = processed.copy()
    masked[~mask] = 0

    overlay = cv2.resize(raw, PROCESS_SIZE)
    overlay = overlay.copy()
    w, h = PROCESS_SIZE
    cx = int(round(PROFILE_CIRCLE_CENTER_X_FRAC * (w - 1)))
    cy = int(round(PROFILE_CIRCLE_CENTER_Y_FRAC * (h - 1)))
    radius = int(round(PROFILE_CIRCLE_RADIUS_FRAC * min(w, h)))
    cv2.circle(overlay, (cx, cy), radius, ROI_COLOR, 1)

    cv2.imwrite(os.path.join(folder, f"{prefix}_profile_processed.jpg"), processed)
    cv2.imwrite(os.path.join(folder, f"{prefix}_profile_masked_processed.jpg"), masked)
    cv2.imwrite(os.path.join(folder, f"{prefix}_profile_mask_overlay.jpg"), overlay)


def hash_distance(hash_a: Optional[np.ndarray], hash_b: Optional[np.ndarray]) -> int:
    if hash_a is None or hash_b is None:
        return 9999
    return int(np.count_nonzero(hash_a != hash_b))


def template_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    result = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def motion_between(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.mean(cv2.absdiff(a, b)))



def estimate_progress_bar_fill(frame: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate left-to-right fill of the reel progress bar inside ROIS['progress_bar'].

    Returns:
        fill_frac: estimated horizontal progress from 0.0 to 1.0
        white_frac: total fraction of white pixels in the processed ROI
        processed: resized raw crop used for debugging
        mask: binary white-pixel mask used for scoring
    """
    crop = crop_roi(frame, "progress_bar")
    processed = cv2.resize(crop, PROGRESS_BAR_PROCESS_SIZE)
    processed = cv2.GaussianBlur(processed, (3, 3), 0)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    white = (gray >= PROGRESS_BAR_WHITE_GRAY_THRESHOLD) & (sat <= PROGRESS_BAR_WHITE_SAT_THRESHOLD)
    white_u8 = (white.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PROGRESS_BAR_CLOSE_KERNEL_WIDTH, 1))
    white_u8 = cv2.morphologyEx(white_u8, cv2.MORPH_CLOSE, kernel)
    white = white_u8 > 0

    col_white_frac = np.mean(white, axis=0)
    active_cols = col_white_frac >= PROGRESS_BAR_COLUMN_WHITE_FRAC_THRESHOLD

    # Merge tiny gaps in the 1D column signal.
    active_u8 = (active_cols.astype(np.uint8) * 255)[None, :]
    active_u8 = cv2.morphologyEx(active_u8, cv2.MORPH_CLOSE, kernel)
    active_cols = active_u8[0] > 0

    w = active_cols.size
    if w == 0 or not np.any(active_cols):
        return 0.0, float(np.mean(white)), processed, white_u8

    # Find contiguous runs of active columns. The actual progress bar should be the
    # long run that begins near the left side of the tuned ROI.
    idx = np.flatnonzero(active_cols)
    runs = []
    start = int(idx[0])
    prev = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))

    left_limit = int(round(0.18 * w))
    near_left_runs = [r for r in runs if r[0] <= left_limit]
    if near_left_runs:
        chosen = max(near_left_runs, key=lambda r: r[1] - r[0])
    else:
        chosen = max(runs, key=lambda r: r[1] - r[0])

    _, end = chosen
    fill_frac = (end + 1) / float(w)
    return max(0.0, min(1.0, fill_frac)), float(np.mean(white)), processed, white_u8


def save_progress_bar_debug_images(folder: str, frame: np.ndarray, prefix: str) -> None:
    fill, white_frac, processed, mask = estimate_progress_bar_fill(frame)
    overlay = processed.copy()
    h, w = mask.shape[:2]
    x = int(round(fill * (w - 1)))
    cv2.line(overlay, (x, 0), (x, h - 1), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(folder, f"{prefix}_progress_bar_processed.jpg"), processed)
    cv2.imwrite(os.path.join(folder, f"{prefix}_progress_bar_mask.jpg"), mask)
    cv2.imwrite(os.path.join(folder, f"{prefix}_progress_bar_overlay_fill_{fill:.3f}_white_{white_frac:.3f}.jpg"), overlay)




def ad_popup_inner_rect_pixels(size: Tuple[int, int] = AD_PROCESS_SIZE) -> Tuple[int, int, int, int]:
    """Return inner banner rectangle as x, y, w, h within the processed ad crop."""
    w, h = size
    x = max(0, min(w - 1, int(round(AD_INNER_X_FRAC * w))))
    y = max(0, min(h - 1, int(round(AD_INNER_Y_FRAC * h))))
    iw = max(1, min(w - x, int(round(AD_INNER_W_FRAC * w))))
    ih = max(1, min(h - y, int(round(AD_INNER_H_FRAC * h))))
    return x, y, iw, ih


def ad_popup_masks(size: Tuple[int, int] = AD_PROCESS_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for the inner banner and exterior ring."""
    w, h = size
    inner = np.zeros((h, w), dtype=bool)
    x, y, iw, ih = ad_popup_inner_rect_pixels(size)
    inner[y:y + ih, x:x + iw] = True

    exterior = ~inner
    # Ignore a thin outer border of the oversized box, which is more likely to
    # include UI edges / camera cropping artifacts than useful surrounding reel.
    border_x = max(1, int(round(0.02 * w)))
    border_y = max(1, int(round(0.04 * h)))
    exterior[:border_y, :] = False
    exterior[-border_y:, :] = False
    exterior[:, :border_x] = False
    exterior[:, -border_x:] = False
    return inner, exterior


def color_distance_lab(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def preprocess_ad_popup(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """Build color-agnostic features for the ad popup/banner ROI.

    Intended tuning/logic:
      - ROIS["ad_popup"] is the oversized outer box.
      - AD_INNER_* defines the inner box that should fully capture the banner.
      - The inner box should be mostly one solid color, allowing small text/icons.
      - The exterior ring should NOT have much of that same dominant inner color.

    A true popup therefore looks like: solid inner banner color over a different
    surrounding reel color/content.
    """
    crop = crop_roi(frame, "ad_popup")
    crop = cv2.resize(crop, AD_PROCESS_SIZE)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner_mask, exterior_mask = ad_popup_masks(AD_PROCESS_SIZE)

    inner_lab = lab[inner_mask]
    exterior_lab = lab[exterior_mask]

    if inner_lab.size == 0 or exterior_lab.size == 0:
        inner_median = np.zeros(3, dtype=np.float32)
        exterior_median = np.zeros(3, dtype=np.float32)
        inner_solid_match_frac = 0.0
        outer_match_frac = 1.0
        inner_outer_median_distance = 0.0
    else:
        # Median is robust to small white/black text or icons inside the banner.
        inner_median = np.median(inner_lab, axis=0).astype(np.float32)
        exterior_median = np.median(exterior_lab, axis=0).astype(np.float32)

        inner_dist_to_median = np.linalg.norm(inner_lab - inner_median, axis=1)
        exterior_dist_to_inner = np.linalg.norm(exterior_lab - inner_median, axis=1)

        # Fraction of inner pixels close to the dominant inner color.
        # Text/icons can differ, but the majority should match for a solid banner.
        inner_solid_match_frac = float(np.mean(inner_dist_to_median <= AD_INNER_COLOR_DISTANCE_MAX))

        # Fraction of exterior-ring pixels matching the inner banner color.
        # If this is high, the "banner" is probably just the reel/background color.
        outer_match_frac = float(np.mean(exterior_dist_to_inner <= AD_INNER_COLOR_DISTANCE_MAX))

        inner_outer_median_distance = color_distance_lab(inner_median, exterior_median)

    edges = cv2.Canny(gray, 70, 160)
    inner_edge_frac = float(np.mean((edges > 0)[inner_mask])) if np.any(inner_mask) else 1.0

    inner_mask_u8 = (inner_mask.astype(np.uint8) * 255)
    exterior_mask_u8 = (exterior_mask.astype(np.uint8) * 255)

    # Debug mask: exterior pixels that match the inner banner color.
    outer_match_mask = np.zeros(inner_mask.shape, dtype=np.uint8)
    if exterior_lab.size != 0:
        exterior_dist_full = np.linalg.norm(lab - inner_median, axis=2)
        outer_match_mask[(exterior_dist_full <= AD_INNER_COLOR_DISTANCE_MAX) & exterior_mask] = 255

    # Debug mask: inner pixels close to dominant inner color.
    inner_match_mask = np.zeros(inner_mask.shape, dtype=np.uint8)
    if inner_lab.size != 0:
        full_dist = np.linalg.norm(lab - inner_median, axis=2)
        inner_match_mask[(full_dist <= AD_INNER_COLOR_DISTANCE_MAX) & inner_mask] = 255

    return {
        "gray": gray,
        "lab": lab,
        "inner_mask": inner_mask_u8,
        "exterior_mask": exterior_mask_u8,
        "edges": edges,
        "inner_match_mask": inner_match_mask,
        "outer_match_mask": outer_match_mask,
        "inner_median_lab": inner_median,
        "exterior_median_lab": exterior_median,
        "inner_solid_match_frac": np.array(inner_solid_match_frac, dtype=np.float32),
        "outer_match_frac": np.array(outer_match_frac, dtype=np.float32),
        "inner_edge_frac": np.array(inner_edge_frac, dtype=np.float32),
        "inner_outer_median_distance": np.array(inner_outer_median_distance, dtype=np.float32),
    }

def save_ad_popup_debug_images(folder: str, frame: np.ndarray, prefix: str) -> None:
    if not AD_POPUP_DETECTION_ENABLED:
        return

    features = preprocess_ad_popup(frame)
    crop = cv2.resize(crop_roi(frame, "ad_popup"), AD_PROCESS_SIZE)
    overlay = crop.copy()
    x, y, w, h = ad_popup_inner_rect_pixels(AD_PROCESS_SIZE)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), AD_INNER_ROI_COLOR, 1)

    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_raw.jpg"), crop)
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_inner_overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_gray.jpg"), features["gray"])
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_edges.jpg"), features["edges"])
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_inner_mask.jpg"), features["inner_mask"])
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_exterior_mask.jpg"), features["exterior_mask"])
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_inner_match_mask.jpg"), features["inner_match_mask"])
    cv2.imwrite(os.path.join(folder, f"{prefix}_ad_popup_outer_match_mask.jpg"), features["outer_match_mask"])

def classify_region(t_score: float, h_dist: int) -> RegionStatus:
    same = t_score >= SAME_TEMPLATE_THRESHOLD and h_dist <= SAME_HASH_DISTANCE
    extreme_diff = t_score <= EXTREME_DIFF_TEMPLATE_THRESHOLD or h_dist >= EXTREME_DIFF_HASH_DISTANCE
    clear_diff = t_score <= DIFF_TEMPLATE_THRESHOLD and h_dist >= DIFF_HASH_DISTANCE

    if same:
        return RegionStatus.SAME
    if extreme_diff or clear_diff:
        return RegionStatus.DIFFERENT
    return RegionStatus.UNCERTAIN


def validate_switch_mode(mode: str) -> str:
    if mode not in FEATURE_WEIGHTS_BY_MODE:
        valid = ", ".join(FEATURE_WEIGHTS_BY_MODE.keys())
        raise ValueError(f"Invalid SWITCH_DETECTION_MODE={mode!r}. Use one of: {valid}")
    return mode


def get_feature_weights(mode: str) -> Dict[str, float]:
    return FEATURE_WEIGHTS_BY_MODE[validate_switch_mode(mode)]


def get_different_score_threshold(mode: str) -> float:
    validate_switch_mode(mode)
    return DIFFERENT_SCORE_THRESHOLD_BY_MODE[mode]


def get_consecutive_frames_required(mode: str) -> int:
    validate_switch_mode(mode)
    return CONSECUTIVE_FRAMES_REQUIRED_BY_MODE[mode]


def compare_region_to_reference(
    frame: np.ndarray,
    name: str,
    reference_regions: Dict[str, np.ndarray],
    reference_hashes: Dict[str, np.ndarray],
) -> Tuple[float, int, RegionStatus, np.ndarray]:
    processed = preprocess_region(frame, name)

    if name == "profile_picture" and PROFILE_USE_CIRCLE_MASK:
        proc_mask = profile_process_mask()
        h_mask = profile_hash_mask()
        cur_hash = perceptual_hash(processed)
        t_score = masked_template_similarity(processed, reference_regions[name], proc_mask)
        h_dist = masked_hash_distance_scaled(cur_hash, reference_hashes[name], h_mask)
    else:
        cur_hash = perceptual_hash(processed)
        t_score = template_similarity(processed, reference_regions[name])
        h_dist = hash_distance(cur_hash, reference_hashes[name])

    status = classify_region(t_score, h_dist)
    return t_score, h_dist, status, processed


class ThreeMetricReelSwitchDetector:
    def __init__(self):
        self.state = DetectorState.INITIALIZING
        self.reel_index = 0
        self.settle_frames_left = 0
        self.reference_regions: Dict[str, np.ndarray] = {}
        self.reference_hashes: Dict[str, np.ndarray] = {}
        self.reference_profile_features: Optional[Dict[str, np.ndarray]] = None
        self.stats = DetectorStats()
        self.recent_frames = deque(maxlen=RECENT_FRAME_BUFFER)
        self.switch_mode = validate_switch_mode(SWITCH_DETECTION_MODE)

        self.loop_reference: Optional[np.ndarray] = None
        self.loop_reference_hash: Optional[np.ndarray] = None
        self.loop_prev_processed: Optional[np.ndarray] = None
        self.loop_reference_frame_idx = 0
        self.loop_armed = False
        self.loop_already_reported = False
        self.loop_motion_score = 0.0
        self.loop_template_score = 0.0
        self.loop_hash_distance = 9999
        self.loop_candidate_count = 0
        self.progress_bar_fill = 0.0
        self.progress_bar_white_frac = 0.0
        self.progress_bar_max_fill_seen = 0.0
        self.progress_bar_candidate_count = 0
        self.progress_bar_armed = False
        self.progress_bar_already_reported = False
        self.ad_reference_features: Optional[Dict[str, np.ndarray]] = None
        self.ad_popup_detected = False
        self.ad_popup_already_reported = False
        self.ad_popup_candidate_count = 0
        self.ad_popup_inner_solid_frac = 0.0
        self.ad_popup_outer_match_frac = 1.0
        self.ad_popup_inner_edge_frac = 1.0
        self.ad_popup_inner_outer_distance = 0.0
        self.ad_popup_baseline_inner_diff = 0.0
        self.ad_popup_baseline_full_diff = 0.0

    def reset_baseline(self, frame: np.ndarray, frame_idx: int) -> None:
        self.reference_regions.clear()
        self.reference_hashes.clear()

        for name in ROIS:
            processed = preprocess_region(frame, name)
            self.reference_regions[name] = processed.copy()
            self.reference_hashes[name] = perceptual_hash(processed).copy()

        self.reference_profile_features = preprocess_profile_features(frame)
        self.ad_reference_features = preprocess_ad_popup(frame) if AD_POPUP_DETECTION_ENABLED else None

        loop_processed = preprocess_region(frame, "main_reel", LOOP_PROCESS_SIZE)
        self.loop_reference = loop_processed.copy()
        self.loop_reference_hash = perceptual_hash(loop_processed).copy()
        self.loop_prev_processed = loop_processed.copy()
        self.loop_reference_frame_idx = frame_idx
        self.loop_armed = False
        self.loop_already_reported = False
        self.loop_motion_score = 0.0
        self.loop_template_score = 0.0
        self.loop_hash_distance = 9999
        self.loop_candidate_count = 0
        self.progress_bar_fill = 0.0
        self.progress_bar_white_frac = 0.0
        self.progress_bar_max_fill_seen = 0.0
        self.progress_bar_candidate_count = 0
        self.progress_bar_armed = False
        self.progress_bar_already_reported = False
        self.ad_popup_detected = False
        self.ad_popup_already_reported = False
        self.ad_popup_candidate_count = 0
        self.ad_popup_inner_solid_frac = 0.0
        self.ad_popup_outer_match_frac = 1.0
        self.ad_popup_inner_edge_frac = 1.0
        self.ad_popup_inner_outer_distance = 0.0
        self.ad_popup_baseline_inner_diff = 0.0
        self.ad_popup_baseline_full_diff = 0.0

        self.stats = DetectorStats()
        self.state = DetectorState.WATCHING
        self.settle_frames_left = 0
        self.save_baseline_debug(frame)
        print(
            f"[BASELINE_SET] reel={self.reel_index} frame={frame_idx} "
            f"switch_mode={self.switch_mode} active_features={','.join(get_feature_weights(self.switch_mode).keys())} "
            f"loop_mode={LOOP_DETECTION_MODE} progress_angle={PROGRESS_BAR_ANGLE_DEGREES:.2f} "
            f"profile_circle_mask={PROFILE_USE_CIRCLE_MASK} ad_popup_detection={AD_POPUP_DETECTION_ENABLED}"
        )

    def update(self, frame: np.ndarray, frame_idx: int) -> list:
        events = []
        self.recent_frames.append(frame.copy())

        if self.state == DetectorState.INITIALIZING:
            self.reset_baseline(frame, frame_idx)
            return events

        if self.state == DetectorState.SETTLING_AFTER_SWITCH:
            self.settle_frames_left -= 1
            if self.settle_frames_left <= 0:
                self.reel_index += 1
                self.reset_baseline(frame, frame_idx)
            return events

        ad_detected_now = self.compare_ad_popup_to_baseline(frame)
        if ad_detected_now:
            print(
                f"[AD_POPUP_DETECTED] reel={self.reel_index} frame={frame_idx} "
                f"inner_solid={self.ad_popup_inner_solid_frac:.3f} "
                f"outer_match={self.ad_popup_outer_match_frac:.3f} "
                f"edge_frac={self.ad_popup_inner_edge_frac:.3f} "
                f"inner_outer_dist={self.ad_popup_inner_outer_distance:.2f} "
                f"base_inner_diff={self.ad_popup_baseline_inner_diff:.2f} "
                f"base_full_diff={self.ad_popup_baseline_full_diff:.2f} "
                f"matches={self.ad_popup_candidate_count}"
            )
            self.save_ad_popup_debug(frame, frame_idx)
            events.append("ad_popup")

        loop_detected = self.compare_loop_to_baseline(frame, frame_idx)
        if loop_detected:
            if LOOP_DETECTION_MODE == "progress_bar":
                print(
                    f"[LOOP_DETECTED] reel={self.reel_index} frame={frame_idx} mode=progress_bar "
                    f"bar_fill={self.progress_bar_fill:.3f} white_frac={self.progress_bar_white_frac:.3f} "
                    f"matches={self.progress_bar_candidate_count}"
                )
            else:
                print(
                    f"[LOOP_DETECTED] reel={self.reel_index} frame={frame_idx} mode=visual_match "
                    f"loop_t={self.loop_template_score:.3f} loop_h={self.loop_hash_distance} "
                    f"matches={self.loop_candidate_count}"
                )
            events.append("loop")

        new_reel_detected = self.compare_to_baseline(frame)
        if new_reel_detected:
            print(
                f"[NEW_REEL_DETECTED] old_reel={self.reel_index} frame={frame_idx} "
                f"mode={self.switch_mode} diff_score={self.stats.different_score:.1f} "
                f"same_score={self.stats.same_score:.1f} changed={self.changed_features_string()}"
            )
            self.save_switch_debug(frame, frame_idx)
            self.state = DetectorState.SETTLING_AFTER_SWITCH
            self.settle_frames_left = SETTLE_FRAMES_AFTER_SWITCH
            events.append("new_reel")

        return events

    def compare_ad_popup_to_baseline(self, frame: np.ndarray) -> bool:
        if not AD_POPUP_DETECTION_ENABLED:
            return False

        cur = preprocess_ad_popup(frame)
        if self.ad_reference_features is None:
            self.ad_reference_features = cur
            return False

        inner_mask = cur["inner_mask"] > 0

        self.ad_popup_inner_solid_frac = float(cur["inner_solid_match_frac"])
        self.ad_popup_outer_match_frac = float(cur["outer_match_frac"])
        self.ad_popup_inner_edge_frac = float(cur["inner_edge_frac"])
        self.ad_popup_inner_outer_distance = float(cur["inner_outer_median_distance"])

        gray_diff = cv2.absdiff(cur["gray"], self.ad_reference_features["gray"])
        self.ad_popup_baseline_inner_diff = float(np.mean(gray_diff[inner_mask])) if np.any(inner_mask) else 0.0
        self.ad_popup_baseline_full_diff = float(np.mean(gray_diff))

        inner_is_solid = self.ad_popup_inner_solid_frac >= AD_INNER_SOLID_MATCH_FRAC_MIN
        inner_is_not_too_edgy = self.ad_popup_inner_edge_frac <= AD_INNER_EDGE_FRAC_MAX

        # Core rule: the surrounding outer ring should not share much of the
        # inner banner's dominant color. If it does, this is likely just reel
        # content/background, not a popup overlay.
        exterior_does_not_match_inner = self.ad_popup_outer_match_frac <= AD_OUTER_MATCH_FRAC_MAX

        # Secondary separation guard: medians should also be perceptually different.
        sharp_inner_vs_exterior = self.ad_popup_inner_outer_distance >= AD_INNER_OUTER_MEDIAN_DISTANCE_MIN
        very_strong_inner_vs_exterior = self.ad_popup_inner_outer_distance >= AD_INNER_OUTER_STRONG_DISTANCE_MIN

        changed_from_baseline = (
            self.ad_popup_baseline_inner_diff >= AD_BASELINE_INNER_DIFF_THRESHOLD
            or self.ad_popup_baseline_full_diff >= AD_BASELINE_FULL_DIFF_THRESHOLD
        )

        candidate = (
            inner_is_solid
            and inner_is_not_too_edgy
            and exterior_does_not_match_inner
            and sharp_inner_vs_exterior
            and ((not AD_REQUIRE_BASELINE_CHANGE) or changed_from_baseline or very_strong_inner_vs_exterior)
        )

        if candidate:
            self.ad_popup_candidate_count += 1
        else:
            self.ad_popup_candidate_count = 0

        self.ad_popup_detected = self.ad_popup_candidate_count >= AD_CONSECUTIVE_FRAMES_REQUIRED

        if self.ad_popup_detected and not self.ad_popup_already_reported:
            self.ad_popup_already_reported = True
            return True

        return False

    def save_ad_popup_debug(self, frame: np.ndarray, frame_idx: int) -> None:
        if not SAVE_DEBUG_FRAMES:
            return

        folder = os.path.join(
            DEBUG_DIR,
            f"reel_{self.reel_index:03d}",
            f"ad_popup_detected_frame_{frame_idx:06d}",
        )
        ensure_dir(folder)
        cv2.imwrite(os.path.join(folder, "full_frame.jpg"), frame)
        annotated = frame.copy()
        draw_rois(annotated)
        cv2.imwrite(os.path.join(folder, "full_frame_annotated.jpg"), annotated)
        save_ad_popup_debug_images(folder, frame, "current")

        if self.ad_reference_features is not None:
            cv2.imwrite(os.path.join(folder, "baseline_ad_popup_gray.jpg"), self.ad_reference_features["gray"])
            cv2.imwrite(os.path.join(folder, "baseline_ad_popup_edges.jpg"), self.ad_reference_features["edges"])
            cv2.imwrite(os.path.join(folder, "baseline_ad_popup_inner_mask.jpg"), self.ad_reference_features["inner_mask"])
            cv2.imwrite(os.path.join(folder, "baseline_ad_popup_exterior_mask.jpg"), self.ad_reference_features["exterior_mask"])

    def compare_progress_bar_to_completion(self, frame: np.ndarray, frame_idx: int) -> bool:
        fill, white_frac, _, _ = estimate_progress_bar_fill(frame)
        self.progress_bar_fill = fill
        self.progress_bar_white_frac = white_frac
        self.progress_bar_max_fill_seen = max(self.progress_bar_max_fill_seen, fill)

        frames_since_reference = frame_idx - self.loop_reference_frame_idx

        # Arm once the progress bar has shown meaningful nonzero progress. This
        # prevents a random bright artifact at startup from instantly counting as done.
        if fill >= PROGRESS_BAR_ARM_MIN_FRAC:
            self.progress_bar_armed = True

        complete = (
            self.progress_bar_armed
            and frames_since_reference >= PROGRESS_BAR_MIN_FRAMES_BEFORE_DETECT
            and fill >= PROGRESS_BAR_COMPLETE_FRAC
            and white_frac >= PROGRESS_BAR_MIN_WHITE_FRAC_COMPLETE
        )

        if complete:
            self.progress_bar_candidate_count += 1
        else:
            self.progress_bar_candidate_count = 0

        if (
            self.progress_bar_candidate_count >= PROGRESS_BAR_CONSECUTIVE_FRAMES_REQUIRED
            and not self.progress_bar_already_reported
        ):
            self.progress_bar_already_reported = True
            return True

        return False

    def compare_loop_to_baseline(self, frame: np.ndarray, frame_idx: int) -> bool:
        if LOOP_DETECTION_MODE == "progress_bar":
            return self.compare_progress_bar_to_completion(frame, frame_idx)

        processed = preprocess_region(frame, "main_reel", LOOP_PROCESS_SIZE)
        current_hash = perceptual_hash(processed)

        self.loop_motion_score = motion_between(processed, self.loop_prev_processed)
        self.loop_template_score = 0.0
        self.loop_hash_distance = 9999

        if self.loop_reference is None or self.loop_reference_hash is None:
            self.loop_prev_processed = processed.copy()
            return False

        if not self.loop_armed:
            if self.loop_motion_score >= LOOP_MOTION_ARM_THRESHOLD:
                self.loop_armed = True
            self.loop_prev_processed = processed.copy()
            return False

        frames_since_reference = frame_idx - self.loop_reference_frame_idx
        self.loop_template_score = template_similarity(processed, self.loop_reference)
        self.loop_hash_distance = hash_distance(current_hash, self.loop_reference_hash)

        similar_to_reference = (
            frames_since_reference >= LOOP_MIN_FRAMES_BEFORE_DETECT
            and self.loop_template_score >= LOOP_TEMPLATE_THRESHOLD
            and self.loop_hash_distance <= LOOP_HASH_DISTANCE_THRESHOLD
        )

        if similar_to_reference:
            self.loop_candidate_count += 1
        else:
            self.loop_candidate_count = 0

        self.loop_prev_processed = processed.copy()

        if (
            self.loop_candidate_count >= LOOP_CONSECUTIVE_FRAMES_REQUIRED
            and not self.loop_already_reported
        ):
            self.loop_already_reported = True
            return True

        return False

    def compare_to_baseline(self, frame: np.ndarray) -> bool:
        same_score = 0.0
        different_score = 0.0
        statuses: Dict[str, RegionStats] = {}

        for name, weight in get_feature_weights(self.switch_mode).items():
            if name == "profile_picture" and PROFILE_ROBUST_MODE:
                cur_features = preprocess_profile_features(frame)
                if self.reference_profile_features is None:
                    self.reference_profile_features = cur_features

                t_score, h_dist, edge_score, color_score, status, lighting_shift = compare_profile_features(
                    cur_features,
                    self.reference_profile_features,
                )

                if status == RegionStatus.SAME:
                    same_score += weight
                    if PROFILE_ADAPT_ON_SAME:
                        self.reference_profile_features = blend_profile_features(
                            self.reference_profile_features,
                            cur_features,
                            PROFILE_BASELINE_ADAPT_ALPHA,
                        )
                elif status == RegionStatus.LIGHTING_SHIFT:
                    # Treat as same for switch logic, but keep the separate label for debugging.
                    same_score += weight
                    if PROFILE_ADAPT_ON_LIGHTING_SHIFT:
                        self.reference_profile_features = blend_profile_features(
                            self.reference_profile_features,
                            cur_features,
                            PROFILE_BASELINE_ADAPT_ALPHA,
                        )
                elif status == RegionStatus.DIFFERENT:
                    different_score += weight

                statuses[name] = RegionStats(
                    template_score=t_score,
                    hash_distance=h_dist,
                    status=status,
                    edge_score=edge_score,
                    color_score=color_score,
                    lighting_shift=lighting_shift,
                )
                continue

            t_score, h_dist, status, _ = compare_region_to_reference(
                frame,
                name,
                self.reference_regions,
                self.reference_hashes,
            )

            if status == RegionStatus.SAME:
                same_score += weight
            elif status == RegionStatus.DIFFERENT:
                different_score += weight

            statuses[name] = RegionStats(t_score, h_dist, status)

        candidate = (
            different_score >= get_different_score_threshold(self.switch_mode)
            and different_score > same_score
        )

        if candidate:
            self.stats.candidate_count += 1
        else:
            self.stats.candidate_count = 0

        self.stats.same_score = same_score
        self.stats.different_score = different_score
        self.stats.statuses = statuses

        return self.stats.candidate_count >= get_consecutive_frames_required(self.switch_mode)

    def set_switch_mode(self, mode: str) -> None:
        self.switch_mode = validate_switch_mode(mode)
        self.stats.candidate_count = 0
        print(f"[MODE_SET] switch_mode={self.switch_mode} active_features={','.join(get_feature_weights(self.switch_mode).keys())}")

    def toggle_switch_mode(self) -> None:
        next_mode = "engagement" if self.switch_mode == "profile_only" else "profile_only"
        self.set_switch_mode(next_mode)

    def changed_features_string(self) -> str:
        changed = [
            name for name, stats in self.stats.statuses.items()
            if stats.status == RegionStatus.DIFFERENT
        ]
        return ",".join(changed) if changed else "none"

    def manual_reset(self, frame: np.ndarray, frame_idx: int) -> None:
        self.reset_baseline(frame, frame_idx)

    def save_manual_frame(self, frame: np.ndarray, annotated: np.ndarray, frame_idx: int) -> None:
        if not SAVE_DEBUG_FRAMES:
            return
        folder = os.path.join(DEBUG_DIR, "manual_frames")
        ensure_dir(folder)
        cv2.imwrite(os.path.join(folder, f"frame_{frame_idx:06d}_raw.jpg"), frame)
        cv2.imwrite(os.path.join(folder, f"frame_{frame_idx:06d}_annotated.jpg"), annotated)
        print(f"[SAVED_FRAME] frame={frame_idx} folder={folder}")

    def save_baseline_debug(self, frame: np.ndarray) -> None:
        if not SAVE_DEBUG_FRAMES:
            return
        folder = os.path.join(DEBUG_DIR, f"reel_{self.reel_index:03d}", "baseline")
        ensure_dir(folder)
        cv2.imwrite(os.path.join(folder, "full_frame.jpg"), frame)
        annotated = frame.copy()
        draw_rois(annotated)
        cv2.imwrite(os.path.join(folder, "full_frame_annotated.jpg"), annotated)
        for name in ROIS:
            cv2.imwrite(os.path.join(folder, f"{name}.jpg"), crop_roi(frame, name))
        save_masked_profile_debug_images(folder, frame, "baseline")
        save_progress_bar_debug_images(folder, frame, "baseline")
        save_ad_popup_debug_images(folder, frame, "baseline")
        if self.loop_reference is not None:
            cv2.imwrite(os.path.join(folder, "main_reel_loop_reference_processed.jpg"), self.loop_reference)

    def save_switch_debug(self, frame: np.ndarray, frame_idx: int) -> None:
        if not SAVE_DEBUG_FRAMES:
            return
        folder = os.path.join(
            DEBUG_DIR,
            f"reel_{self.reel_index:03d}",
            f"switch_detected_frame_{frame_idx:06d}",
        )
        ensure_dir(folder)

        cv2.imwrite(os.path.join(folder, "full_frame.jpg"), frame)
        annotated = frame.copy()
        draw_rois(annotated)
        cv2.imwrite(os.path.join(folder, "full_frame_annotated.jpg"), annotated)

        for name in ROIS:
            cv2.imwrite(os.path.join(folder, f"current_{name}.jpg"), crop_roi(frame, name))
            if name in self.reference_regions:
                cv2.imwrite(os.path.join(folder, f"baseline_processed_{name}.jpg"), self.reference_regions[name])

        save_masked_profile_debug_images(folder, frame, "current")
        save_progress_bar_debug_images(folder, frame, "current")
        save_ad_popup_debug_images(folder, frame, "current")
        if "profile_picture" in self.reference_regions:
            ref_profile = self.reference_regions["profile_picture"].copy()
            if PROFILE_USE_CIRCLE_MASK:
                mask = profile_process_mask()
                ref_profile[~mask] = 0
            cv2.imwrite(os.path.join(folder, "baseline_profile_masked_processed.jpg"), ref_profile)

        for i, raw in enumerate(self.recent_frames):
            cv2.imwrite(os.path.join(folder, f"recent_{i:02d}.jpg"), raw)


class ServoCommander:
    def __init__(self, session: requests.Session):
        self.session = session
        self.last_command_time = 0.0

    def can_send(self) -> bool:
        return (time.time() - self.last_command_time) >= SERVO_COOLDOWN_SECONDS

    def send(self, cmd: str, reason: str = "") -> bool:
        if not ENABLE_AUTO_SCROLL:
            print(f"[SERVO_DISABLED] cmd={cmd} reason={reason}")
            return False

        if not self.can_send():
            print(f"[SERVO_SKIPPED_COOLDOWN] cmd={cmd} reason={reason}")
            return False

        try:
            r = self.session.get(
                SERVO_BASE_URL,
                params={"cmd": cmd},
                timeout=SERVO_COMMAND_TIMEOUT_SECONDS,
            )
            ok = r.status_code == 200
            print(f"[SERVO_COMMAND] cmd={cmd} reason={reason} status={r.status_code} text={r.text!r}")
            if ok:
                self.last_command_time = time.time()
            return ok

        except requests.RequestException as e:
            print(f"[SERVO_COMMAND_ERROR] cmd={cmd} reason={reason} error={repr(e)}")
            return False


def handle_detector_events(events: list, detector: ThreeMetricReelSwitchDetector, servo: ServoCommander) -> None:
    if not events:
        return

    # Prefer loop/ad scroll triggers over new-reel. New-reel is mainly a confirmation signal.
    if "loop" in events and SCROLL_ON_LOOP:
        if servo.send("forward", reason="loop_detected"):
            time.sleep(RESET_BASELINE_AFTER_SCROLL_DELAY_SECONDS)
            detector.state = DetectorState.SETTLING_AFTER_SWITCH
            detector.settle_frames_left = SETTLE_FRAMES_AFTER_SWITCH
        return

    if "ad_popup" in events and SCROLL_ON_AD_POPUP:
        if servo.send("forward", reason="ad_popup_detected"):
            time.sleep(RESET_BASELINE_AFTER_SCROLL_DELAY_SECONDS)
            detector.state = DetectorState.SETTLING_AFTER_SWITCH
            detector.settle_frames_left = SETTLE_FRAMES_AFTER_SWITCH
        return

    if "new_reel" in events and SCROLL_ON_NEW_REEL:
        servo.send("forward", reason="new_reel_detected")


def get_frame(session: requests.Session):
    try:
        r = session.get(URL, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        print("[REQUEST_ERROR]", repr(e))
        return None, None

    if r.status_code != 200:
        print("[BAD_STATUS]", r.status_code)
        return None, None

    if r.headers.get("content-type") != "image/jpeg":
        print("[BAD_CONTENT_TYPE]", r.headers.get("content-type"))
        return None, None

    img_bytes = np.frombuffer(r.content, dtype=np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        print("[DECODE_FAILED]")
        return None, None

    return frame, len(r.content)


def draw_rois(frame: np.ndarray) -> None:
    for name, roi in ROIS.items():
        if name == "progress_bar":
            corners = progress_bar_rotated_corners(frame).astype(np.int32)
            cv2.polylines(frame, [corners], isClosed=True, color=ROI_COLOR, thickness=ROI_THICKNESS)

            if PROGRESS_BAR_DRAW_CENTERLINE:
                left_mid = ((corners[0] + corners[3]) / 2.0).astype(np.int32)
                right_mid = ((corners[1] + corners[2]) / 2.0).astype(np.int32)
                cv2.line(frame, tuple(left_mid), tuple(right_mid), ROI_COLOR, ROI_THICKNESS)
            continue

        x, y, w, h = norm_roi_to_pixels(frame, roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), ROI_COLOR, ROI_THICKNESS)

        if name == "ad_popup" and AD_POPUP_DETECTION_ENABLED:
            ix = x + int(round(AD_INNER_X_FRAC * w))
            iy = y + int(round(AD_INNER_Y_FRAC * h))
            iw = int(round(AD_INNER_W_FRAC * w))
            ih = int(round(AD_INNER_H_FRAC * h))
            ix2 = max(ix + 1, min(x + w, ix + iw))
            iy2 = max(iy + 1, min(y + h, iy + ih))
            cv2.rectangle(frame, (ix, iy), (ix2, iy2), AD_INNER_ROI_COLOR, ROI_THICKNESS)

        if name == "profile_picture" and PROFILE_USE_CIRCLE_MASK and PROFILE_DRAW_CIRCLE_MASK:
            cx = x + int(round(PROFILE_CIRCLE_CENTER_X_FRAC * (w - 1)))
            cy = y + int(round(PROFILE_CIRCLE_CENTER_Y_FRAC * (h - 1)))
            radius = int(round(PROFILE_CIRCLE_RADIUS_FRAC * min(w, h)))
            cv2.circle(frame, (cx, cy), radius, ROI_COLOR, ROI_THICKNESS)


def draw_status_overlay(frame: np.ndarray, detector: ThreeMetricReelSwitchDetector) -> None:
    line1 = (
        f"state={detector.state.value} reel={detector.reel_index} "
        f"switch_diff={detector.stats.different_score:.1f} switch_same={detector.stats.same_score:.1f} "
        f"switch_mode={detector.switch_mode} switch_cand={detector.stats.candidate_count}/{get_consecutive_frames_required(detector.switch_mode)}"
    )
    line2 = (
        f"loop_mode={LOOP_DETECTION_MODE} bar_fill={detector.progress_bar_fill:.3f} "
        f"bar_cand={detector.progress_bar_candidate_count}/{PROGRESS_BAR_CONSECUTIVE_FRAMES_REQUIRED} "
        f"visual_loop_t={detector.loop_template_score:.3f}"
    )
    cv2.putText(frame, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ROI_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ROI_COLOR, 1, cv2.LINE_AA)


def main() -> None:
    session = requests.Session()
    detector = ThreeMetricReelSwitchDetector()
    servo = ServoCommander(session)
    frame_idx = 0
    last_frame = None

    print("Starting reel switch + loop detector.")
    print("Terminal output is event-only, with loop events and switch events. The display window only shows ROI boxes.")
    print(f"Initial switch mode: {SWITCH_DETECTION_MODE}")
    print(f"Loop detection mode: {LOOP_DETECTION_MODE}")
    print(f"Progress bar angle degrees: {PROGRESS_BAR_ANGLE_DEGREES}")
    print(f"Ad popup detection enabled: {AD_POPUP_DETECTION_ENABLED}")
    print(f"Auto scroll enabled: {ENABLE_AUTO_SCROLL}")
    print(f"Scroll triggers: loop={SCROLL_ON_LOOP}, ad_popup={SCROLL_ON_AD_POPUP}, new_reel={SCROLL_ON_NEW_REEL}")
    print("Controls: q=quit, r=reset baseline, m=toggle switch mode, s=save raw+annotated frame")
    print("Servo controls: 1=forward, 2=backward, 3=doubletap, h=home")

    try:
        while True:
            frame, _ = get_frame(session)

            if frame is None:
                time.sleep(ERROR_SLEEP_SECONDS)
                continue

            last_frame = frame.copy()
            events = detector.update(frame, frame_idx)
            handle_detector_events(events, detector, servo)

            display = frame.copy()
            if DRAW_ROIS:
                draw_rois(display)

            cv2.imshow(DISPLAY_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[QUIT]")
                break
            if key == ord("r"):
                print("[MANUAL_RESET]")
                detector.manual_reset(frame, frame_idx)
            if key == ord("m"):
                detector.toggle_switch_mode()
            if key == ord("s"):
                detector.save_manual_frame(frame, display, frame_idx)
            if key == ord("1"):
                servo.send("forward", reason="manual_key_1")
            if key == ord("2"):
                servo.send("backward", reason="manual_key_2")
            if key == ord("3"):
                servo.send("doubletap", reason="manual_key_3")
            if key == ord("h"):
                servo.send("home", reason="manual_key_h")

            frame_idx += 1

    finally:
        if last_frame is not None and SAVE_DEBUG_FRAMES:
            ensure_dir(DEBUG_DIR)
            cv2.imwrite(os.path.join(DEBUG_DIR, "last_frame.jpg"), last_frame)
            annotated = last_frame.copy()
            draw_rois(annotated)
            cv2.imwrite(os.path.join(DEBUG_DIR, "last_frame_annotated.jpg"), annotated)
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

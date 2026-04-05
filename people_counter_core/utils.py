from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
from .config import (
    ALLOWED_EXTENSIONS, DUPLICATE_IOU_THRES, MAX_ASPECT_RATIO,
    MAX_BOX_AREA, MIN_BOX_AREA, MIN_SCORE_FOR_TRACK
)

def clamp_rect(rect, w, h):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def expand_rect(rect, margin, w, h):
    x1, y1, x2, y2 = rect
    return [max(0, x1 - margin), max(0, y1 - margin), min(w - 1, x2 + margin), min(h - 1, y2 + margin)]

def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / max(1, union)

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def center_distance(a, b):
    ax, ay = center_of(a)
    bx, by = center_of(b)
    return math.hypot(ax - bx, ay - by)

def extract_hist(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32,), dtype=np.float32)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((32,), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)

def hist_similarity(h1, h2):
    if h1 is None or h2 is None or len(h1) == 0 or len(h2) == 0:
        return 0.0
    score = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return float((score + 1.0) / 2.0)

def is_valid_person_box(box, score):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    area = w * h; aspect = h / w
    if score < MIN_SCORE_FOR_TRACK: return False
    if area < MIN_BOX_AREA or area > MAX_BOX_AREA: return False
    if aspect > MAX_ASPECT_RATIO: return False
    return True

def deduplicate_detections(dets):
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    kept = []
    for box, score, hist in dets:
        dup = False
        for kbox, _, _ in kept:
            if iou(box, kbox) >= DUPLICATE_IOU_THRES:
                dup = True
                break
        if not dup:
            kept.append((box, score, hist))
    return kept

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return cleaned[:80] or "video"

def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

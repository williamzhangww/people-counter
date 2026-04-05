from __future__ import annotations
import csv
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import torch
from ultralytics import YOLO

from .config import (
    CONF_THRES,
    CROP_MARGIN,
    DEVICE_AUTO,
    DEVICE_PREFERENCE,
    DRAW_TRACK_IDS,
    FONT_BIG,
    FULLSCREEN,
    IOU_NMS_THRES,
    MODEL_NAME,
    OUTPUT_DIR,
    PERSON_CLASS_ID,
    ROI,
    SHOW_LIVE_WINDOW,
    SHOW_REGION_RECT,
    THICK_BIG,
    USE_FULL_FRAME,
    WINDOW_NAME,
    MIN_TRACK_FRAMES_TO_COUNT,
    TRACK_MAX_AGE_FRAMES,
    VISIT_GAP_SECONDS,
    USAGE_MIN_SECONDS,
)
from .tracker import HybridTracker
from .utils import (
    allowed_file,
    clamp_rect,
    deduplicate_detections,
    expand_rect,
    extract_hist,
    format_seconds,
    is_valid_person_box,
    safe_stem,
)


def resolve_device() -> str:
    if not DEVICE_AUTO:
        if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        return DEVICE_PREFERENCE
    if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_model_file() -> Path:
    model_path = Path(MODEL_NAME)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    downloaded_path = None

    try:
        from ultralytics.utils.downloads import attempt_download_asset
        downloaded = attempt_download_asset(model_path.name)
        if downloaded:
            downloaded_path = Path(downloaded)
    except Exception:
        downloaded_path = None

    if downloaded_path is None or not downloaded_path.exists():
        model = YOLO(model_path.name)
        candidates = [
            getattr(model, "ckpt_path", None),
            getattr(getattr(model, "model", None), "pt_path", None),
            getattr(getattr(model, "predictor", None), "model_path", None),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                downloaded_path = Path(candidate)
                break

    if downloaded_path and downloaded_path.exists():
        if downloaded_path.resolve() != model_path.resolve():
            shutil.copy2(downloaded_path, model_path)
        return model_path

    raise FileNotFoundError(
        f"Could not download model automatically. Please place {model_path.name} in the models folder."
    )


def output_paths(base_name: Path):
    stem = safe_stem(base_name.name)
    return {
        "usage_events_csv": OUTPUT_DIR / f"{stem}_usage_events.csv",
        "visit_events_csv": OUTPUT_DIR / f"{stem}_visit_events.csv",
        "annotated_video": OUTPUT_DIR / f"{stem}-annotated.mp4",
    }


def _finalize_event_row(row_id: int, start_frame: int, end_frame: int, visible_frames: int, fps: float):
    start_seconds = max(0.0, (start_frame - 1) / fps)
    end_seconds = max(start_seconds, (end_frame - 1) / fps)
    duration_seconds = max(0.0, visible_frames / fps)
    return {
        "id": row_id,
        "start_time": round(start_seconds, 3),
        "end_time": round(end_seconds, 3),
        "duration": round(duration_seconds, 3),
    }


def run_pipeline(video_path: str | Path | None = None, show_live_window: bool | None = None, progress_callback: Optional[Callable[[int, str], None]] = None, output_name: str | None = None):
    if video_path is None:
        raise RuntimeError("No input video path was provided.")
    video_path = Path(video_path)
    if not allowed_file(video_path.name):
        raise RuntimeError(f"Unsupported file type: {video_path.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(output_name) if output_name else video_path
    paths = output_paths(base_name)
    live_window = SHOW_LIVE_WINDOW if show_live_window is None else show_live_window
    device = resolve_device()

    if progress_callback:
        progress_callback(1, f"Initializing model on {device}...")
    model_path = ensure_model_file()
    model = YOLO(str(model_path))
    model.to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")
    h, w = first.shape[:2]
    roi = [0, 0, w - 1, h - 1] if USE_FULL_FRAME else clamp_rect(ROI, w, h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if live_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if FULLSCREEN:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    writer = cv2.VideoWriter(str(paths["annotated_video"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create annotated video: {paths['annotated_video']}")

    tracker = HybridTracker()
    active_events: Dict[int, Dict[str, int]] = {}
    visit_events = []
    usage_events = []
    next_visit_id = 1
    next_usage_id = 1
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_wall_time = time.time()
    allowed_gap_frames = max(TRACK_MAX_AGE_FRAMES, int(round(VISIT_GAP_SECONDS * fps)))
    visible_persons_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_rel = (frame_idx - 1) / fps
        if progress_callback and total_frames > 0 and frame_idx % 5 == 0:
            pct = max(2, min(92, int((frame_idx / total_frames) * 92)))
            elapsed = max(0.001, time.time() - start_wall_time)
            frames_per_sec = frame_idx / elapsed
            remaining_frames = max(0, total_frames - frame_idx)
            eta_sec = remaining_frames / max(0.001, frames_per_sec)
            progress_callback(pct, f"Processing frame {frame_idx}/{total_frames} on {device}... ETA {format_seconds(eta_sec)}")

        crop_rect = [0, 0, w - 1, h - 1] if USE_FULL_FRAME else expand_rect(roi, CROP_MARGIN, w, h)
        cx1, cy1, cx2, cy2 = crop_rect
        crop = frame[cy1:cy2, cx1:cx2]

        results = model(crop, classes=[PERSON_CLASS_ID], conf=CONF_THRES, iou=IOU_NMS_THRES, verbose=False, device=device)
        detections = []
        if results:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.int().cpu().tolist()
                confs = r.boxes.conf.float().cpu().tolist()
                for box, score in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    global_box = (x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1)
                    if not is_valid_person_box(global_box, score):
                        continue
                    hist = extract_hist(frame, global_box)
                    detections.append((global_box, score, hist))

        detections = deduplicate_detections(detections)
        visible_tracks = tracker.update(detections, frame_idx)

        visible_ids_this_frame = set()
        counted_tracks = []
        for tid, bbox, hits in visible_tracks:
            if hits >= MIN_TRACK_FRAMES_TO_COUNT:
                visible_ids_this_frame.add(tid)
                counted_tracks.append((tid, bbox, hits))

        visible_persons_history.append(len(visible_ids_this_frame))

        for tid in sorted(visible_ids_this_frame):
            state = active_events.get(tid)
            if state is None:
                active_events[tid] = {
                    "start_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "visible_frames": 1,
                }
            else:
                state["last_seen_frame"] = frame_idx
                state["visible_frames"] += 1

        finalizable_ids = []
        for tid, state in active_events.items():
            if tid in visible_ids_this_frame:
                continue
            gap_frames = frame_idx - state["last_seen_frame"]
            if gap_frames > allowed_gap_frames:
                finalizable_ids.append(tid)

        for tid in sorted(finalizable_ids):
            state = active_events.pop(tid)
            visit_row = _finalize_event_row(
                row_id=next_visit_id,
                start_frame=state["start_frame"],
                end_frame=state["last_seen_frame"],
                visible_frames=state["visible_frames"],
                fps=fps,
            )
            visit_events.append({
                "visit_id": visit_row["id"],
                "start_time": visit_row["start_time"],
                "end_time": visit_row["end_time"],
                "duration": visit_row["duration"],
            })
            next_visit_id += 1

            if visit_row["duration"] >= USAGE_MIN_SECONDS:
                usage_events.append({
                    "event_id": next_usage_id,
                    "start_time": visit_row["start_time"],
                    "end_time": visit_row["end_time"],
                    "duration": visit_row["duration"],
                })
                next_usage_id += 1

        display = frame.copy()
        if SHOW_REGION_RECT:
            cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
        for tid, bbox, hits in counted_tracks:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if DRAW_TRACK_IDS:
                cv2.putText(display, f"ID {tid}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        usage_count = len(usage_events)
        visit_count = len(visit_events)
        engagement_rate = (usage_count / visit_count * 100.0) if visit_count > 0 else 0.0
        cv2.putText(display, f"People now: {len(visible_ids_this_frame)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_BIG, (0, 0, 255), THICK_BIG)
        cv2.putText(display, f"Video time: {t_rel:.1f}s", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display, f"Visits completed: {visit_count}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display, f"Usages completed: {usage_count}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display, f"Engagement: {engagement_rate:.1f}%", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(display)

        if live_window:
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    for tid in sorted(active_events.keys()):
        state = active_events[tid]
        visit_row = _finalize_event_row(
            row_id=next_visit_id,
            start_frame=state["start_frame"],
            end_frame=state["last_seen_frame"],
            visible_frames=state["visible_frames"],
            fps=fps,
        )
        visit_events.append({
            "visit_id": visit_row["id"],
            "start_time": visit_row["start_time"],
            "end_time": visit_row["end_time"],
            "duration": visit_row["duration"],
        })
        next_visit_id += 1
        if visit_row["duration"] >= USAGE_MIN_SECONDS:
            usage_events.append({
                "event_id": next_usage_id,
                "start_time": visit_row["start_time"],
                "end_time": visit_row["end_time"],
                "duration": visit_row["duration"],
            })
            next_usage_id += 1

    cap.release()
    writer.release()
    if live_window:
        cv2.destroyAllWindows()

    if progress_callback:
        progress_callback(95, "Writing usage events CSV...")
    with open(paths["usage_events_csv"], "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["event_id", "start_time", "end_time", "duration"])
        for row in usage_events:
            writer_csv.writerow([row["event_id"], row["start_time"], row["end_time"], row["duration"]])

    if progress_callback:
        progress_callback(97, "Writing visit events CSV...")
    with open(paths["visit_events_csv"], "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["visit_id", "start_time", "end_time", "duration"])
        for row in visit_events:
            writer_csv.writerow([row["visit_id"], row["start_time"], row["end_time"], row["duration"]])

    if progress_callback:
        progress_callback(99, "Finalizing...")

    total_visits = len(visit_events)
    total_usages = len(usage_events)
    engagement_rate = round((total_usages / total_visits), 4) if total_visits > 0 else 0.0
    max_concurrent_people = max(visible_persons_history, default=0)
    avg_concurrent_people = round(sum(visible_persons_history) / len(visible_persons_history), 3) if visible_persons_history else 0.0

    return {
        "usage_events_csv": str(paths["usage_events_csv"]),
        "visit_events_csv": str(paths["visit_events_csv"]),
        "annotated_video": str(paths["annotated_video"]),
        "total_visits": total_visits,
        "total_usages": total_usages,
        "engagement_rate": engagement_rate,
        "max_concurrent_people": max_concurrent_people,
        "avg_concurrent_people": avg_concurrent_people,
        "video_path": str(video_path),
        "device": device,
        "usage_definition_seconds": USAGE_MIN_SECONDS,
        "gap_tolerance_seconds": VISIT_GAP_SECONDS,
    }

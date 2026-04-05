from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .config import (
    CENTER_DIST_NORM, MATURE_HITS, MATCH_CENTER_WEIGHT, MATCH_HIST_WEIGHT,
    MATCH_IOU_WEIGHT, MIN_MATCH_SCORE, REID_CENTER_DIST, REID_HIST_SIM_THRES,
    REID_MAX_GAP_FRAMES, SUPPRESS_NEW_NEARBY_IOU, TRACK_MAX_AGE_FRAMES,
)
from .utils import center_distance, hist_similarity, iou

@dataclass
class Track:
    tid: int
    bbox: Tuple[int, int, int, int]
    hist: np.ndarray
    hits: int = 1
    age_since_seen: int = 0
    last_seen_frame: int = 0
    counted_as_unique: bool = False

class HybridTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.recently_lost: List[Track] = []

    def _match_score(self, track: Track, det_box, det_hist):
        iou_score = iou(track.bbox, det_box)
        center_score = max(0.0, 1.0 - center_distance(track.bbox, det_box) / CENTER_DIST_NORM)
        hist_score = hist_similarity(track.hist, det_hist)
        return MATCH_IOU_WEIGHT * iou_score + MATCH_CENTER_WEIGHT * center_score + MATCH_HIST_WEIGHT * hist_score

    def _try_reid(self, det_box, det_hist, frame_idx):
        best = None; best_score = -1.0
        for old in self.recently_lost:
            gap = frame_idx - old.last_seen_frame
            if gap > REID_MAX_GAP_FRAMES: continue
            if center_distance(old.bbox, det_box) > REID_CENTER_DIST: continue
            sim = hist_similarity(old.hist, det_hist)
            if sim >= REID_HIST_SIM_THRES and sim > best_score:
                best_score = sim; best = old
        if best is None:
            return None
        self.recently_lost = [t for t in self.recently_lost if t.tid != best.tid]
        best.bbox = det_box; best.hist = det_hist; best.hits += 1; best.age_since_seen = 0; best.last_seen_frame = frame_idx
        self.tracks[best.tid] = best
        return best.tid

    def update(self, detections, frame_idx):
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs = []
        for tid, tr in self.tracks.items():
            for di, (dbox, _, dhist) in enumerate(detections):
                pairs.append((self._match_score(tr, dbox, dhist), tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in pairs:
            if score < MIN_MATCH_SCORE: break
            if tid in unmatched_tracks and di in unmatched_dets:
                tr = self.tracks[tid]; dbox, _, dhist = detections[di]
                tr.bbox = dbox; tr.hist = dhist; tr.hits += 1; tr.age_since_seen = 0; tr.last_seen_frame = frame_idx
                unmatched_tracks.remove(tid); unmatched_dets.remove(di)

        to_remove = []
        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]; tr.age_since_seen += 1
            if tr.age_since_seen > TRACK_MAX_AGE_FRAMES: to_remove.append(tid)

        for tid in to_remove:
            self.recently_lost.append(self.tracks.pop(tid))
        self.recently_lost = [t for t in self.recently_lost if frame_idx - t.last_seen_frame <= REID_MAX_GAP_FRAMES]

        for di in list(unmatched_dets):
            dbox, _, dhist = detections[di]
            near_mature = False
            for tr in self.tracks.values():
                if tr.hits >= MATURE_HITS and (iou(tr.bbox, dbox) >= SUPPRESS_NEW_NEARBY_IOU or center_distance(tr.bbox, dbox) < 60):
                    near_mature = True; break
            if near_mature: continue
            reused_tid = self._try_reid(dbox, dhist, frame_idx)
            if reused_tid is not None: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid=tid, bbox=dbox, hist=dhist, hits=1, age_since_seen=0, last_seen_frame=frame_idx, counted_as_unique=False)

        return [(tid, tr.bbox, tr.hits) for tid, tr in self.tracks.items() if tr.age_since_seen == 0]

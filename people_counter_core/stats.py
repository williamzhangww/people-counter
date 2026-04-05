from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set
from .config import WINDOW_SECONDS

@dataclass
class WindowBucket:
    index: int
    start_s: float
    end_s: float
    unique_ids: Set[int] = field(default_factory=set)
    new_ids: Set[int] = field(default_factory=set)
    frame_counts: List[int] = field(default_factory=list)

    def to_row(self):
        avg_people = sum(self.frame_counts) / len(self.frame_counts) if self.frame_counts else 0.0
        max_people = max(self.frame_counts) if self.frame_counts else 0
        min_people = min(self.frame_counts) if self.frame_counts else 0
        return [self.index, f"{self.start_s:.3f}", f"{self.end_s:.3f}", len(self.unique_ids), len(self.new_ids), f"{avg_people:.3f}", max_people, min_people, len(self.frame_counts)]

def get_bucket(window_map: Dict[int, WindowBucket], bucket_index: int) -> WindowBucket:
    if bucket_index not in window_map:
        start_s = bucket_index * WINDOW_SECONDS
        end_s = (bucket_index + 1) * WINDOW_SECONDS
        window_map[bucket_index] = WindowBucket(bucket_index, start_s, end_s)
    return window_map[bucket_index]

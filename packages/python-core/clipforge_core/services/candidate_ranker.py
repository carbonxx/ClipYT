"""
ClipForge AI — Candidate Ranking, Scene Snapping & Duration Clamping Service
Deduplicates candidate clips, snaps start/end times to nearest scene cut boundaries,
enforces min/max duration via boundary-aware clamping, and sorts by composite score.
"""
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def clamp_to_boundary(
    start_sec: float,
    end_sec: float,
    max_length_sec: float,
    min_length_sec: float,
    transcript_segments: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    tolerance_sec: float = 15.0,
) -> Tuple[float, float, str]:
    """
    Clamp a clip's duration to [min_length_sec, max_length_sec] by finding
    the nearest valid sentence-end or scene-cut boundary.

    Returns (clamped_start, clamped_end, clamp_method) where clamp_method
    is one of: 'none', 'sentence_boundary', 'scene_boundary', 'raw_fallback'.
    """
    duration = end_sec - start_sec

    # --- OVER-LENGTH CLAMPING ---
    if duration > max_length_sec:
        hard_limit = start_sec + max_length_sec
        search_floor = hard_limit - tolerance_sec

        # Strategy 1: Find nearest Whisper segment end at or before hard_limit
        best_sentence_end = None
        for seg in transcript_segments:
            seg_end = seg.get("end", 0.0)
            if search_floor <= seg_end <= hard_limit:
                if best_sentence_end is None or seg_end > best_sentence_end:
                    best_sentence_end = seg_end

        if best_sentence_end is not None:
            return start_sec, round(best_sentence_end, 2), "sentence_boundary"

        # Strategy 2: Find nearest scene-cut end at or before hard_limit
        best_scene_end = None
        for scene in scenes:
            scene_end = scene.get("end_sec", 0.0)
            if search_floor <= scene_end <= hard_limit:
                if best_scene_end is None or scene_end > best_scene_end:
                    best_scene_end = scene_end

        if best_scene_end is not None:
            return start_sec, round(best_scene_end, 2), "scene_boundary"

        # Strategy 3: Raw fallback — log explicitly
        logger.warning(
            f"[DurationClamp] No sentence or scene boundary found within "
            f"{tolerance_sec}s of hard limit {hard_limit:.1f}s for clip "
            f"[{start_sec:.1f}s-{end_sec:.1f}s]. Using raw chop fallback."
        )
        return start_sec, round(hard_limit, 2), "raw_fallback"

    # --- UNDER-LENGTH CLAMPING ---
    if duration < min_length_sec:
        target_end = start_sec + min_length_sec
        # Guard: never extend past max_length_sec even when fixing under-length
        upper_cap = start_sec + max_length_sec

        # Find nearest sentence end at or after target_end (capped at upper_cap)
        best_sentence_end = None
        for seg in transcript_segments:
            seg_end = seg.get("end", 0.0)
            if target_end <= seg_end <= min(target_end + tolerance_sec, upper_cap):
                if best_sentence_end is None or seg_end < best_sentence_end:
                    best_sentence_end = seg_end

        if best_sentence_end is not None:
            return start_sec, round(best_sentence_end, 2), "sentence_boundary"

        # Scene-boundary extension (capped at upper_cap)
        best_scene_end = None
        for scene in scenes:
            scene_end = scene.get("end_sec", 0.0)
            if target_end <= scene_end <= min(target_end + tolerance_sec, upper_cap):
                if best_scene_end is None or scene_end < best_scene_end:
                    best_scene_end = scene_end

        if best_scene_end is not None:
            return start_sec, round(best_scene_end, 2), "scene_boundary"

        # Raw fallback — extend to exactly min_length_sec (capped at upper_cap)
        fallback_end = min(target_end, upper_cap)
        logger.warning(
            f"[DurationClamp] No boundary found to extend under-length clip "
            f"[{start_sec:.1f}s-{end_sec:.1f}s] to {min_length_sec}s. "
            f"Using raw extension fallback."
        )
        return start_sec, round(fallback_end, 2), "raw_fallback"

    # Duration is within bounds
    return start_sec, end_sec, "none"


def snap_to_scene_boundaries(
    start_sec: float,
    end_sec: float,
    scenes: List[Dict[str, Any]],
    tolerance_sec: float = 1.2,
) -> tuple[float, float]:
    """
    Snap candidate start and end times to the nearest scene cut boundary if within tolerance.
    Prevents visually jarring cuts a fraction of a second before or after a camera transition.
    """
    snapped_start = start_sec
    snapped_end = end_sec

    for scene in scenes:
        scene_start = scene.get("start_sec", 0.0)
        scene_end = scene.get("end_sec", 0.0)

        # Check if candidate start is close to a scene start
        if abs(start_sec - scene_start) <= tolerance_sec:
            snapped_start = scene_start

        # Check if candidate end is close to a scene end
        if abs(end_sec - scene_end) <= tolerance_sec:
            snapped_end = scene_end

    return round(snapped_start, 2), round(snapped_end, 2)


DANGLING_CONNECTORS = {
    # English
    "and", "but", "so", "or", "because", "then", "which", "that", "with", "to", "if", "when", "like", "as", "how",
    # Hindi / Hinglish
    "aur", "lekin", "kyunki", "ki", "toh", "ya", "yaani", "matlab", "agar", "jab", "par", "bhi", "se"
}

META_TALK_PHRASES = [
    "anything more you want me to add",
    "anything else you want to add",
    "anything else you want me to add",
    "does that make sense",
    "does that make any sense",
    "do you know what i mean",
    "you know what i mean",
    "that's about it",
    "that's all i have",
    "did i answer your question",
    "does that answer your question",
    "any other questions",
]


def is_sentence_complete(text: str) -> bool:
    """Checks whether text ends on a complete sentence boundary rather than a hanging connector."""
    clean = text.strip()
    if not clean or clean.endswith(("...", "…", ",", "–", "-", ";", ":")):
        return False
    words = clean.lower().rstrip(".,?!:;–-\"'").split()
    if words and words[-1] in DANGLING_CONNECTORS:
        return False
    return clean.endswith((".", "?", "!", "।", "\"", "”"))


def snap_to_sentence_boundaries(
    start_sec: float,
    end_sec: float,
    transcript_segments: List[Dict[str, Any]],
    tolerance_sec: float = 3.0,
    ensure_complete: bool = True,
    lead_in_pad: float = 0.15,
    tail_release_pad: float = 0.35,
) -> tuple[float, float]:
    """
    Snap candidate start and end times outwards to encompass the entire spoken segment,
    ensuring sentences never cut off on dangling connectors or mid-thought, and adding
    natural acoustic release padding.
    """
    if not transcript_segments:
        return round(start_sec, 2), round(end_sec, 2)

    snapped_start = start_sec
    snapped_end = end_sec
    start_idx = None
    end_idx = None

    # Find starting segment
    for idx, seg in enumerate(transcript_segments):
        s = seg.get("start", 0.0)
        e = seg.get("end", 0.0)
        if s <= start_sec < e or (idx == 0 and start_sec < s):
            snapped_start = s
            start_idx = idx
            break

    # If ensure_complete, walk backward if the starting segment is mid-sentence
    if ensure_complete and start_idx is not None:
        current_idx = start_idx
        while current_idx > 0:
            prev_seg_text = transcript_segments[current_idx - 1].get("text", "")
            # If the previous segment ends with a complete sentence boundary, we are at the start of a new one
            if is_sentence_complete(prev_seg_text):
                break
            current_idx -= 1
            snapped_start = transcript_segments[current_idx].get("start", snapped_start)
            start_idx = current_idx
            if (snapped_end - snapped_start) > 65.0:
                break

    # Apply lead-in padding without bleeding into previous speech
    if start_idx is not None:
        prev_end = transcript_segments[start_idx - 1].get("end", 0.0) if start_idx > 0 else 0.0
        snapped_start = max(prev_end, snapped_start - lead_in_pad)

    # Find ending segment
    for idx in range(len(transcript_segments) - 1, -1, -1):
        seg = transcript_segments[idx]
        s = seg.get("start", 0.0)
        e = seg.get("end", 0.0)
        if s < end_sec <= e or (idx == len(transcript_segments) - 1 and end_sec > e):
            snapped_end = e
            end_idx = idx
            break

    # If ensure_complete, advance forward if the ending segment does not finish a complete sentence
    if ensure_complete and end_idx is not None:
        current_idx = end_idx
        while current_idx < len(transcript_segments) - 1:
            seg_text = transcript_segments[current_idx].get("text", "")
            if is_sentence_complete(seg_text):
                snapped_end = transcript_segments[current_idx].get("end", snapped_end)
                end_idx = current_idx
                break
            current_idx += 1
            snapped_end = transcript_segments[current_idx].get("end", snapped_end)
            end_idx = current_idx
            if (snapped_end - snapped_start) > 65.0:
                break

    # Strip conversational filler / meta-talk from the end
    import re
    if end_idx is not None:
        while end_idx >= (start_idx or 0):
            seg_text = transcript_segments[end_idx].get("text", "").strip().lower()
            clean_text = re.sub(r'[^\w\s]', '', seg_text)
            
            # Check if this segment contains any meta-talk phrases
            if any(phrase in clean_text for phrase in META_TALK_PHRASES):
                logger.info(f"[CandidateRanker] Stripping meta-talk from clip end: '{seg_text}'")
                end_idx -= 1
                if end_idx >= 0:
                    snapped_end = transcript_segments[end_idx].get("end", snapped_end)
                else:
                    break
            else:
                break

    # Apply natural room-tone / vocal release padding so audio doesn't hit a digital scissor cut
    if end_idx is not None:
        next_start = (
            transcript_segments[end_idx + 1].get("start", snapped_end + 1.0)
            if end_idx < len(transcript_segments) - 1
            else snapped_end + tail_release_pad
        )
        snapped_end = min(snapped_end + tail_release_pad, next_start - 0.05)

    return round(snapped_start, 2), round(snapped_end, 2)


def deduplicate_and_rank_candidates(
    candidates: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]] | None = None,
    max_overlap_ratio: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Deduplicates candidates that overlap significantly and sorts them by composite rank:
    composite_rank = (editorial_potential * 0.5) + ((transformation_score / 100.0) * 0.5)
    """
    if not candidates:
        return []

    scenes = scenes or []

    # 1. Snap to scene boundaries if available
    processed = []
    for cand in candidates:
        c = dict(cand)
        start = float(c.get("start_sec", 0.0))
        end = float(c.get("end_sec", 0.0))

        if scenes:
            snapped_s, snapped_e = snap_to_scene_boundaries(start, end, scenes)
            c["start_sec"] = snapped_s
            c["end_sec"] = snapped_e

        # AUDIT-P1-05: Product policy prohibits virality/monetization prediction.
        # Primary candidate editorial score is editorial_potential, with graceful fallback to legacy virality_score.
        editorial_score = float(
            c.get("editorial_potential", c.get("virality_score", c.get("score", 0.5)))
        )
        transformation = float(c.get("transformation_score", 50))
        composite = (editorial_score * 0.5) + ((transformation / 100.0) * 0.5)
        c["composite_rank"] = round(composite, 3)
        processed.append(c)

    # 2. Sort by composite rank descending
    processed.sort(key=lambda x: x["composite_rank"], reverse=True)

    # 3. Deduplicate overlapping clips
    kept: List[Dict[str, Any]] = []
    for cand in processed:
        start_a = cand["start_sec"]
        end_a = cand["end_sec"]
        dur_a = max(0.1, end_a - start_a)

        overlaps = False
        for existing in kept:
            start_b = existing["start_sec"]
            end_b = existing["end_sec"]

            # Calculate intersection
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            overlap_dur = max(0.0, overlap_end - overlap_start)

            if overlap_dur / dur_a > max_overlap_ratio:
                overlaps = True
                break

        if not overlaps:
            kept.append(cand)

    return kept

"""
ClipForge AI — Brief-Aware Candidate Selection Worker (v2)

Pipeline stage 3 (LLM Queue):
- Uses OpenAI-compatible LLM Gateway (OmniRoute/FreeLLMAPI)
- Integrates transcript, scene boundaries, editorial template, rights basis, and campaign brief
- Computes 0–100 Transformation Score and breakdown per Section 2.4
- Snaps candidates to scene cut boundaries and deduplicates overlapping excerpts
- Persists selections.json and populates Clip database records with transformation scores
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from clipforge_core.celery_app import celery_app
from clipforge_core.config import settings
from clipforge_core.database import get_sync_session
from clipforge_core.models import Clip, Job, Project, ProjectAuditEvent
from clipforge_core.services.candidate_ranker import (
    clamp_to_boundary,
    deduplicate_and_rank_candidates,
    snap_to_sentence_boundaries,
    snap_to_scene_boundaries,
)
from clipforge_core.services.llm_client import LLMClientError, llm_client
from clipforge_core.services.temporal_binner import compute_temporal_bins, format_bin_directives, validate_bin_membership
from clipforge_core.services.transformation_scorer import calculate_transformation_score

logger = logging.getLogger(__name__)

SELECTION_SYSTEM_PROMPT = """You are an expert video editor and transformation strategist for short-form social video (YouTube Shorts, TikTok, Instagram Reels).

Your task is to analyze a source video transcript, scene boundaries, and project brief to identify the highest-potential, transformation-ready clipping candidates.

EDITORIAL PRINCIPLES:
1. Self-contained Narrative: Each clip must have a strong hook, clear body point/evidence, and a satisfying conclusion or punchline.
2. High Transformation Potential: Favor moments where original commentary, callouts, and explanatory context add significant value.
3. Natural Speech Boundaries: Start and end at natural pause points.
4. Brief Alignment: Strictly adhere to the tone, required mentions, and banned topics in the campaign brief.
5. NO FILLERS OR DEAD AIR (CRITICAL): Absolutely DO NOT select segments where the speaker says "uh", "um", stutters, repeats words, or has long awkward pauses.
6. NO META-TALK: Strictly exclude interviewer questions or meta-talk like "Does that make sense?".

You MUST respond with valid JSON matching the requested schema exactly. No markdown fences, no conversational text."""


# Content focus prompt directives
_CONTENT_FOCUS_DIRECTIVES = {
    "balanced": (
        "Select a balanced mix of clips: include both high-energy action/highlight "
        "moments AND conversational/insightful moments. Ensure diversity in the type of content selected."
    ),
    "contestant_primary": (
        "PRIORITIZE action, core events, and high-intensity highlights. "
        "Focus on the main subjects performing actions or delivering the core "
        "entertainment value of the video. At least 70% of clips should be action/highlight driven."
    ),
    "judges_primary": (
        "PRIORITIZE commentary, deep insights, explanations, and reactions. "
        "Focus on the speakers delivering thesis statements, educational takeaways, "
        "or deep discussions. At least 70% of clips should feature insights or commentary as the primary subject."
    ),
}


def _build_selection_prompt(
    transcript: dict,
    scenes: list,
    campaign_brief: dict,
    editorial_template: str,
    rights_basis: str,
    clip_count: int,
    min_length_sec: int,
    max_length_sec: int,
    custom_prompt: str | None = None,
    temporal_bins_directive: str = "",
    content_focus: str = "balanced",
) -> str:
    """Build structured LLM prompt with optional temporal and content focus directives."""
    segments = transcript.get("segments", [])
    formatted_segments = []
    current_asset = None
    prev_end = 0.0
    for seg in segments:
        asset_id = seg.get("asset_id", "primary")
        if asset_id != current_asset:
            formatted_segments.append(f"\n--- SOURCE VIDEO: {asset_id} ---")
            current_asset = asset_id
            prev_end = 0.0 # reset on new video
            
        import re
        s = seg.get("start", 0.0)
        e = seg.get("end", 0.0)
        t = seg.get("text", "").strip()
        
        if prev_end > 0 and (s - prev_end) >= 2.0:
            formatted_segments.append(f"[WARNING: DEAD AIR {s - prev_end:.1f}s]")
            
        prev_end = e
        
        # Simple heuristic for fillers (uh, um, hmm)
        t_clean = re.sub(r'[^\w\s]', '', t.lower())
        has_filler = bool(re.search(r'\b(uh|uhh|um|umm|hmm|hmmm)\b', t_clean))
        warning = " [WARNING: CONTAINS FILLER]" if has_filler else ""
        
        formatted_segments.append(f"[{s:.1f}s - {e:.1f}s] {t}{warning}")

    transcript_text = "\n".join(formatted_segments)
    total_duration = transcript.get("duration_sec", 0.0)

    # Build optional sections
    focus_directive = _CONTENT_FOCUS_DIRECTIVES.get(content_focus, _CONTENT_FOCUS_DIRECTIVES["balanced"])
    temporal_section = f"\n{temporal_bins_directive}\n" if temporal_bins_directive else ""

    prompt = f"""## SOURCE VIDEO INFORMATION
- Total Duration: {total_duration:.1f}s
- Language: {transcript.get('language', 'unknown')}
- Rights Basis: {rights_basis}
- Editorial Template: {editorial_template}

## TRANSCRIPT WITH TIMESTAMPS
{transcript_text}

## SCENE CUT BOUNDARIES (First 20)
{json.dumps(scenes[:20], indent=2)}

## CAMPAIGN BRIEF
{json.dumps(campaign_brief, indent=2)}

## CONTENT FOCUS
{focus_directive}
{temporal_section}
## USER GUIDANCE
{custom_prompt if custom_prompt else "Identify the most engaging standalone moments matching the campaign brief and editorial template."}

## TASK
Select up to {clip_count} highlight candidates.

CRITICAL MANDATORY RULES:
1. MULTI-SOURCE AWARENESS:
   The transcript may contain multiple source videos separated by "--- SOURCE VIDEO ---". 
   - You MUST extract at least one highly engaging clip from EACH source video if multiple exist.
   - Do NOT stitch scenes across different source videos unless they are perfectly related. A single clip should generally be self-contained within the same source video.

2. MULTI-SCENE NARRATIVE STITCHING (MANDATORY):
   Every clip MUST be composed of 1 to 3 distinct scenes (`segments`) stitched together to create a dynamic, viral story that does NOT lose context:
   - Scene 1 (The Hook): 5 to 10 seconds. The most attention-grabbing quote, controversy, question, or emotional reaction.
   - Scene 2 (The Context & Payoff): 15 to 30 seconds. The backstory or dialogue that explains the context and reaches the conclusion/punchline.
   (Optional Scene 3: 8 to 15 seconds if a 3rd scene provides the solution or resolution).
   The scenes do NOT have to be contiguous; stitching non-contiguous moments creates the highest retention viral clips!

2. CLIP DURATION CONSTRAINT:
   Combined duration across all scenes in each clip MUST be between {min_length_sec} and {max_length_sec} seconds (typically 25 to 50 seconds total).
   NEVER output a single 2 to 5 second fragment. Every clip must provide complete context so the audience understands the full story from beginning to end.

3. HINGLISH & AUDIO QUALITY:
   The video may be in Hinglish (mixed Hindi and English). Evaluate content quality across both languages. STRICTLY AVOID selecting any segments where the speaker stutters, gets stuck, repeats words, or where there is dead air.

4. COMPLETE THOUGHTS:
   Each segment MUST start at the beginning of a full sentence and end at the end of a complete sentence. DO NOT cut off mid-thought.

5. AVOID CONVERSATIONAL FILLER, INTERVIEWERS, & META-TALK (MANDATORY):
   - Strictly exclude end-of-clip conversational filler or vague wrap-ups (e.g., "Anything more you want me to add?", "Does that make sense?").
   - STRICTLY exclude interviewer questions and other people interrupting. The clip MUST ONLY feature the primary speaker delivering their point seamlessly. Adjust timestamps to trim out anyone else's voice.

6. Hook Type must be one of: "question", "bold_statement", "surprising_stat", "story_loop", "controversial_thesis".
7. Score `editorial_potential` realistically from 0.0 to 1.0 based on hook strength and virality.
8. Keep reasoning brief (1 short sentence) and suggested_callouts concise.
9. Output JSON directly matching the schema below without any conversational preamble or thinking text.

## REQUIRED JSON FORMAT
Return a JSON object:
{{
  "clips": [
    {{
      "title": "Short punchy title",
      "hook_type": "bold_statement",
      "hook_text": "Exact hook quote from Scene 1",
      "key_takeaway": "What viewer learns by the end",
      "editorial_potential": 0.95,
      "reasoning": "Starts with the explosive claim at 45s, then stitches the context and resolution from 56s.",
      "suggested_callouts": ["Term 1", "Statistic 2"],
      "segments": [
        {{ "scene_role": "hook", "start_sec": 48.9, "end_sec": 54.9 }},
        {{ "scene_role": "context_and_payoff", "start_sec": 56.2, "end_sec": 75.8 }}
      ]
    }}
  ]
}}"""
    return prompt


from clipforge_core.services.progress import update_job_progress


def _update_project_status(project_id: str, status: str) -> None:
    """Update project status."""
    session = get_sync_session()
    try:
        project = session.query(Project).filter(Project.id == uuid.UUID(project_id)).first()
        if project:
            project.status = status
            session.commit()
    except Exception as e:
        logger.error(f"Failed to update project status: {e}")
        session.rollback()
    finally:
        session.close()


@celery_app.task(
    name="clipforge_core.workers.select.select_clips",
    queue="llm",
    bind=True,
    max_retries=3,
    default_retry_delay=15,
)
def select_clips(
    self,
    project_id: str,
    clip_count: int = 5,
    min_length_sec: int = 20,
    max_length_sec: int = 60,
    custom_prompt: str | None = None,
    time_range_start: float | None = None,
    time_range_end: float | None = None,
    temporal_distribution: str = "even_spread",
    content_focus: str = "balanced",
) -> Dict[str, Any]:
    """
    Candidate selection task running on the 'llm' queue.
    """
    logger.info(f"[LLM Select] Starting candidate selection for project {project_id}")
    session_check = get_sync_session()
    try:
        proj_check = session_check.query(Project).filter(Project.id == uuid.UUID(project_id)).first()
        if not proj_check:
            logger.warning(f"[LLM Select] Project {project_id} not found in database. Aborting orphaned select task.")
            return {"error": "Project not found"}
    finally:
        session_check.close()

    update_job_progress(project_id, stage="select", status="running", percent=10.0, detail="Parsing transcript & brief...")
    _update_project_status(project_id, "selecting")

    project_dir = Path(settings.MEDIA_DIR) / project_id
    analysis_files = list(project_dir.glob("analysis*.json"))
    transcript_files = list(project_dir.glob("transcript*.json"))

    if not analysis_files and not transcript_files:
        error_msg = f"Transcripts missing for project {project_id}"
        update_job_progress(project_id, stage="select", status="failed", error_message=error_msg, force_write=True)
        _update_project_status(project_id, "failed")
        raise FileNotFoundError(error_msg)

    # Load and merge multiple transcripts/scenes into a single mega-timeline
    merged_transcript = {"segments": [], "duration_sec": 0.0, "language": "unknown"}
    merged_scenes = []
    source_mapping = []

    current_offset = 0.0

    def get_asset_id_from_path(p: Path) -> str:
        name = p.stem
        if "_" in name:
            return name.split("_", 1)[1]
        return "primary"

    files_to_process = analysis_files if analysis_files else transcript_files
    for f in sorted(files_to_process):
        asset_id = get_asset_id_from_path(f)
        data = json.loads(f.read_text(encoding="utf-8"))
        
        if f.name.startswith("analysis"):
            ts_data = data.get("transcript", {})
            sc_data = data.get("scenes", [])
        else:
            ts_data = data
            sc_data = []

        dur = ts_data.get("duration_sec", 0.0)
        source_mapping.append({
            "asset_id": asset_id,
            "offset_start": current_offset,
            "offset_end": current_offset + dur,
        })
        
        merged_transcript["language"] = ts_data.get("language", merged_transcript["language"])

        for seg in ts_data.get("segments", []):
            new_seg = seg.copy()
            new_seg["start"] = new_seg.get("start", 0.0) + current_offset
            new_seg["end"] = new_seg.get("end", 0.0) + current_offset
            new_seg["asset_id"] = asset_id
            merged_transcript["segments"].append(new_seg)

        for scene in sc_data:
            new_scene = scene.copy()
            new_scene["start_sec"] = new_scene.get("start_sec", 0.0) + current_offset
            new_scene["end_sec"] = new_scene.get("end_sec", 0.0) + current_offset
            new_scene["asset_id"] = asset_id
            merged_scenes.append(new_scene)

        current_offset += dur

    merged_transcript["duration_sec"] = current_offset
    transcript = merged_transcript
    scenes = merged_scenes

    # Fetch Project & Brief from DB
    session = get_sync_session()
    try:
        project = session.query(Project).filter(Project.id == uuid.UUID(project_id)).first()
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        editorial_template = project.editorial_template or "explainer"
        rights_basis = project.rights_basis or "owned"
        campaign_brief = project.campaign_brief.brief_json if project.campaign_brief else {}
        if time_range_start is None and project.time_range_start is not None:
            time_range_start = project.time_range_start
        if time_range_end is None and project.time_range_end is not None:
            time_range_end = project.time_range_end
        if temporal_distribution == "even_spread" and getattr(project, "temporal_distribution", None):
            temporal_distribution = project.temporal_distribution
        if content_focus == "balanced" and getattr(project, "content_focus", None):
            content_focus = project.content_focus
    finally:
        session.close()

    # Filter transcript by time range if requested
    if time_range_start is not None or time_range_end is not None:
        t_start = time_range_start or 0.0
        t_end = time_range_end or float("inf")
        filtered_segs = [
            s for s in transcript.get("segments", [])
            if s.get("start", 0.0) >= t_start and s.get("end", 0.0) <= t_end
        ]
        transcript["segments"] = filtered_segs

    # Compute temporal bins for even spread
    temporal_bins_directive = ""
    temporal_bins = []
    total_source_dur_for_bins = transcript.get("duration_sec", 60.0)
    if temporal_distribution == "even_spread" and clip_count > 1:
        effective_start = time_range_start or 0.0
        effective_end = time_range_end or total_source_dur_for_bins
        temporal_bins = compute_temporal_bins(
            total_duration_sec=total_source_dur_for_bins,
            clip_count=clip_count,
            time_range_start=effective_start,
            time_range_end=effective_end,
        )
        temporal_bins_directive = format_bin_directives(temporal_bins)

    # Build prompt
    prompt = _build_selection_prompt(
        transcript=transcript,
        scenes=scenes,
        campaign_brief=campaign_brief,
        editorial_template=editorial_template,
        rights_basis=rights_basis,
        clip_count=clip_count,
        min_length_sec=min_length_sec,
        max_length_sec=max_length_sec,
        custom_prompt=custom_prompt,
        temporal_bins_directive=temporal_bins_directive,
        content_focus=content_focus,
    )

    update_job_progress(project_id, stage="select", percent=30.0, detail="Awaiting LLM extraction (this takes a moment)...")

    try:
        # Run async LLM completion in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_json = loop.run_until_complete(
                llm_client.complete_json(
                    prompt=prompt,
                    system=SELECTION_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )
        finally:
            loop.close()

        update_job_progress(project_id, stage="select", percent=80.0, detail="Ranking candidates & deduplicating...")

        raw_clips: List[Dict[str, Any]] = (
            response_json.get("clips", []) if isinstance(response_json, dict) else response_json
        )

        total_source_dur = transcript.get("duration_sec", 60.0)

        # Compute transformation score and enrich candidates
        transcript_segments = transcript.get("segments", [])

        enriched_candidates = []
        clamp_stats = {"none": 0, "sentence_boundary": 0, "scene_boundary": 0, "raw_fallback": 0}
        for raw in raw_clips:
            segments = raw.get("segments", [])
            if not segments:
                s_val = float(raw.get("start_sec", 0.0))
                e_val = float(raw.get("end_sec", s_val + min_length_sec))
                segments = [{"start_sec": s_val, "end_sec": e_val}]
            
            # --- MULTI-SCENE NARRATIVE EXPANSION ---
            # If the candidate only has 1 segment (e.g. LLM selected a single quote),
            # synthesize a 2-scene narrative (Hook + Context & Payoff) from the surrounding dialogue.
            total_raw_dur = sum(float(s.get("end_sec", 0.0)) - float(s.get("start_sec", 0.0)) for s in segments)
            if len(segments) == 1 and total_raw_dur < min_length_sec:
                hook_s = float(segments[0].get("start_sec", 0.0))
                hook_e = float(segments[0].get("end_sec", hook_s + 6.0))
                hook_dur = max(4.0, min(10.0, hook_e - hook_s))
                hook_e = hook_s + hook_dur
                
                # Locate hook end in transcript
                hook_end_idx = 0
                for i, ts in enumerate(transcript_segments):
                    if ts.get("end", 0.0) >= hook_e:
                        hook_end_idx = i
                        break
                
                # Build context scene starting at the segment after the hook
                needed_context_dur = max(float(min_length_sec) - hook_dur, 16.0)
                if hook_end_idx + 1 < len(transcript_segments):
                    context_start = transcript_segments[hook_end_idx + 1].get("start", hook_e + 0.5)
                else:
                    context_start = hook_e + 0.5
                context_end = context_start + needed_context_dur
                
                segments = [
                    {"scene_role": "hook", "start_sec": hook_s, "end_sec": hook_e},
                    {"scene_role": "context_and_payoff", "start_sec": context_start, "end_sec": context_end},
                ]

            processed_segments = []
            total_duration = 0.0
            overall_start = None
            overall_end = None
            clamp_method = "sentence_boundary"

            for seg in segments:
                start_s = float(seg.get("start_sec", 0.0))
                end_s = float(seg.get("end_sec", start_s + 6.0))
                
                # Determine which asset this segment belongs to based on start_s
                asset_start_limit = 0.0
                asset_end_limit = total_source_dur
                for m in source_mapping:
                    if m["offset_start"] <= start_s <= m["offset_end"]:
                        asset_start_limit = m["offset_start"]
                        asset_end_limit = m["offset_end"]
                        break
                
                # Initial clamp before snapping
                start_s = max(asset_start_limit, start_s)
                end_s = min(asset_end_limit, max(start_s + 1.0, end_s))

                # Snap LLM timestamps to exact sentence boundaries with lead-in and release padding
                start_s, end_s = snap_to_sentence_boundaries(
                    start_sec=start_s,
                    end_sec=end_s,
                    transcript_segments=transcript_segments,
                    ensure_complete=True,
                    lead_in_pad=0.15,
                    tail_release_pad=0.35,
                )
                
                # FINAL CLAMP: prevent snap_to_sentence_boundaries from bleeding into adjacent videos
                start_s = max(asset_start_limit, start_s)
                end_s = min(asset_end_limit, end_s)

                # Snap to visual scene cuts if nearby
                if scenes:
                    start_s, end_s = snap_to_scene_boundaries(start_s, end_s, scenes, tolerance_sec=1.2)

                # Ensure individual segment has minimum sensible duration of at least 3.5s
                if (end_s - start_s) < 3.5:
                    end_s = start_s + 3.5

                processed_segments.append({
                    "start_sec": round(start_s, 2),
                    "end_sec": round(end_s, 2),
                    "scene_role": seg.get("scene_role", "scene"),
                })

            # Merge overlapping segments within the clip to prevent repeated video parts
            if processed_segments:
                processed_segments.sort(key=lambda x: x["start_sec"])
                merged = [processed_segments[0]]
                for current in processed_segments[1:]:
                    last = merged[-1]
                    if current["start_sec"] <= last["end_sec"]:
                        # Overlap or contiguous, merge them
                        last["end_sec"] = max(last["end_sec"], current["end_sec"])
                        last["scene_role"] = f"{last['scene_role']}_and_{current['scene_role']}"
                    else:
                        merged.append(current)
                processed_segments = merged
                
            total_duration = sum(s["end_sec"] - s["start_sec"] for s in processed_segments)
            if processed_segments:
                overall_start = processed_segments[0]["start_sec"]
                overall_end = max(s["end_sec"] for s in processed_segments)
            else:
                overall_start = 0.0
                overall_end = 0.0

            # Ensure total clip duration satisfies min_length_sec
            if total_duration < min_length_sec and processed_segments:
                deficit = float(min_length_sec) - total_duration
                last_seg = processed_segments[-1]
                target_end = last_seg["end_sec"] + deficit
                
                # Prevent crossing video boundaries
                max_allowed_end = target_end
                for m in source_mapping:
                    if m["offset_start"] <= last_seg["start_sec"] <= m["offset_end"]:
                        max_allowed_end = min(target_end, m["offset_end"])
                        break
                target_end = max_allowed_end

                _, snapped_ext = snap_to_sentence_boundaries(
                    start_sec=last_seg["end_sec"],
                    end_sec=target_end,
                    transcript_segments=transcript_segments,
                    ensure_complete=True,
                    lead_in_pad=0.0,
                    tail_release_pad=0.35,
                )
                last_seg["end_sec"] = max(round(target_end, 2), snapped_ext)
                # Final safety clamp
                last_seg["end_sec"] = min(last_seg["end_sec"], max_allowed_end)
                
                total_duration = sum(s["end_sec"] - s["start_sec"] for s in processed_segments)
                overall_end = max(overall_end or 0.0, last_seg["end_sec"])

            # Ensure total clip duration does not exceed max_length_sec
            if total_duration > max_length_sec and processed_segments:
                excess = total_duration - float(max_length_sec)
                longest_seg = max(processed_segments, key=lambda s: s["end_sec"] - s["start_sec"])
                longest_seg["end_sec"] = round(max(longest_seg["start_sec"] + 5.0, longest_seg["end_sec"] - excess), 2)
                total_duration = sum(s["end_sec"] - s["start_sec"] for s in processed_segments)
                overall_end = max(s["end_sec"] for s in processed_segments)

            clamp_stats[clamp_method] = clamp_stats.get(clamp_method, 0) + 1
            duration = total_duration
            start_s = overall_start if overall_start is not None else 0.0
            end_s = overall_end if overall_end is not None else start_s + min_length_sec

            t_score_data = calculate_transformation_score(
                clip_duration_sec=duration,
                total_source_duration_sec=total_source_dur,
                has_commentary=True,
                editorial_template=editorial_template,
                callout_count=len(raw.get("suggested_callouts", [])),
                has_hook=bool(raw.get("hook_text")),
                has_takeaway=bool(raw.get("key_takeaway")),
            )

            editorial_pot = round(
                float(raw.get("editorial_potential", raw.get("virality_score", raw.get("score", 0.75)))), 2
            )
            cand = {
                "start_sec": round(start_s, 2),
                "end_sec": round(end_s, 2),
                "segments": processed_segments,
                "raw_start_sec": round(float(raw.get("start_sec", 0.0)), 2),
                "raw_end_sec": round(float(raw.get("end_sec", 0.0)), 2),
                "raw_duration_sec": round(float(raw.get("end_sec", 0.0)) - float(raw.get("start_sec", 0.0)), 2),
                "title": raw.get("title", f"Clip @ {int(start_s)}s"),
                "hook_type": raw.get("hook_type", "bold_statement"),
                "hook_text": raw.get("hook_text", ""),
                "key_takeaway": raw.get("key_takeaway", ""),
                "editorial_potential": editorial_pot,
                "virality_score": editorial_pot,  # Backward compatibility
                "transformation_score": t_score_data["score"],
                "transformation_breakdown": t_score_data["breakdown"],
                "transformation_band": t_score_data["band"],
                "reasoning": raw.get("reasoning", "Strong highlight candidate matching editorial template"),
                "suggested_callouts": raw.get("suggested_callouts", []),
                "clamp_method": clamp_method,
            }
            enriched_candidates.append(cand)

        logger.info(f"[LLM Select] Clamp stats: {clamp_stats}")

        # Bin-membership validation for even_spread mode
        if temporal_bins and temporal_distribution == "even_spread":
            enriched_candidates, bin_violations = validate_bin_membership(
                enriched_candidates, temporal_bins
            )
            if bin_violations:
                logger.warning(
                    f"[LLM Select] {len(bin_violations)} candidates discarded "
                    f"due to bin-membership violations"
                )

        # Snap to scene cut boundaries and deduplicate
        final_clips = deduplicate_and_rank_candidates(enriched_candidates, scenes=scenes)

        # Unmap timestamps back to their native asset_ids
        def unmap_ts(ts: float) -> tuple[str, float]:
            for m in source_mapping:
                if m["offset_start"] <= ts <= m["offset_end"]:
                    return m["asset_id"], round(ts - m["offset_start"], 2)
            if source_mapping:
                m = source_mapping[-1]
                return m["asset_id"], round(ts - m["offset_start"], 2)
            return "primary", round(ts, 2)

        for clip in final_clips:
            # Map top-level timestamps (using the first segment's asset_id for the clip's primary reference)
            primary_asset, mapped_start = unmap_ts(clip["start_sec"])
            _, mapped_end = unmap_ts(clip["end_sec"])
            clip["start_sec"] = mapped_start
            clip["end_sec"] = mapped_end
            clip["asset_id"] = primary_asset
            
            for seg in clip.get("segments", []):
                seg_asset, s_mapped = unmap_ts(seg["start_sec"])
                _, e_mapped = unmap_ts(seg["end_sec"])
                seg["start_sec"] = s_mapped
                seg["end_sec"] = e_mapped
                seg["asset_id"] = seg_asset

        # Limit to requested clip count
        final_clips = final_clips[:clip_count]

        # Persist Clip records in DB
        db_session = get_sync_session()
        try:
            pid = uuid.UUID(project_id)
            existing_clips = db_session.query(Clip).filter(Clip.project_id == pid).all()
            
            for c in final_clips:
                matched = False
                for ex in existing_clips:
                    if abs(float(ex.start_sec) - c["start_sec"]) < 0.2 and abs(float(ex.end_sec) - c["end_sec"]) < 0.2:
                        matched = True
                        c["clip_id"] = str(ex.id)
                        break
                        
                if not matched:
                    clip_record = Clip(
                        id=uuid.uuid4(),
                        project_id=pid,
                        start_sec=c["start_sec"],
                        end_sec=c["end_sec"],
                        score=c.get("editorial_potential", c.get("virality_score", 0.75)),
                        transformation_score=c["transformation_score"],
                        transformation_breakdown=c["transformation_breakdown"],
                        reasoning=f"[{c['hook_type']}] {c['title']} — {c['reasoning']}",
                        review_status="pending",
                    )
                    db_session.add(clip_record)
                    db_session.flush() # flush to get the id if needed, though we already generated it
                    c["clip_id"] = str(clip_record.id)
                    existing_clips.append(clip_record) # add to existing so we don't duplicate within the same batch

            # Record audit event
            audit = ProjectAuditEvent(
                id=uuid.uuid4(),
                project_id=pid,
                event_type="candidates_selected",
                payload={
                    "candidate_count": len(final_clips),
                    "avg_transformation_score": round(
                        sum(c["transformation_score"] for c in final_clips) / len(final_clips), 1
                    ) if final_clips else 0,
                },
            )
            db_session.add(audit)
            db_session.commit()
        except Exception as e:
            logger.error(f"Failed to record clips in DB: {e}")
            db_session.rollback()
        finally:
            db_session.close()

        # Save selections.json to disk (Now contains clip_id)
        selections_path = project_dir / "selections.json"
        selections_payload = {
            "project_id": project_id,
            "clips": final_clips,
            "total_selected": len(final_clips),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        selections_path.write_text(json.dumps(selections_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Archive to MinIO
        try:
            from clipforge_core.services.storage import default_storage
            default_storage.save_file(selections_path, f"{project_id}/selections.json")
        except Exception as e:
            logger.error(f"[LLM Select] Failed to archive selections to MinIO: {e}")

        update_job_progress(project_id, stage="select", status="success", percent=100.0, detail="Selection complete.", force_write=True)
        logger.info(f"[LLM Select] Selected {len(final_clips)} clips for project {project_id}")
        return selections_payload

    except LLMClientError as e:
        error_msg = f"LLM Gateway error: {e.message}"
        logger.error(f"[LLM Select] {error_msg}")
        update_job_progress(project_id, stage="select", status="failed", error_message=error_msg, force_write=True)
        _update_project_status(project_id, "failed")
        raise

    except Exception as e:
        error_msg = f"Candidate selection error: {e}"
        logger.error(f"[LLM Select] {error_msg}")
        if self.request.retries < self.max_retries:
            update_job_progress(project_id, stage="select", status="retrying", error_message=error_msg, force_write=True)
            raise self.retry(exc=e)
        else:
            update_job_progress(project_id, stage="select", status="failed", error_message=error_msg, force_write=True)
            _update_project_status(project_id, "failed")
            raise

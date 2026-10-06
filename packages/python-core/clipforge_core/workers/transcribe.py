"""
ClipForge AI — Transcribe Worker

Pipeline stage 2: Transcribe source video using faster-whisper.

Per TRD section 2 and user requirements:
- Uses faster-whisper, CPU-mode by default
- Config flag to switch to GPU later (WHISPER_DEVICE env var)
- Outputs timestamped transcript as JSON (word-level + segment-level)
- Writes granular status to jobs table (transcribe stage)

Output: {MEDIA_DIR}/{project_id}/transcript.json

Celery queue: transcribe (concurrency=2 in production, CPU-bound)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlx_whisper

from clipforge_core.celery_app import celery_app
from clipforge_core.config import settings
from clipforge_core.database import get_sync_session
from clipforge_core.models import Job, Project

logger = logging.getLogger(__name__)

def _update_job_status(
    project_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update the transcribe job status in the database."""
    session = get_sync_session()
    try:
        job = (
            session.query(Job)
            .filter(
                Job.project_id == uuid.UUID(project_id),
                Job.stage == "transcribe",
            )
            .first()
        )

        if job:
            job.status = status
            job.error_message = error_message
            if status == "running":
                job.started_at = datetime.now(timezone.utc)
            if status in ("success", "failed"):
                job.completed_at = datetime.now(timezone.utc)
            session.commit()
        else:
            logger.warning(f"No transcribe job found for project {project_id}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
        session.rollback()
    finally:
        session.close()


def _update_project_status(project_id: str, status: str) -> None:
    """Update the project-level status."""
    session = get_sync_session()
    try:
        project = (
            session.query(Project)
            .filter(
                Project.id == uuid.UUID(project_id),
            )
            .first()
        )
        if project:
            project.status = status
            session.commit()
    except Exception as e:
        logger.error(f"Failed to update project status: {e}")
        session.rollback()
    finally:
        session.close()


def transcribe_audio(source_path: str, output_dir: str, project_id: str | None = None, output_name: str = "transcript.json") -> dict:
    """
    Transcribe a video/audio file using faster-whisper.

    Args:
        source_path: Path to the source video file
        output_dir: Directory to write transcript
        output_name: Filename to write transcript to

    Returns:
        dict with:
            - segments: list of {start, end, text, words}
            - full_text: concatenated transcript
            - language: detected language
            - duration_sec: total audio duration
    """
    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    model_name = f"mlx-community/whisper-{settings.WHISPER_MODEL_SIZE}-mlx"
    # Special cases for MLX repo naming
    if settings.WHISPER_MODEL_SIZE == "large-v3":
        model_name = "mlx-community/whisper-large-v3-mlx"
    elif settings.WHISPER_MODEL_SIZE == "base":
        model_name = "mlx-community/whisper-base-mlx"

    logger.info(f"Transcribing: {source.name} using MLX ({model_name})")

    if project_id:
        try:
            from clipforge_core.services.progress import update_job_progress
            update_job_progress(
                project_id=project_id,
                stage="analysis",
                percent=30.0,
                detail=f"Transcribing via Apple Silicon MLX GPU ({model_name})...",
                force_write=True,
            )
        except Exception:
            pass

    # Run transcription (blocking, extremely fast on Mac GPU)
    result = mlx_whisper.transcribe(
        str(source),
        path_or_hf_repo=model_name,
        word_timestamps=True,
        initial_prompt="This is a Hinglish video with mixed Hindi and English speech.",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    language = result.get("language", "unknown")
    logger.info(f"Detected language: {language}")

    # Process segments into structured format
    segments = []
    full_text_parts = []
    
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue

        words = []
        for word in segment.get("words", []):
            word_text = word.get("word", "").strip()
            if word_text:
                words.append(
                    {
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                        "word": word_text,
                        "probability": round(word.get("probability", 1.0), 3),
                    }
                )

        seg_data = {
            "id": segment.get("id", len(segments)),
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "text": text,
            "words": words,
        }
        segments.append(seg_data)
        full_text_parts.append(text)

    if project_id:
        try:
            from clipforge_core.services.progress import update_job_progress
            update_job_progress(
                project_id=project_id,
                stage="analysis",
                percent=60.0,
                detail=f"Transcription complete: {len(segments)} segments.",
                force_write=True,
            )
        except Exception:
            pass

    full_text = " ".join(full_text_parts)
    
    # Infer duration from the last segment since MLX doesn't return info object
    duration_sec = 0.0
    if segments:
        duration_sec = segments[-1]["end"]

    transcript = {
        "language": language,
        "language_probability": 1.0,
        "duration_sec": duration_sec,
        "segment_count": len(segments),
        "segments": segments,
        "full_text": full_text,
    }

    # Write transcript to disk
    output_path = Path(output_dir) / output_name
    output_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        f"Transcription complete: {len(segments)} segments, {len(full_text)} chars, {duration_sec:.1f}s duration"
    )

    return transcript


@celery_app.task(
    name="app.workers.transcribe.transcribe_source",
    queue="analysis",
    bind=True,
    max_retries=1,
    default_retry_delay=15,
)
def transcribe_source(self, project_id: str, source_path: str) -> dict:
    """
    Transcribe the source video for a project.

    Args:
        project_id: UUID of the project
        source_path: Path to the downloaded source video

    Returns:
        dict with transcript_path and summary stats

    Updates jobs table with granular status:
        pending -> running -> success/failed
    """
    logger.info(f"[Transcribe] Starting for project {project_id}")

    _update_job_status(project_id, "running")
    _update_project_status(project_id, "transcribing")

    output_dir = str(Path(source_path).parent)

    try:
        transcript = transcribe_audio(source_path, output_dir)

        transcript_path = str(Path(output_dir) / "transcript.json")

        result = {
            "project_id": project_id,
            "transcript_path": transcript_path,
            "language": transcript["language"],
            "duration_sec": transcript["duration_sec"],
            "segment_count": transcript["segment_count"],
            "text_length": len(transcript["full_text"]),
        }

        # Archive to MinIO
        try:
            from clipforge_core.services.storage import default_storage
            default_storage.save_file(Path(transcript_path), f"{project_id}/transcript.json")
        except Exception as e:
            logger.error(f"[Transcribe] Failed to archive transcript to MinIO: {e}")

        _update_job_status(project_id, "success")
        logger.info(
            f"[Transcribe] Complete for project {project_id}: "
            f"{result['segment_count']} segments, {result['duration_sec']}s"
        )

        return result

    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(f"[Transcribe] Failed for project {project_id}: {error_msg}")
        _update_job_status(project_id, "failed", error_message=error_msg)
        _update_project_status(project_id, "failed")
        raise

    except Exception as e:
        error_msg = f"Transcription error: {e}"
        logger.error(f"[Transcribe] Error for project {project_id}: {error_msg}")

        if self.request.retries < self.max_retries:
            _update_job_status(project_id, "retrying", error_message=error_msg)
            raise self.retry(exc=e)
        else:
            _update_job_status(project_id, "failed", error_message=error_msg)
            _update_project_status(project_id, "failed")
            raise

"""Gemini-powered dataset assessment for portrait matting fine-tuning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re

import numpy as np
from google import genai
from google.genai import types
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ASSESSMENT_JSONL = "gemini_assessment.jsonl"
ASSESSMENT_SUMMARY = "gemini_assessment_summary.json"


@dataclass
class GeminiAssessmentConfig:
    api_key: str
    model: str
    timeout_seconds: int = 45
    max_output_tokens: int = 512


def _find_alpha_for_image(img_path: Path, images_dir: Path, alphas_dir: Path) -> Path | None:
    rel = img_path.relative_to(images_dir)
    for ext in (".png", ".jpg", ".jpeg"):
        candidates = (
            alphas_dir / rel.with_suffix(ext),
            alphas_dir / (img_path.stem + ext),
            alphas_dir / rel.parent.name / (img_path.stem + ext),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _image_to_jpeg_bytes(path: Path, max_side: int = 1024) -> bytes:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _alpha_coverage(alpha_path: Path | None) -> float | None:
    if alpha_path is None:
        return None
    alpha = Image.open(alpha_path)
    if alpha.mode == "RGBA":
        alpha = alpha.split()[-1]
    elif alpha.mode != "L":
        alpha = alpha.convert("L")
    arr = np.array(alpha).astype(np.float32) / 255.0
    return float(arr.mean())


def _normalize_issues(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    issues: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip().lower().replace(" ", "_")
        if value:
            issues.append(value)
    return sorted(set(issues))


def _normalize_assessment(raw: dict) -> dict:
    quality_score = raw.get("quality_score", 50)
    try:
        quality_score = int(quality_score)
    except Exception:
        quality_score = 50
    quality_score = max(0, min(100, quality_score))

    difficulty = str(raw.get("difficulty", "medium")).lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    split = str(raw.get("recommended_split", "train")).lower()
    if split not in {"train", "val", "exclude"}:
        split = "train"

    confidence = raw.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(raw.get("rationale", "")).strip()
    if len(rationale) > 220:
        rationale = rationale[:220]

    return {
        "quality_score": quality_score,
        "difficulty": difficulty,
        "recommended_split": split,
        "issues": _normalize_issues(raw.get("issues")),
        "confidence": confidence,
        "rationale": rationale,
    }


def _fallback_assessment(width: int, height: int, alpha_cov: float | None) -> dict:
    issues: list[str] = []
    if min(width, height) < 256:
        issues.append("low_resolution")
    if alpha_cov is not None:
        if alpha_cov < 0.01:
            issues.append("empty_alpha")
        elif alpha_cov > 0.99:
            issues.append("full_alpha")

    score = 80
    if "low_resolution" in issues:
        score -= 25
    if "empty_alpha" in issues or "full_alpha" in issues:
        score -= 30

    split = "train"
    if score < 40:
        split = "exclude"
    elif score < 60:
        split = "val"

    return {
        "quality_score": max(0, min(100, score)),
        "difficulty": "medium",
        "recommended_split": split,
        "issues": issues,
        "confidence": 0.35,
        "rationale": "Heuristic fallback due to Gemini response error.",
    }


def _assess_with_gemini(
    *,
    client: genai.Client,
    config: GeminiAssessmentConfig,
    image_bytes: bytes,
) -> dict:
    prompt = (
        "Evaluate this portrait-matting training sample. "
        "Return strict JSON with keys: quality_score (0-100), difficulty (easy|medium|hard), "
        "recommended_split (train|val|exclude), issues (string[]), confidence (0-1), rationale (short). "
        "Consider edge detail, blur, motion blur, lighting, occlusions, and background complexity."
    )

    schema = {
        "type": "object",
        "properties": {
            "quality_score": {"type": "integer"},
            "difficulty": {"type": "string"},
            "recommended_split": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": [
            "quality_score",
            "difficulty",
            "recommended_split",
            "issues",
            "confidence",
            "rationale",
        ],
    }

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=config.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.05,
                    max_output_tokens=config.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, dict):
                return _normalize_assessment(parsed)

            text = response.text if getattr(response, "text", None) else ""
            if not text:
                raise ValueError("Empty Gemini response")

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Some responses may include wrappers/text; extract first JSON object.
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    raise
                data = json.loads(m.group(0))

            if not isinstance(data, dict):
                raise ValueError("Gemini response is not a JSON object")
            return _normalize_assessment(data)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Gemini assessment failed: {last_error}")


def assess_dataset_with_gemini(
    *,
    dataset_path: Path,
    config: GeminiAssessmentConfig,
    limit: int = 200,
    overwrite: bool = False,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    images_dir = dataset_path / "images"
    alphas_dir = dataset_path / "alphas"
    if not images_dir.exists():
        raise ValueError("Dataset must contain an images/ directory")

    all_images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not all_images:
        raise ValueError("No supported images found in dataset")

    max_images = max(1, min(limit, len(all_images)))
    image_paths = all_images[:max_images]

    out_jsonl = dataset_path / ASSESSMENT_JSONL
    out_summary = dataset_path / ASSESSMENT_SUMMARY
    if overwrite and out_jsonl.exists():
        out_jsonl.unlink()
    if overwrite and out_summary.exists():
        out_summary.unlink()

    client = genai.Client(api_key=config.api_key)
    rows: list[dict] = []
    issue_counts: dict[str, int] = {}
    split_counts = {"train": 0, "val": 0, "exclude": 0}
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    processed = 0
    failed = 0

    for i, image_path in enumerate(image_paths, start=1):
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        alpha_path = _find_alpha_for_image(image_path, images_dir, alphas_dir)
        alpha_cov = _alpha_coverage(alpha_path)

        try:
            image_bytes = _image_to_jpeg_bytes(image_path)
            assessment = _assess_with_gemini(
                client=client,
                config=config,
                image_bytes=image_bytes,
            )
            err = None
        except Exception as exc:
            failed += 1
            assessment = _fallback_assessment(width, height, alpha_cov)
            err = str(exc)

        for issue in assessment["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        split_counts[assessment["recommended_split"]] += 1
        difficulty_counts[assessment["difficulty"]] += 1

        row = {
            "index": i,
            "image_path": str(image_path.relative_to(dataset_path)),
            "width": width,
            "height": height,
            "alpha_path": str(alpha_path.relative_to(dataset_path)) if alpha_path else None,
            "alpha_coverage": alpha_cov,
            "assessment": assessment,
            "model_error": err,
            "assessed_at": int(time.time()),
            "model": config.model,
        }
        rows.append(row)
        processed += 1

        if progress_cb:
            progress_cb(
                {
                    "processed": processed,
                    "total": max_images,
                    "failed": failed,
                    "current_image": row["image_path"],
                }
            )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    avg_quality = float(
        np.mean([float(r["assessment"]["quality_score"]) for r in rows]) if rows else 0.0
    )
    summary = {
        "dataset_id": dataset_path.name,
        "model": config.model,
        "total_images": len(all_images),
        "assessed_images": len(rows),
        "failed_images": failed,
        "avg_quality_score": round(avg_quality, 2),
        "split_counts": split_counts,
        "difficulty_counts": difficulty_counts,
        "top_issues": sorted(
            [{"issue": k, "count": v} for k, v in issue_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:12],
        "generated_at": int(time.time()),
        "files": {
            "details_jsonl": str(out_jsonl.name),
            "summary_json": str(out_summary.name),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_assessment_results(dataset_path: Path) -> dict:
    summary_path = dataset_path / ASSESSMENT_SUMMARY
    details_path = dataset_path / ASSESSMENT_JSONL
    if not summary_path.exists():
        raise FileNotFoundError("No Gemini assessment summary found")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    details: list[dict] = []
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                details.append(json.loads(line))

    return {"summary": summary, "details": details}

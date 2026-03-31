# rag/experiment_manager.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import pandas as pd


EXPERIMENT_DIR = Path("experiment_runs")

# file names
QUESTION_LEVEL_RESULTS_CSV = EXPERIMENT_DIR / "question_level_results.csv"
RUNS_LOG_JSON = EXPERIMENT_DIR / "experiment_runs_log.json"


def ensure_experiment_dir() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_key(text: str) -> str:
    return (text or "").strip().lower()


def is_duplicate_run(pdf_name: str, experiment_id: str) -> bool:
    """
    Duplicate = same PDF name + same experiment_id
    """
    if not RUNS_LOG_JSON.exists():
        return False

    with open(RUNS_LOG_JSON, "r", encoding="utf-8") as f:
        runs = json.load(f)

    target_pdf = _normalize_key(pdf_name)
    target_exp = _normalize_key(experiment_id)

    for run in runs:
        existing_pdf = _normalize_key(run.get("pdf_name", ""))
        existing_exp = _normalize_key(run.get("experiment_id", ""))
        if existing_pdf == target_pdf and existing_exp == target_exp:
            return True

    return False


def save_experiment_run(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    pdf_name: str,
    config: Dict[str, Any],
    notes: str = "",
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Save one evaluation run.
    Returns: (saved_ok, message)
    """
    ensure_experiment_dir()

    experiment_id = config.get("experiment_id", "")

    if not overwrite and is_duplicate_run(pdf_name=pdf_name, experiment_id=experiment_id):
        return (
            False,
            f"Duplicate run detected for PDF '{pdf_name}' with experiment ID '{experiment_id}'."
        )

    run_df = results_df.copy()
    run_df["source_pdf"] = pdf_name
    run_df["run_notes"] = notes
    run_df["experiment_id"] = experiment_id

    for key, value in config.items():
        run_df[f"config_{key}"] = value

    if QUESTION_LEVEL_RESULTS_CSV.exists():
        existing = pd.read_csv(QUESTION_LEVEL_RESULTS_CSV)
        combined = pd.concat([existing, run_df], ignore_index=True)
    else:
        combined = run_df

    combined.to_csv(QUESTION_LEVEL_RESULTS_CSV, index=False)

    run_record = {
        "experiment_id": experiment_id,
        "pdf_name": pdf_name,
        "notes": notes,
        "num_questions": int(len(results_df)),
        "summary": summary,
        "config": config,
    }

    if RUNS_LOG_JSON.exists():
        with open(RUNS_LOG_JSON, "r", encoding="utf-8") as f:
            runs = json.load(f)
    else:
        runs = []

    runs.append(run_record)

    with open(RUNS_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    return True, "Evaluation run saved successfully."


def load_all_results() -> Optional[pd.DataFrame]:
    if not QUESTION_LEVEL_RESULTS_CSV.exists():
        return None
    return pd.read_csv(QUESTION_LEVEL_RESULTS_CSV)


def load_runs_log() -> List[dict]:
    if not RUNS_LOG_JSON.exists():
        return []
    with open(RUNS_LOG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_all_experiments() -> None:
    if QUESTION_LEVEL_RESULTS_CSV.exists():
        QUESTION_LEVEL_RESULTS_CSV.unlink()
    if RUNS_LOG_JSON.exists():
        RUNS_LOG_JSON.unlink()


def get_available_experiment_ids(runs_log: List[dict]) -> List[str]:
    ids = []
    seen = set()
    for run in runs_log:
        exp_id = run.get("experiment_id", "")
        if exp_id and exp_id not in seen:
            ids.append(exp_id)
            seen.add(exp_id)
    return ids


def filter_results_by_experiment_id(
    all_results_df: pd.DataFrame,
    experiment_id: str
) -> pd.DataFrame:
    if all_results_df is None or all_results_df.empty:
        return pd.DataFrame()

    if "experiment_id" not in all_results_df.columns:
        return pd.DataFrame()

    return all_results_df[
        all_results_df["experiment_id"].astype(str) == str(experiment_id)
    ].copy()


def filter_runs_by_experiment_id(
    runs_log: List[dict],
    experiment_id: str
) -> List[dict]:
    return [run for run in runs_log if str(run.get("experiment_id", "")) == str(experiment_id)]
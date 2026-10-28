"""Freeze the exploratory subject split for the recorded EEG demo."""

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import mne
from mne.datasets import eegbci

from clean import FILENAME, MAX_REJECTION_FRACTION, MIN_KEPT_PER_CLASS, QC_DIR
from loader import DATA_ROOT, RUNS, SUBJECTS, get_files


REPLAY_SUBJECT = 20
OUTPUT = Path(__file__).resolve().parent / "demo_artifacts" / "split.json"


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _recording_metadata(path):
    with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as raw:
        eegbci.standardize(raw)
        return {
            "channels": raw.ch_names,
            "channel_types": raw.get_channel_types(),
            "sfreq": float(raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
            "annotation_counts": {
                label: int(sum(raw.annotations.description == label))
                for label in sorted(set(raw.annotations.description))
            },
        }


def build_split(data_root=DATA_ROOT, qc_dir=QC_DIR):
    """Inspect recording headers and freeze groups using existing, hash-verified QC."""
    data_root, qc_dir = Path(data_root).resolve(), Path(qc_dir).resolve()
    policy_path, summary_path = qc_dir / "policy.json", qc_dir / "summary.csv"
    policy = json.loads(policy_path.read_text())
    if (policy["min_kept_per_class"] != MIN_KEPT_PER_CLASS
            or policy["max_rejection_fraction_per_class"] != MAX_REJECTION_FRACTION):
        raise ValueError("QC thresholds differ from clean.py; regenerate the QC reports")
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    qc = {}
    for row in rows:
        for key in ("subject", "run", "total", "kept", "rejected"):
            row[key] = int(row[key])
        if (row["total"] <= 0 or row["kept"] < 0 or row["rejected"] < 0
                or row["total"] != row["kept"] + row["rejected"]):
            raise ValueError("QC summary contains invalid counts")
        row["review_required"] = (row["kept"] < MIN_KEPT_PER_CLASS
                                  or row["rejected"] / row["total"] > MAX_REJECTION_FRACTION)
        key = row["subject"], row["run"], row["condition"]
        if key in qc:
            raise ValueError(f"Duplicate QC row for {key}")
        qc[key] = row

    recordings = []
    for path in get_files(data_root):
        match = FILENAME.fullmatch(path.name)
        subject, run = int(match["subject"]), int(match["run"])
        digest = _sha256(path)
        if digest != policy["source_sha256"].get(path.name):
            raise ValueError(f"Recording does not match the QC source hash: {path.name}")
        counts = {}
        for condition in ("left_hand", "right_hand"):
            counts[condition] = qc[subject, run, condition]
        metadata = _recording_metadata(path)
        for condition, annotation in (("left_hand", "T1"), ("right_hand", "T2")):
            if metadata["annotation_counts"].get(annotation, 0) != counts[condition]["total"]:
                raise ValueError(f"QC trial count does not match annotations: {path.name}")
        recordings.append({
            "subject": subject, "run": run, "path": str(path.relative_to(data_root)),
            "sha256": digest, "metadata": metadata, "qc": counts,
        })

    training, excluded = [], []
    for subject in sorted(set(SUBJECTS) | {REPLAY_SUBJECT}):
        subject_files = [record for record in recordings if record["subject"] == subject]
        missing_runs = sorted(set(RUNS) - {record["run"] for record in subject_files})
        flagged_runs = [record["run"] for record in subject_files
                        if any(row["review_required"] for row in record["qc"].values())]
        reason = (f"Missing required runs {missing_runs}" if missing_runs else
                  f"QC review required for runs {flagged_runs}" if flagged_runs else None)
        if reason:
            if subject == REPLAY_SUBJECT:
                raise ValueError(f"Replay subject {subject} is unusable: {reason}")
            excluded.append({"subject": subject, "reason": reason})
        elif subject != REPLAY_SUBJECT:
            training.append(subject)
    if not training:
        raise ValueError("No usable training subjects")
    for record in recordings:
        record["partition"] = ("replay" if record["subject"] == REPLAY_SUBJECT else
                               "training" if record["subject"] in training else "excluded")
    return {
        "schema_version": 1,
        "status": "exploratory_development",
        "inspection_history": "These subjects have already informed notebook exploration and tuning; this is not untouched final validation.",
        "selection_rule": "Replay subject 20 is fixed independently of model accuracy. Training requires all imagery runs and no QC review flags.",
        "data_root": str(data_root),
        "required_runs": list(RUNS),
        "training_subjects": training,
        "replay_subjects": [REPLAY_SUBJECT],
        "excluded_subjects": excluded,
        "qc_provenance": {
            "source": "existing cached QC; preprocessing was not rerun",
            "directory": str(qc_dir),
            "policy_sha256": _sha256(policy_path),
            "summary_sha256": _sha256(summary_path),
            "policy_mne_version": policy["mne_version"],
        },
        "inspection_versions": {"python": platform.python_version(), "mne": mne.__version__},
        "recordings": recordings,
    }


def main():
    manifest = build_split()
    contents = json.dumps(manifest, indent=2) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=OUTPUT.parent) as handle:
        handle.write(contents)
        handle.flush()
        os.link(handle.name, OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Training subjects: {manifest['training_subjects']}")
    print(f"Replay subjects: {manifest['replay_subjects']}")
    print(f"Excluded subjects: {manifest['excluded_subjects']}")


if __name__ == "__main__":
    main()

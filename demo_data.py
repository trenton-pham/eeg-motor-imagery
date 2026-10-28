"""Prepare the frozen demo split as offline EEG windows and a trial timeline."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import warnings

import mne
import numpy as np
import pandas as pd
import scipy
import sklearn

import clean
from demo_split import _sha256


ROOT = Path(__file__).resolve().parent
TRIAL_ID = ["subject", "run", "source_file", "trial_in_run", "event_sample"]
LABELS = {1: "left_hand", 2: "right_hand"}


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _selected_records(split):
    if split["schema_version"] != 1:
        raise ValueError("Unsupported split schema")
    training, replay = split["training_subjects"], split["replay_subjects"]
    if (not training or not replay or len(set(training)) != len(training)
            or len(set(replay)) != len(replay) or set(training) & set(replay)
            or any(type(subject) is not int or subject < 1 for subject in training + replay)):
        raise ValueError("Training and replay subjects must be nonempty, unique, and disjoint")
    groups = {subject: "training" for subject in training}
    groups.update({subject: "replay" for subject in replay})
    provenance = split["qc_provenance"]
    for filename, key in (("policy.json", "policy_sha256"), ("summary.csv", "summary_sha256")):
        if _sha256(Path(provenance["directory"]) / filename) != provenance[key]:
            raise ValueError(f"Frozen QC provenance changed: {filename}")
    root = Path(split["data_root"]).resolve()
    records, paths, seen = [], [], set()
    for record in split["recordings"]:
        expected = groups.get(record["subject"], "excluded")
        if record["partition"] != expected:
            raise ValueError("Recording partition disagrees with subject membership")
        if expected == "excluded":
            continue
        key = record["subject"], record["run"]
        path = (root / record["path"]).resolve()
        if key in seen or path in paths:
            raise ValueError("Duplicate selected recording")
        seen.add(key)
        if not path.is_relative_to(root):
            raise ValueError("Recording path escapes the frozen data root")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen source hash changed: {path.name}")
        records.append(record)
        paths.append(path)
    expected = {(subject, run) for subject in groups for run in split["required_runs"]}
    if (not split["required_runs"] or len(set(split["required_runs"])) != len(split["required_runs"])
            or seen != expected):
        raise ValueError("Selected subjects do not have exactly the required runs")
    return records, paths


def _trial_timeline(dataset, records):
    epochs = dataset.epochs
    if not len(epochs):
        raise ValueError("No retained EEG windows")
    if not np.isfinite(epochs.get_data(copy=False)).all():
        raise ValueError("EEG windows must contain finite data")
    trials = dataset.trial_qc.copy().reset_index(drop=True)
    catalogue = {(record["subject"], record["run"]): record for record in records}
    trials["partition"] = [catalogue[subject, run]["partition"]
                           for subject, run in zip(trials.subject, trials.run)]
    for key, record in catalogue.items():
        selected = trials.subject.eq(key[0]) & trials.run.eq(key[1])
        rows = trials.loc[selected]
        header = record["metadata"]
        if (epochs.ch_names != header["channels"]
                or epochs.get_channel_types() != header["channel_types"]
                or epochs.info["sfreq"] != header["sfreq"]):
            raise ValueError("EEG channel order, types, or sample rate changed")
        for condition in LABELS.values():
            condition_rows = rows.loc[rows.condition.eq(condition)]
            expected = record["qc"][condition]
            if (len(condition_rows) != expected["total"]
                    or int(condition_rows.kept.sum()) != expected["kept"]):
                raise ValueError("Fresh QC counts differ from the frozen split")
    trial_keys = pd.MultiIndex.from_frame(trials[TRIAL_ID])
    epoch_keys = pd.MultiIndex.from_frame(epochs.metadata[TRIAL_ID])
    kept_keys = trial_keys[trials.kept.to_numpy()]
    if len(epoch_keys) != len(kept_keys) or set(epoch_keys) != set(kept_keys):
        raise ValueError("Retained epoch identities do not match kept QC trials")
    positions = trial_keys.get_indexer(epoch_keys)
    aligned = trials.iloc[positions]
    if (not np.array_equal(aligned.label.to_numpy(), epochs.events[:, -1])
            or not np.allclose(aligned.event_onset_s, epochs.metadata.event_onset_s)):
        raise ValueError("Epoch labels or original onset times disagree with trial QC")
    trials["epoch_index"] = pd.array([None] * len(trials), dtype="Int64")
    trials.loc[positions, "epoch_index"] = np.arange(len(epochs))
    trials["window_start_s"] = trials.event_onset_s + epochs.tmin
    trials["window_end_s"] = trials.event_onset_s + epochs.tmax
    epochs.metadata = epochs.metadata.assign(
        partition=aligned.partition.to_numpy(), epoch_index=np.arange(len(epochs)))
    return trials


def prepare_data(split_path, output_dir):
    """Publish validated EEG windows and QC without changing the frozen inputs."""
    split_path, output_dir = Path(split_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    split_bytes = split_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes)
    records, files = _selected_records(split)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        dataset = clean.load_dataset(files)
    trials = _trial_timeline(dataset, records)
    versions = {"python": platform.python_version(), "mne": mne.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".prepared-") as temporary:
        stage = Path(temporary)
        clean.save_qc(dataset, files, stage / "qc")
        policy = json.loads((stage / "qc/policy.json").read_text())
        if policy["source_sha256"] != {path.name: record["sha256"]
                                      for path, record in zip(files, records)}:
            raise ValueError("Source recordings changed during preprocessing")
        recipe = {key: value for key, value in policy.items() if key != "source_sha256"}
        clean_source_sha256 = _sha256(Path(clean.__file__))
        preprocessing_id = _identity({"policy": recipe, "versions": versions,
                                      "clean_source_sha256": clean_source_sha256})
        dataset.epochs.save(stage / "epochs-epo.fif", fmt="double", verbose="ERROR")
        trials.to_csv(stage / "trials.csv", index=False)
        (stage / "split.json").write_bytes(split_bytes)
        metadata = {
            "schema_version": 1, "status": split["status"],
            "split_sha256": split_sha256,
            "training_subjects": split["training_subjects"],
            "replay_subjects": split["replay_subjects"],
            "recordings": [{**record, "duration_s": record["metadata"]["n_samples"]
                            / record["metadata"]["sfreq"]} for record in records],
            "channels": dataset.epochs.ch_names, "sfreq": dataset.epochs.info["sfreq"],
            "window_seconds_after_cue": [dataset.epochs.tmin, dataset.epochs.tmax],
            "units": "V", "shape": list(dataset.epochs.get_data(copy=False).shape),
            "label_mapping": LABELS, "versions": versions,
            "preprocessing_id": preprocessing_id,
            "dataset_id": _identity({"preprocessing_id": preprocessing_id,
                "split_sha256": split_sha256,
                "sources": policy["source_sha256"],
                "retained_trials": trials.loc[trials.kept].sort_values("epoch_index")[TRIAL_ID].to_dict("records")}),
            "clean_source_sha256": clean_source_sha256,
            "warnings": sorted({str(item.message) for item in captured}),
            "continuous_eeg": False,
            "limitations": ["Whole-recording preprocessing and cue-aligned offline windows.",
                "Rest, rejected trials, and samples outside retained windows are gaps.",
                "These participants already informed development; this is exploratory data."],
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "demo_artifacts/split.json")
    parser.add_argument("--out", type=Path, default=ROOT / "demo_artifacts/prepared")
    args = parser.parse_args()
    print(f"Prepared {prepare_data(args.split, args.out)}")


if __name__ == "__main__":
    main()

"""Load the EEGBCI left/right motor-imagery development dataset."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci


DATA_ROOT = Path(__file__).resolve().parent / "data" / "raw" / "eegbci"
SUBJECTS = (1, 2, 3, 4, 5)
RUNS = (4, 8, 12)
EXPECTED_RECORDINGS = {(subject, run) for subject in SUBJECTS for run in RUNS}
FILENAME = re.compile(r"S(?P<subject>\d{3})R(?P<run>\d{2})\.edf")
QC_DIR = Path(__file__).resolve().parent / "data" / "processed" / "qc"

# QC thresholds are fixed independently of trial labels.
BAD_CHANNEL_MEDIAN_PTP_UV = 150.0
BAD_CHANNEL_RELATIVE_PTP = 5.0
FLAT_CHANNEL_MEDIAN_PTP_UV = 1.0
MAX_INTERPOLATED_CHANNELS = 4
MAX_EPOCH_PTP_UV = 250.0
MIN_EPOCH_PTP_UV = 1.0
MIN_KEPT_PER_CLASS = 5
MAX_REJECTION_FRACTION = 0.4


@dataclass
class LoadedDataset:
    epochs: mne.Epochs
    channel_qc: pd.DataFrame
    trial_qc: pd.DataFrame


def get_files(data_root: Path = DATA_ROOT) -> list[Path]:
    """Find all intended recordings, rejecting missing or duplicate files."""
    found = {}
    for path in Path(data_root).rglob("*.edf"):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            continue
        subject = int(match.group("subject"))
        run = int(match.group("run"))
        key = (subject, run)
        if key not in EXPECTED_RECORDINGS or path.parent.name != f"S{subject:03d}":
            continue
        if key in found:
            raise ValueError(f"Duplicate recording for subject {subject}, run {run}: {found[key]} and {path}")
        found[key] = path

    missing = sorted(EXPECTED_RECORDINGS - found.keys())
    if missing:
        raise FileNotFoundError(f"Missing EEGBCI recordings under {data_root}: {missing}")
    return [found[key] for key in sorted(EXPECTED_RECORDINGS)]


def download_data(data_root: Path = DATA_ROOT) -> list[Path]:
    """Download the intended runs, then validate that all 15 are available."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    eegbci.load_data(
        subjects=list(SUBJECTS),
        runs=list(RUNS),
        path=data_root,
        update_path=False,
    )
    return get_files(data_root)


def _read_recording(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    return raw


def inspect_recording(path: Path) -> None:
    """Plot the recording, power spectrum, and sensor positions."""
    raw = _read_recording(path)
    print(raw)
    print(raw.annotations)
    print(raw.ch_names)
    raw.plot(duration=20, n_channels=20, scalings="auto")
    raw.compute_psd(fmin=1, fmax=70).plot()
    raw.plot_sensors(show_names=True)


def _make_epochs_with_qc(path: Path) -> LoadedDataset:
    """Screen channels before referencing and trials after preprocessing."""
    path = Path(path)
    match = FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected EDF filename: {path.name}")
    subject, run = int(match.group("subject")), int(match.group("run"))
    if (subject, run) not in EXPECTED_RECORDINGS or path.parent.name != f"S{subject:03d}":
        raise ValueError(f"Unexpected subject or run: {path}")

    raw = _read_recording(path)
    if not np.isfinite(raw.get_data()).all():
        raise ValueError(f"Nonfinite EEG samples in {path}")
    events, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR"
    )
    if len(events) != 15:
        raise ValueError(f"Expected 15 task events in {path}, found {len(events)}")
    raw.filter(l_freq=8, h_freq=30, fir_design="firwin", verbose="ERROR")

    # Screen channels first so outliers do not contaminate the average reference.
    probe = mne.Epochs(
        raw, events, event_id={"left_hand": 1, "right_hand": 2},
        tmin=0.5, tmax=3.5, baseline=None, preload=True,
        reject_by_annotation=True, verbose="ERROR",
    )
    if len(probe) == 0:
        raise ValueError(f"No trials available for channel screening in {path}")
    probe_ptp_uv = np.ptp(probe.get_data(copy=False), axis=-1) * 1e6
    channel_medians = np.median(probe_ptp_uv, axis=0)
    reference_median = float(np.median(channel_medians))
    if reference_median <= 0:
        raise ValueError(f"No usable EEG amplitude in {path}")
    high_channel = (channel_medians > BAD_CHANNEL_MEDIAN_PTP_UV) & (
        channel_medians > BAD_CHANNEL_RELATIVE_PTP * reference_median
    )
    flat_channel = channel_medians < FLAT_CHANNEL_MEDIAN_PTP_UV
    bad_mask = high_channel | flat_channel
    bad_channels = [name for name, bad in zip(raw.ch_names, bad_mask) if bad]
    if len(bad_channels) > MAX_INTERPOLATED_CHANNELS:
        raise ValueError(f"Too many bad channels in {path}: {bad_channels}")
    channel_qc = pd.DataFrame(
        {
            "subject": subject,
            "run": run,
            "source_file": path.name,
            "channel": raw.ch_names,
            "median_ptp_uv": channel_medians,
            "run_median_ptp_uv": reference_median,
            "relative_ptp": channel_medians / reference_median,
            "reason": np.where(
                high_channel, "PERSISTENT_HIGH_PTP",
                np.where(flat_channel, "PERSISTENT_FLAT_PTP", ""),
            ),
            "interpolated": bad_mask,
        }
    )
    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left_hand": 1, "right_hand": 2},
        tmin=0.5,
        tmax=3.5,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    trial_qc = pd.DataFrame(
        {
            "subject": subject,
            "run": run,
            "source_file": path.name,
            "trial_in_run": np.arange(1, len(events) + 1),
            "event_sample": events[:, 0],
            "event_onset_s": (events[:, 0] - raw.first_samp) / raw.info["sfreq"],
            "label": events[:, -1],
            "condition": np.where(events[:, -1] == 1, "left_hand", "right_hand"),
            "max_ptp_uv": np.nan,
            "min_ptp_uv": np.nan,
            "max_ptp_channel": "",
            "drop_reason": [";".join(reason) for reason in epochs.drop_log],
            "kept": False,
        }
    )
    ptp_uv = np.ptp(epochs.get_data(copy=False), axis=-1) * 1e6
    high = np.max(ptp_uv, axis=1) > MAX_EPOCH_PTP_UV
    flat = np.min(ptp_uv, axis=1) < MIN_EPOCH_PTP_UV
    for event_index, values, is_high, is_flat in zip(epochs.selection, ptp_uv, high, flat):
        reasons = []
        if is_high:
            reasons.append("HIGH_PTP")
        if is_flat:
            reasons.append("FLAT_PTP")
        trial_qc.loc[event_index, "max_ptp_uv"] = values.max()
        trial_qc.loc[event_index, "min_ptp_uv"] = values.min()
        trial_qc.loc[event_index, "max_ptp_channel"] = raw.ch_names[values.argmax()]
        trial_qc.loc[event_index, "drop_reason"] = ";".join(reasons)
        trial_qc.loc[event_index, "kept"] = not reasons

    epochs.metadata = trial_qc.iloc[epochs.selection][
        ["subject", "run", "source_file", "trial_in_run", "event_sample", "event_onset_s"]
    ].reset_index(drop=True)
    to_drop = np.flatnonzero(high | flat)
    if len(to_drop):
        epochs.drop(to_drop, reason="QC_PTP_OR_FLAT", verbose="ERROR")
    return LoadedDataset(epochs, channel_qc, trial_qc)


def make_epochs_tuned(path: Path) -> mne.Epochs:
    """Return screened trials filtered to 8-30 Hz, 0.5-3.5 s after each cue."""
    return _make_epochs_with_qc(path).epochs


def load_dataset(files: list[Path] | None = None) -> LoadedDataset:
    """Load all intended runs with channel and trial QC tables."""
    if files is None:
        files = get_files()
    if len(files) != len(EXPECTED_RECORDINGS):
        raise ValueError(f"Expected 15 recordings, received {len(files)}")
    identities = []
    for path in files:
        match = FILENAME.fullmatch(Path(path).name)
        if match is None:
            raise ValueError(f"Unexpected EDF filename: {path}")
        identities.append((int(match.group("subject")), int(match.group("run"))))
    if set(identities) != EXPECTED_RECORDINGS:
        raise ValueError("Recordings must contain each intended subject/run pair exactly once")
    results = [_make_epochs_with_qc(path) for _, path in sorted(zip(identities, files))]
    return LoadedDataset(
        mne.concatenate_epochs([result.epochs for result in results]),
        pd.concat([result.channel_qc for result in results], ignore_index=True),
        pd.concat([result.trial_qc for result in results], ignore_index=True),
    )


def load_epochs(files: list[Path] | None = None) -> mne.Epochs:
    """Return the screened MNE Epochs object for modeling."""
    return load_dataset(files).epochs


def save_qc(dataset: LoadedDataset, files: list[Path], qc_dir: Path = QC_DIR) -> pd.DataFrame:
    """Save QC tables, preprocessing settings, and source file hashes."""
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    dataset.channel_qc.to_csv(qc_dir / "channels.csv", index=False)
    dataset.trial_qc.to_csv(qc_dir / "trials.csv", index=False)
    summary = dataset.trial_qc.groupby(
        ["subject", "run", "condition"], as_index=False
    ).agg(total=("kept", "size"), kept=("kept", "sum"))
    summary["rejected"] = summary["total"] - summary["kept"]
    summary["review_required"] = (
        (summary["kept"] < MIN_KEPT_PER_CLASS)
        | (summary["rejected"] / summary["total"] > MAX_REJECTION_FRACTION)
    )
    summary.to_csv(qc_dir / "summary.csv", index=False)
    hashes = {}
    for path in files:
        path = Path(path)
        with path.open("rb") as handle:
            hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    policy = {
        "mne_version": mne.__version__,
        "filter_hz": [8, 30],
        "epoch_seconds_after_cue": [0.5, 3.5],
        "reference": "average after bad-channel interpolation",
        "bad_channel_median_ptp_uv": BAD_CHANNEL_MEDIAN_PTP_UV,
        "bad_channel_relative_ptp": BAD_CHANNEL_RELATIVE_PTP,
        "flat_channel_median_ptp_uv": FLAT_CHANNEL_MEDIAN_PTP_UV,
        "max_interpolated_channels_per_run": MAX_INTERPOLATED_CHANNELS,
        "max_epoch_ptp_uv": MAX_EPOCH_PTP_UV,
        "min_epoch_ptp_uv": MIN_EPOCH_PTP_UV,
        "min_kept_per_class": MIN_KEPT_PER_CLASS,
        "max_rejection_fraction_per_class": MAX_REJECTION_FRACTION,
        "source_sha256": hashes,
    }
    (qc_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="show plots for the first recording")
    parser.add_argument("--inspect-file", metavar="EDF", help="show plots for a named recording")
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR, help="directory for QC reports")
    args = parser.parse_args()

    try:
        files = get_files()
    except FileNotFoundError:
        files = download_data()
    print(f"Found {len(files)} intended recordings")
    if args.inspect:
        inspect_recording(files[0])
    if args.inspect_file:
        selected = [path for path in files if path.name == args.inspect_file]
        if not selected:
            parser.error(f"Recording not found: {args.inspect_file}")
        inspect_recording(selected[0])
    dataset = load_dataset(files)
    epochs = dataset.epochs
    summary = save_qc(dataset, files, args.qc_dir)
    print(f"Epochs: {len(epochs)}, shape: {epochs.get_data(copy=False).shape}")
    counts = pd.Series(epochs.events[:, -1]).value_counts().sort_index().to_dict()
    print(f"Label counts (1=left, 2=right): {counts}")
    print(summary.to_string(index=False))
    flagged = summary.loc[summary["review_required"]]
    if not flagged.empty:
        print("Review required before interpreting model scores for these class/run cells:")
        print(flagged[["subject", "run", "condition", "kept", "rejected"]].to_string(index=False))
    print(f"QC reports: {args.qc_dir}")


if __name__ == "__main__":
    main()

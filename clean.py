"""Clean and epoch the downloaded EEGBCI recordings."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from threadpoolctl import threadpool_limits

from loader import get_files


FILENAME = re.compile(r"S(?P<subject>\d{3})R(?P<run>\d{2})\.edf")
QC_DIR = Path(__file__).resolve().parent / "data" / "processed" / "qc"

# QC thresholds are fixed independently of trial labels.
BAD_CHANNEL_MEDIAN_PTP_UV = 150.0
BAD_CHANNEL_RELATIVE_PTP = 5.0
FLAT_CHANNEL_MEDIAN_PTP_UV = 1.0
MAX_EPOCH_PTP_UV = 250.0
MIN_EPOCH_PTP_UV = 1.0
MIN_KEPT_PER_CLASS = 5
MAX_REJECTION_FRACTION = 0.4
ICA_PARAMS = {
    "n_components": 0.99,
    "method": "infomax",
    "random_state": 97,
    "max_iter": 1000,
    "fit_params": {"extended": True, "verbose": True},
}
ICA_POLICY = {
    **ICA_PARAMS,
    "fit_filter_hz": [1, 40],
    "proxy_channels": ["Fp1", "Fp2"],
    "eog_zscore_threshold": 3.0,
    "minimum_absolute_proxy_correlation": 0.5,
    "pca_reconstruction": "all components",
    "nonconvergence": "leave recording unchanged",
}


@dataclass
class LoadedDataset:
    epochs: mne.Epochs
    channel_qc: pd.DataFrame
    trial_qc: pd.DataFrame
    ica_qc: pd.DataFrame


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


def _screen_channels(raw, events, path, subject, run):
    probe = mne.Epochs(
        raw, events, event_id={"left_hand": 1, "right_hand": 2},
        tmin=0.5, tmax=3.5, baseline=None, preload=True,
        reject_by_annotation=True, verbose="ERROR",
    )
    probe_ptp_uv = np.ptp(probe.get_data(copy=False), axis=-1) * 1e6
    channel_medians = np.median(probe_ptp_uv, axis=0)
    reference_median = float(np.median(channel_medians))
    high_channel = (channel_medians > BAD_CHANNEL_MEDIAN_PTP_UV) & (
        channel_medians > BAD_CHANNEL_RELATIVE_PTP * reference_median
    )
    flat_channel = channel_medians < FLAT_CHANNEL_MEDIAN_PTP_UV
    bad_mask = high_channel | flat_channel
    return pd.DataFrame(
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


def _infomax_stopping(log, max_iter):
    steps = re.findall(r"^step (\d+) -", log, flags=re.MULTILINE)
    actual_steps = int(steps[-1])
    return actual_steps, actual_steps < max_iter


def _apply_ica(raw, bad_channels):
    proxies = ICA_POLICY["proxy_channels"]
    fit_raw = raw.copy()
    fit_raw.info["bads"] = bad_channels
    fit_raw.filter(*ICA_POLICY["fit_filter_hz"], fir_design="firwin", verbose="ERROR")
    ica = mne.preprocessing.ICA(**ICA_PARAMS)
    fit_log = StringIO()
    with threadpool_limits(limits=1), redirect_stdout(fit_log):
        ica.fit(fit_raw, verbose=True)
    rank = mne.compute_rank(fit_raw, rank=None, verbose="ERROR")["eeg"]
    # Infomax can replace n_iter_ with max_iter after a successful early stop.
    actual_steps, converged = _infomax_stopping(fit_log.getvalue(), ica.max_iter)
    candidates, scores = ica.find_bads_eog(
        fit_raw, ch_name=proxies, threshold=ICA_POLICY["eog_zscore_threshold"], verbose="ERROR"
    )
    scores = np.asarray(scores)
    excluded = [
        int(index) for index in candidates
        if np.max(np.abs(scores[:, index])) >= ICA_POLICY["minimum_absolute_proxy_correlation"]
    ] if converged else []
    if excluded:
        ica.apply(raw, exclude=excluded, verbose="ERROR")
    return {
        "n_components": int(ica.n_components_),
        "rank": int(rank),
        "n_iter": int(ica.n_iter_),
        "actual_steps": actual_steps,
        "converged": converged,
        "excluded_components": json.dumps(excluded),
        "candidates": json.dumps([int(index) for index in candidates]),
        "proxy_correlations": json.dumps(scores.tolist()),
        "bad_channels_excluded_from_fit": json.dumps(bad_channels),
    }


def _make_epochs_with_qc(path: Path) -> LoadedDataset:
    """Screen channels before referencing and trials after preprocessing."""
    path = Path(path)
    match = FILENAME.fullmatch(path.name)
    subject, run = int(match.group("subject")), int(match.group("run"))

    raw = _read_recording(path)
    events, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR"
    )
    raw.notch_filter(freqs=[60], fir_design="firwin", verbose="ERROR")

    baseline = raw.copy().filter(l_freq=8, h_freq=30, fir_design="firwin", verbose="ERROR")
    baseline_qc = _screen_channels(baseline, events, path, subject, run)
    bad_channels = baseline_qc.loc[baseline_qc["interpolated"], "channel"].tolist()
    ica_report = _apply_ica(raw, bad_channels)
    raw.filter(l_freq=8, h_freq=30, fir_design="firwin", verbose="ERROR")
    channel_qc = _screen_channels(raw, events, path, subject, run)
    bad_channels = channel_qc.loc[channel_qc["interpolated"], "channel"].tolist()
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
    ica_qc = pd.DataFrame([{
        "subject": subject, "run": run, "source_file": path.name, **ica_report,
    }])
    return LoadedDataset(epochs, channel_qc, trial_qc, ica_qc)


def make_epochs_tuned(path: Path) -> mne.Epochs:
    """Return screened trials filtered to 8-30 Hz, 0.5-3.5 s after each cue."""
    return _make_epochs_with_qc(path).epochs


def load_dataset(files: list[Path] | None = None) -> LoadedDataset:
    """Load all intended runs with channel and trial QC tables."""
    if files is None:
        files = get_files()
    if not files:
        raise ValueError("No EEG recordings found; run loader.py to download data first")
    results = [_make_epochs_with_qc(path) for path in sorted(files)]
    return LoadedDataset(
        mne.concatenate_epochs([result.epochs for result in results]),
        pd.concat([result.channel_qc for result in results], ignore_index=True),
        pd.concat([result.trial_qc for result in results], ignore_index=True),
        pd.concat([result.ica_qc for result in results], ignore_index=True),
    )


def load_epochs(files: list[Path] | None = None) -> mne.Epochs:
    """Return the screened MNE Epochs object for modeling."""
    return load_dataset(files).epochs


def save_qc(dataset: LoadedDataset, files: list[Path], qc_dir: Path = QC_DIR) -> pd.DataFrame:
    """Save QC tables, preprocessing settings, and source file hashes."""
    qc_dir = Path(qc_dir)
    summary = dataset.trial_qc.groupby(
        ["subject", "run", "condition"], as_index=False
    ).agg(total=("kept", "size"), kept=("kept", "sum"))
    summary["rejected"] = summary["total"] - summary["kept"]
    summary["review_required"] = (
        (summary["kept"] < MIN_KEPT_PER_CLASS)
        | (summary["rejected"] / summary["total"] > MAX_REJECTION_FRACTION)
    )
    hashes = {}
    for path in files:
        path = Path(path)
        with path.open("rb") as handle:
            hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    policy = {
        "mne_version": mne.__version__,
        "notch_hz": [60],
        "ica": ICA_POLICY,
        "filter_hz": [8, 30],
        "epoch_seconds_after_cue": [0.5, 3.5],
        "reference": "average after bad-channel interpolation",
        "bad_channel_median_ptp_uv": BAD_CHANNEL_MEDIAN_PTP_UV,
        "bad_channel_relative_ptp": BAD_CHANNEL_RELATIVE_PTP,
        "flat_channel_median_ptp_uv": FLAT_CHANNEL_MEDIAN_PTP_UV,
        "max_epoch_ptp_uv": MAX_EPOCH_PTP_UV,
        "min_epoch_ptp_uv": MIN_EPOCH_PTP_UV,
        "min_kept_per_class": MIN_KEPT_PER_CLASS,
        "max_rejection_fraction_per_class": MAX_REJECTION_FRACTION,
        "source_sha256": hashes,
    }
    policy_json = json.dumps(policy, indent=2) + "\n"
    qc_dir.mkdir(parents=True, exist_ok=True)
    # shortcut: QC files publish separately; use a versioned export if crash recovery is required.
    dataset.channel_qc.to_csv(qc_dir / "channels.csv", index=False)
    dataset.trial_qc.to_csv(qc_dir / "trials.csv", index=False)
    dataset.ica_qc.to_csv(qc_dir / "ica.csv", index=False)
    summary.to_csv(qc_dir / "summary.csv", index=False)
    (qc_dir / "policy.json").write_text(policy_json)
    return summary


if __name__ == "__main__":
    files = get_files()
    dataset = load_dataset(files)
    save_qc(dataset, files)
    print(f"Kept {len(dataset.epochs)} of {len(dataset.trial_qc)} trials")

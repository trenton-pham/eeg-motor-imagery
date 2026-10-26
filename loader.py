"""Download and locate EEGBCI motor-imagery recordings."""

from pathlib import Path

from mne.datasets import eegbci


DATA_ROOT = Path(__file__).resolve().parent / "data" / "raw" / "eegbci"
SUBJECTS = tuple(range(1, 21))
RUNS = (4, 8, 12)


def get_files(data_root: Path = DATA_ROOT) -> list[Path]:
    """Find downloaded recordings for the selected subjects and runs."""
    names = {f"S{subject:03d}R{run:02d}.edf" for subject in SUBJECTS for run in RUNS}
    return sorted(path for path in Path(data_root).rglob("*.edf") if path.name in names)


def download_data(data_root: Path = DATA_ROOT) -> list[Path]:
    """Download the selected subjects and runs."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    eegbci.load_data(
        subjects=list(SUBJECTS),
        runs=list(RUNS),
        path=data_root,
        update_path=False,
    )
    return get_files(data_root)


if __name__ == "__main__":
    download_data()
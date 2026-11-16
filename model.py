from pathlib import Path
import sys

import mne

from loader import get_files
from clean import load_dataset, save_qc

def root_path():
    project_root = next(
        path for path in (Path.cwd(), *Path.cwd().parents)
        if (path / "clean.py").is_file()
    )
    sys.path.insert(0, str(project_root))
    mne.set_log_level("WARNING")

def load_files():
    files = get_files()
    dataset = load_dataset(files)
    qc_summary = save_qc(dataset, files)

    print(f"Kept {len(dataset.epochs)} of {len(dataset.trial_qc)} trials")
    print(f"Interpolated channel/run pairs: {dataset.channel_qc['interpolated'].sum()}")
    print(qc_summary.loc[qc_summary["review_required"]].to_string(index=False))

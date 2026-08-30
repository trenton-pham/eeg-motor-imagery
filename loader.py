from pathlib import Path
from mne.datasets import eegbci

data_root = Path("data/raw/eegbci").resolve()
data_root.mkdir(parents=True, exist_ok=True)

subjects = [1, 2, 3, 4, 5]
runs = [4, 8, 12]

files = eegbci.load_data(
    subjects=subjects,
    runs=runs,
    path=data_root,
    update_path=False,
)

print(f"Downloaded/found {len(files)} files")

for file in files:
    print(file)
Download recordings with `python loader.py`. Clean downloaded recordings and write QC reports with `python clean.py`.

```python
from loader import get_files
from clean import load_dataset, save_qc

files = get_files()
dataset = load_dataset(files)
save_qc(dataset, files)
epochs = dataset.epochs
```

Cleaning applies the 60 Hz notch, ICA, the 8-30 Hz filter, and channel and trial QC. It does not download files. There are no custom validation exceptions or recovery handlers; Python and MNE report their own failures.

### References
- <a href="https://mne.tools/stable/index.html">MNE Documentation</a>
- <a href="https://medium.com/@hirok4/theory-and-implementation-of-csp-common-spatial-pattern-algorithm-f84300e3b4d6">Theory and Implementation of CSP Algorithm</a>
- Codex/ChatGPT

Download recordings with `python loader.py`. Clean downloaded recordings and write QC reports with `python clean.py`.

```python
from loader import get_files
from clean import load_dataset, save_qc

files = get_files()
dataset = load_dataset(files)
save_qc(dataset, files)
epochs = dataset.epochs
```

Cleaning applies the 60 Hz notch, ICA, the 8-30 Hz filter, and channel and trial QC. It does not download files.

### Freeze the demo split

After generating the existing QC reports, run `python demo_split.py` in the project environment.

This creates `demo_artifacts/split.json` and refuses to overwrite an existing split. Subject 20 is the fixed replay subject. Training uses the other subjects with all three imagery runs and no QC review flags. With the current reports, subjects 9, 13, 17, and 19 are excluded, leaving 15 training subjects.

The manifest records recording hashes, header metadata, class counts, exclusions, and QC report hashes. It verifies that recordings still match the cached QC sources. It does not repeat preprocessing or train a model. This split is exploratory because these subjects have already informed model development. Keep genuinely unused data separate for final evaluation.

Regenerate QC after changing preprocessing before creating a new split. Source-hash checks do not validate later edits to `clean.py`.

### References
- <a href="https://mne.tools/stable/index.html">MNE Documentation</a>
- <a href="https://medium.com/@hirok4/theory-and-implementation-of-csp-common-spatial-pattern-algorithm-f84300e3b4d6">Theory and Implementation of CSP Algorithm</a>
- <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10856895/">Convolutional Neural Network Architectures for EEG Feature Extraction</a>
- <a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00039/full">Filter Bank Common Spatial Pattern Algorithm</a>
- Codex/ChatGPT

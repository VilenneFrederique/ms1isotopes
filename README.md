# ms1isotopes

[![PyPI](https://img.shields.io/pypi/v/ms1isotopes)](https://pypi.org/project/ms1isotopes/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://github.com/VilenneFrederique/ms1isotopes/actions/workflows/tests.yml/badge.svg)](https://github.com/VilenneFrederique/ms1isotopes/actions)

A Python package for extracting and evaluating MS1 isotope distributions
from LC-MS/MS proteomics data.

## What it does

Given an mzML file and a list of peptide identifications, `ms1isotopes`:

1. Extracts experimental isotope envelopes (m/z + intensity for up to 6 peaks)
   across the chromatographic elution window
2. Computes theoretical isotope distributions using the
   [BRAIN algorithm](https://doi.org/10.1021/ac303439m) (via `brainpy`)
3. Scores similarity between experimental and theoretical distributions
   using the spectral angle
4. Outputs annotated results in Excel, TSV, or hierarchical JSON

## Installation

```bash
pip install ms1isotopes
```

### Dependencies

- Python ≥ 3.9
- numpy, pandas, numba
- pyteomics (for mzML parsing)
- lxml (XML backend for pyteomics)
- brainpy (brain-isotopic-distribution)
- openpyxl (for Excel I/O)

## Quick start

### Command line

```bash
ms1isotopes extract \
    -m sample.mzML \
    -i identifications.xlsx \
    -o results \
    --tolerance 5
```

This produces `results.xlsx`, `results.tsv`, and `results.json`.

### Python API

```python
import ms1isotopes

# Load input list (Excel, CSV, or TSV)
input_list = ms1isotopes.read_input_list("identifications.xlsx")

# Run extraction
results = ms1isotopes.extract_ms1_isotope_distributions(
    mzml_path="sample.mzML",
    input_list=input_list,
    tolerance_ppm=5.0,
    n_isotopes=6,
)

# Save results
ms1isotopes.results_to_excel(results, "output.xlsx")
ms1isotopes.results_to_json(results, "output.json")
```

### Computing isotope distributions without mzML data

```python
import ms1isotopes
import numpy as np

# Peptide → atomic composition
comp = ms1isotopes.sequence_to_composition(
    "PEPTIDE", charge_state=2, n_carbamidomethyl=0
)
# → {'C': 34, 'H': 57, 'N': 7, 'O': 15, 'S': 0}

# Theoretical isotope distribution (via BRAIN)
mz_values, intensities = ms1isotopes.theoretical_isotope_distribution(comp, n_peaks=6)

# Spectral angle against experimental data
experimental = np.array([10000, 5500, 2000, 500, np.nan, np.nan])
sa = ms1isotopes.spectral_angle(intensities, experimental)
```

## Input list format

The input list must contain these columns:

| Column | Description |
|--------|-------------|
| `Peptide_Sequence` | Amino acid sequence (single-letter codes) |
| `Charge_State` | Precursor charge state |
| `Modifications` | PTMs, e.g. `C(57.0214)M(15.9949)` or empty |
| `RetentionTime` | PSM retention time (seconds) |
| `RetentionTimeWindowBefore` | Extraction window before RT (seconds) |
| `RetentionTimeWindowAfter` | Extraction window after RT (seconds) |

Optional columns: `MZ` (observed m/z), `ErrorTolerance` (ppm).

## Output columns

The output DataFrame contains experimental isotope peaks, theoretical
distributions from BRAIN, atomic composition, spectral angle, peak counts,
and metadata. See the
[documentation](https://github.com/VilenneFrederique/ms1isotopes/wiki)
for the full schema.

## Testing

```bash
pip install ms1isotopes[dev]
pytest tests/
```

## Citation

If you use ms1isotopes in your research, please cite:

> [Work in progress]

## License

GPLv3. See [LICENSE](LICENSE) for details.

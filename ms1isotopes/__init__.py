"""
ms1isotopes — Extract and evaluate MS1 isotope distributions from LC-MS/MS data.

A pure Python package for extracting isotope distributions from mzML files,
computing theoretical distributions via BRAIN, and scoring their similarity.
"""

from .isotopes import (
    sequence_to_composition,
    theoretical_isotope_distribution,
    spectral_angle,
    count_peaks_and_consecutive,
)
from .mass import peptide_mass, mass_to_mz, mz_to_isotope_mz
from .io import read_input_list, results_to_excel, results_to_json, results_to_tsv


def extract_ms1_isotope_distributions(*args, **kwargs):
    """Lazy wrapper — imports pyopenms only when extraction is actually called."""
    from .extraction import extract_ms1_isotope_distributions as _extract
    return _extract(*args, **kwargs)


__version__ = "0.1.0"

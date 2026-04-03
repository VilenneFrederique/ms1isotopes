"""
Isotope distribution computation and spectral angle scoring.

Replaces the R/BRAIN dependency using the brainpy package,
and handles all post-extraction feature computation (spectral angle,
peak counting, consecutiveness) that was previously split across
R and Python.
"""

import numpy as np
import brainpy
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Amino acid residue compositions: {C, H, N, O, S}
# ---------------------------------------------------------------------------
AMINO_ACID_COMPOSITION: Dict[str, Dict[str, int]] = {
    "A": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0},
    "R": {"C": 6, "H": 12, "N": 4, "O": 1, "S": 0},
    "N": {"C": 4, "H": 6, "N": 2, "O": 2, "S": 0},
    "D": {"C": 4, "H": 5, "N": 1, "O": 3, "S": 0},
    "C": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},
    "E": {"C": 5, "H": 7, "N": 1, "O": 3, "S": 0},
    "Q": {"C": 5, "H": 8, "N": 2, "O": 2, "S": 0},
    "G": {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0},
    "H": {"C": 6, "H": 7, "N": 3, "O": 1, "S": 0},
    "I": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "L": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},
    "K": {"C": 6, "H": 12, "N": 2, "O": 1, "S": 0},
    "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},
    "F": {"C": 9, "H": 9, "N": 1, "O": 1, "S": 0},
    "P": {"C": 5, "H": 7, "N": 1, "O": 1, "S": 0},
    "S": {"C": 3, "H": 5, "N": 1, "O": 2, "S": 0},
    "T": {"C": 4, "H": 7, "N": 1, "O": 2, "S": 0},
    "W": {"C": 11, "H": 10, "N": 2, "O": 1, "S": 0},
    "Y": {"C": 9, "H": 9, "N": 1, "O": 2, "S": 0},
    "V": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 0},
}

# Modification delta compositions
MODIFICATIONS = {
    "Carbamidomethyl": {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0},  # +57.0215 on C
    "Oxidation": {"C": 0, "H": 0, "N": 0, "O": 1, "S": 0},         # +15.9949 on M
    "Cysteinylation": {"C": 3, "H": 5, "N": 1, "O": 2, "S": 1},      # +119.0041 on C by iodoacetamide
    "Pyro_Glu_from_E": {"C": 0, "H": -2, "N": 0, "O": -1, "S": 0},   # -18.0106 on E
    "Pyro_Glu_from_Q": {"C": 0, "H": -3, "N": -1, "O": 0, "S": 0},   # -17.0265 on Q
    "Acetylation": {"C": 2, "H": 2, "N": 0, "O": 1, "S": 0},          # +42.0106 on N-term
}


def sequence_to_composition(
    sequence: str,
    charge_state: int = 0,
    n_carbamidomethyl: int = 0,
    n_oxidation: int = 0,
    n_cysteinylation: int = 0,
    n_pyro_glu_E: int = 0,
    n_pyro_glu_Q: int = 0,
    n_acetylation: int = 0
    ) -> Dict[str, int]:
    """
    Convert a peptide sequence to atomic composition.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
    charge_state : int
        Ion charge state (adds protons).
    n_carbamidomethyl : int
        Number of carbamidomethylated cysteines.
    n_oxidation : int
        Number of oxidised methionines.
    n_cysteinylation : int
        Number of cysteine modifications by iodoacetamide.
    n_pyro_glu_E : int
        Number of pyro-glutamate modifications on E.
    n_pyro_glu_Q : int
        Number of pyro-glutamate modifications on Q.
    n_acetylation : int
        Number of N-terminal acetylations.
    Returns
    -------
    dict
        {"C": int, "H": int, "N": int, "O": int, "S": int}
    """
    comp = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0}

    for aa in sequence:
        if aa not in AMINO_ACID_COMPOSITION:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        for el, count in AMINO_ACID_COMPOSITION[aa].items():
            comp[el] += count

    # Terminal water
    comp["H"] += 2
    comp["O"] += 1

    # Modifications
    for el, count in MODIFICATIONS["Carbamidomethyl"].items():
        comp[el] += count * n_carbamidomethyl
    for el, count in MODIFICATIONS["Oxidation"].items():
        comp[el] += count * n_oxidation
    for el, count in MODIFICATIONS["Cysteinylation"].items():
        comp[el] += count * n_cysteinylation
    for el, count in MODIFICATIONS["Pyro_Glu_from_E"].items():
        comp[el] += count * n_pyro_glu_E
    for el, count in MODIFICATIONS["Pyro_Glu_from_Q"].items():
        comp[el] += count * n_pyro_glu_Q
    for el, count in MODIFICATIONS["Acetylation"].items():
        comp[el] += count * n_acetylation
    # Protonation
    comp["H"] += charge_state

    return comp


def theoretical_isotope_distribution(
    composition: Dict[str, int],
    n_peaks: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical isotopic distribution using brainpy (BRAIN algorithm).

    Parameters
    ----------
    composition : dict
        Atomic composition.
    n_peaks : int
        Number of isotopic peaks.

    Returns
    -------
    tuple of (mz_array, intensity_array)
        Arrays of length n_peaks with m/z values and relative intensities.
    """
    dist = brainpy.IsotopicDistribution(composition, order=n_peaks)
    peaks = dist.aggregated_isotopic_variants()

    # brainpy may return more peaks than requested
    mz_values = np.array([p.mz for p in peaks[:n_peaks]])
    intensities = np.array([p.intensity for p in peaks[:n_peaks]])

    # Pad if fewer peaks returned
    if len(mz_values) < n_peaks:
        mz_values = np.pad(mz_values, (0, n_peaks - len(mz_values)))
        intensities = np.pad(intensities, (0, n_peaks - len(intensities)))

    return mz_values, intensities


def spectral_angle(
    theoretical: np.ndarray,
    experimental: np.ndarray,
) -> float:
    """
    Compute spectral angle between theoretical and experimental distributions.

    Parameters
    ----------
    theoretical : np.ndarray
        Theoretical isotope intensities.
    experimental : np.ndarray
        Experimental isotope intensities (NaN for missing peaks).

    Returns
    -------
    float
        Spectral angle in radians [0, π/2]. Lower = more similar.
    """
    valid = ~np.isnan(experimental)
    if valid.sum() < 2:
        return np.nan

    exp = experimental[valid]
    theo = theoretical[valid]

    exp_sum = np.sum(exp)
    if exp_sum <= 0:
        return np.nan
    exp_norm = exp / exp_sum

    num = np.dot(theo, exp_norm)
    denom = np.linalg.norm(theo) * np.linalg.norm(exp_norm)
    if denom == 0:
        return np.nan

    return float(np.arccos(np.clip(num / denom, -1.0, 1.0)))


def count_peaks_and_consecutive(intensities: np.ndarray) -> Tuple[int, bool]:
    """
    Count detected isotope peaks and check if they are consecutive.

    Replaces the 95-line R for-loop + if/else chain.

    Parameters
    ----------
    intensities : np.ndarray
        Array of isotope peak intensities (NaN = not detected).

    Returns
    -------
    tuple of (n_peaks, is_consecutive)
    """
    detected = ~np.isnan(intensities)
    n_peaks = int(detected.sum())

    if n_peaks <= 1:
        return n_peaks, False

    # Consecutive = first n_peaks positions are all detected
    is_consecutive = bool(np.all(detected[:n_peaks]))
    return n_peaks, is_consecutive

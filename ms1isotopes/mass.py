"""
Mass and m/z calculations for peptides.

Consolidates the two mass calculator functions (mass_calculator and
mass_calculator_simple) from the original script into a single,
clean interface.
"""

import numpy as np
from numba import njit
from typing import Optional

# Monoisotopic residue masses (Da)
MONOISOTOPIC_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}

# Average residue masses (Da)
AVERAGE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "E": 129.1155, "Q": 128.1307, "G": 57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}

WATER_MONO = 18.010565
WATER_AVG = 18.0153
PROTON = 1.007276
CAM_MASS = 57.021464   # Carbamidomethylation
OX_MASS = 15.994915    # Oxidation


def peptide_mass(
    sequence: str,
    mass_type: str = "monoisotopic",
    n_carbamidomethyl: int = 0,
    n_oxidation: int = 0,
) -> float:
    """
    Compute neutral peptide mass.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    mass_type : str
        "monoisotopic" or "average".
    n_carbamidomethyl : int
        Number of CAM-modified cysteines.
    n_oxidation : int
        Number of oxidised methionines.

    Returns
    -------
    float
        Neutral mass in Dalton.
    """
    mass_dict = MONOISOTOPIC_MASS if mass_type == "monoisotopic" else AVERAGE_MASS
    water = WATER_MONO if mass_type == "monoisotopic" else WATER_AVG

    mass = water
    for aa in sequence:
        if aa not in mass_dict:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        mass += mass_dict[aa]

    mass += CAM_MASS * n_carbamidomethyl
    mass += OX_MASS * n_oxidation

    return mass


@njit
def mass_to_mz(mass: float, charge: int, isotope: int = 0) -> float:
    """Compute m/z from neutral mass, charge, and isotope offset."""
    return (mass + isotope * 1.003355 + charge * 1.007276) / charge


@njit
def mz_to_isotope_mz(mz_observed: float, isotope: int, charge: int) -> float:
    """Compute isotope peak m/z from an observed monoisotopic m/z."""
    return mz_observed + isotope * 1.003355 / charge

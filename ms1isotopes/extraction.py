"""
MS1 isotope distribution extraction from mzML files.

Core pipeline: mzML + input list → XIC → isotope peak picking →
theoretical comparison → annotated output.

Uses pyteomics for mzML parsing (pure Python, no compiled binaries).
"""

import numpy as np
import pandas as pd
from numba import njit
from pyteomics import mzml
from typing import Dict, List, Optional, Tuple
import re

from .mass import peptide_mass, mass_to_mz, mz_to_isotope_mz
from .isotopes import (
    sequence_to_composition,
    theoretical_isotope_distribution,
    spectral_angle,
    count_peaks_and_consecutive,
)


# ---------------------------------------------------------------------------
# Spectrum loading and indexing
# ---------------------------------------------------------------------------

def _get_rt(spectrum: dict) -> float:
    """Extract retention time (in seconds) from a pyteomics spectrum dict."""
    try:
        scan_info = spectrum["scanList"]["scan"][0]
        rt = scan_info["scan start time"]
        # pyteomics returns a unitfloat; handle minutes vs seconds
        if hasattr(rt, "unit_info") and rt.unit_info == "minute":
            return float(rt) * 60.0
        return float(rt)
    except (KeyError, IndexError):
        return 0.0


def _get_tic(spectrum: dict) -> float:
    """Extract total ion current from a pyteomics spectrum dict."""
    return float(spectrum.get("total ion current", 0.0))


def _get_peaks(spectrum: dict) -> np.ndarray:
    """
    Extract peaks as (N, 2) array of [m/z, intensity].

    Returns an empty (0, 2) array if no peaks are present.
    """
    mz_array = spectrum.get("m/z array")
    int_array = spectrum.get("intensity array")

    if mz_array is None or int_array is None or len(mz_array) == 0:
        return np.empty((0, 2), dtype=np.float64)

    return np.column_stack((
        np.asarray(mz_array, dtype=np.float64),
        np.asarray(int_array, dtype=np.float64),
    ))


def load_ms1_spectra(mzml_path: str, verbose: bool = True) -> List[dict]:
    """
    Load all MS1 spectra from an mzML file into memory.

    Each spectrum is stored as a lightweight dict with pre-extracted
    fields for fast access during XIC construction and peak picking.

    Parameters
    ----------
    mzml_path : str
        Path to the mzML file.
    verbose : bool
        Print loading progress.

    Returns
    -------
    list of dict
        Each dict has keys: id, rt, tic, peaks (N×2 ndarray).
    """
    ms1_spectra = []

    with mzml.MzML(mzml_path) as reader:
        for spectrum in reader:
            if spectrum.get("ms level", 0) != 1:
                continue

            ms1_spectra.append({
                "id": spectrum["id"],
                "rt": _get_rt(spectrum),
                "tic": _get_tic(spectrum),
                "peaks": _get_peaks(spectrum),
            })

    if verbose:
        print(f"Loaded {len(ms1_spectra)} MS1 spectra from {mzml_path}")

    return ms1_spectra


def build_spectrum_index(spectra: List[dict]) -> Dict[str, int]:
    """Build a dict mapping spectrum IDs to list indices for O(1) lookup."""
    return {s["id"]: i for i, s in enumerate(spectra)}


# ---------------------------------------------------------------------------
# Isotope peak extraction (numba-accelerated)
# ---------------------------------------------------------------------------

@njit
def extract_isotope_peaks(
    peaks: np.ndarray,
    mz: float,
    charge: int,
    tolerance_ppm: float,
    n_isotopes: int = 6,
) -> np.ndarray:
    """
    Extract isotope peak m/z and intensities from a peak list.

    Parameters
    ----------
    peaks : np.ndarray
        Shape (N, 2) array of [m/z, intensity] pairs.
    mz : float
        Monoisotopic m/z to search from.
    charge : int
        Charge state.
    tolerance_ppm : float
        Mass tolerance in ppm.
    n_isotopes : int
        Number of isotope peaks to extract (default 6: M+0 to M+5).

    Returns
    -------
    np.ndarray
        Shape (n_isotopes, 2) array. Each row is [m/z, intensity].
        NaN for peaks not found.
    """
    out = np.empty((n_isotopes, 2))

    for i in range(n_isotopes):
        expected_mz = mz + i * 1.003355 / charge

        tol = expected_mz / (1_000_000 / tolerance_ppm)
        lower = expected_mz - tol
        upper = expected_mz + tol

        mask = (peaks[:, 0] >= lower) & (peaks[:, 0] <= upper)
        subset = peaks[mask]

        if subset.shape[0] == 0:
            out[i, 0] = np.nan
            out[i, 1] = np.nan
        else:
            idx = np.argmax(subset[:, 1])
            out[i, 0] = subset[idx, 0]
            out[i, 1] = subset[idx, 1]

    return out


# ---------------------------------------------------------------------------
# XIC construction
# ---------------------------------------------------------------------------

def extract_ion_chromatogram(
    spectra: List[dict],
    mz: float,
    tolerance_ppm: float,
) -> pd.DataFrame:
    """
    Build an extracted ion chromatogram (XIC) for a target m/z.

    Parameters
    ----------
    spectra : list of dict
        Pre-loaded MS1 spectra from load_ms1_spectra().
    mz : float
        Target m/z value.
    tolerance_ppm : float
        Mass tolerance in ppm.

    Returns
    -------
    pd.DataFrame
        Columns: SpectraID, RetentionTime, Intensity.
    """
    tol_window = mz / (1_000_000 / tolerance_ppm)
    lower = mz - tol_window
    upper = mz + tol_window

    records = []
    for spec in spectra:
        peaks = spec["peaks"]
        if peaks.shape[0] == 0:
            continue

        in_window = peaks[(peaks[:, 0] >= lower) & (peaks[:, 0] <= upper)]
        if in_window.shape[0] > 0:
            max_intensity = float(np.max(in_window[:, 1]))
            if max_intensity > 0:
                records.append({
                    "SpectraID": spec["id"],
                    "RetentionTime": spec["rt"],
                    "Intensity": max_intensity,
                })

    if not records:
        return pd.DataFrame(columns=["SpectraID", "RetentionTime", "Intensity"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Peptide_Sequence", "Charge_State", "Modifications",
    "RetentionTime", "RetentionTimeWindowBefore", "RetentionTimeWindowAfter",
]


def parse_modifications(mod_string: str) -> Tuple[int, int, int, int, int, int]:
    """
    Parse modification string to count CAM cysteines and oxidised methionines.

    Handles formats like:
    - "C(57.0214)" or "2C(57.0214)M(15.9949)"
    - NaN / None / "None" → (0, 0, 0, 0, 0)
    """
    if pd.isna(mod_string) or mod_string == "None" or mod_string is None:
        return 0, 0, 0, 0, 0, 0

    mod_str = str(mod_string)
    n_cam = len(re.findall(r"C\(57\.0214\)", mod_str))
    n_ox = len(re.findall(r"M\(15\.9949\)", mod_str))
    n_cysteinylation = len(re.findall(r"C\(119\.0041\)", mod_str))
    n_pyro_glu_E = len(re.findall(r"E\(-18\.0106\)", mod_str))
    n_pyro_glu_Q = len(re.findall(r"Q\(-17\.0265\)", mod_str))
    n_acetylation = len(re.findall(r"Acetylation\(42\.0106\)", mod_str))
    return n_cam, n_ox, n_cysteinylation, n_pyro_glu_E, n_pyro_glu_Q, n_acetylation


def validate_input(df: pd.DataFrame) -> None:
    """Validate that required columns exist in the input list."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_ms1_isotope_distributions(
    mzml_path: str,
    input_list: pd.DataFrame,
    tolerance_ppm: float = 5.0,
    n_isotopes: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract MS1 isotope distributions for all entries in the input list.

    Parameters
    ----------
    mzml_path : str
        Path to the mzML file.
    input_list : pd.DataFrame
        Input list with columns: Peptide_Sequence, Charge_State,
        Modifications, RetentionTime, RetentionTimeWindowBefore,
        RetentionTimeWindowAfter. Optional: MZ, ErrorTolerance.
    tolerance_ppm : float
        Default mass tolerance in ppm (overridden by ErrorTolerance column).
    n_isotopes : int
        Number of isotope peaks to extract.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Full annotated results including experimental and theoretical
        isotope distributions, spectral angle, and quality metrics.
    """
    validate_input(input_list)

    # Load and index MS1 spectra
    spectra = load_ms1_spectra(mzml_path, verbose=verbose)
    spec_index = build_spectrum_index(spectra)

    if verbose:
        print(f"Processing {len(input_list)} entries from input list")

    all_results = []

    for idx in range(len(input_list)):
        row = input_list.iloc[idx]

        # --- Extract metadata ---
        sequence = row["Peptide_Sequence"]
        charge = int(row["Charge_State"])
        modifications = row["Modifications"]
        rt = float(row["RetentionTime"])
        rt_before = float(row["RetentionTimeWindowBefore"])
        rt_after = float(row["RetentionTimeWindowAfter"])

        n_cam, n_ox, n_cysteinylation, n_pyro_glu_E, n_pyro_glu_Q, n_acetylation = parse_modifications(modifications)

        # --- Compute masses ---
        mono_mass = peptide_mass(sequence, "monoisotopic", n_cam, n_ox, n_cysteinylation, n_pyro_glu_E, n_pyro_glu_Q, n_acetylation)
        avg_mass = peptide_mass(sequence, "average", n_cam, n_ox, n_cysteinylation, n_pyro_glu_E, n_pyro_glu_Q, n_acetylation)
        theo_mz = mass_to_mz(mono_mass, charge, isotope=0)

        # Use observed m/z if provided, else theoretical
        ppm = tolerance_ppm
        if "ErrorTolerance" in input_list.columns and pd.notna(row.get("ErrorTolerance")):
            ppm = float(row["ErrorTolerance"])

        if "MZ" in input_list.columns and pd.notna(row.get("MZ")):
            obs_mz = float(row["MZ"])
        else:
            obs_mz = theo_mz

        # --- Build XIC ---
        xic = extract_ion_chromatogram(spectra, obs_mz, ppm)
        xic = xic[xic["RetentionTime"].between(rt - rt_before, rt + rt_after)]

        if len(xic) == 0:
            continue

        # --- Theoretical isotope distribution (via brainpy) ---
        comp = sequence_to_composition(sequence, charge, n_cam, n_ox, n_cysteinylation, n_pyro_glu_E, n_pyro_glu_Q, n_acetylation)
        theo_mzs, theo_intensities = theoretical_isotope_distribution(comp, n_isotopes)

        # --- Extract isotopes from each spectrum in the XIC ---
        for _, xic_row in xic.iterrows():
            spectrum_id = xic_row["SpectraID"]

            if spectrum_id not in spec_index:
                continue

            spec = spectra[spec_index[spectrum_id]]
            peaks = spec["peaks"]

            if peaks.shape[0] == 0:
                continue

            # Extract isotope peaks
            iso = extract_isotope_peaks(peaks, obs_mz, charge, ppm, n_isotopes)

            # Compute quality metrics
            exp_intensities = iso[:, 1]
            n_peaks, consecutive = count_peaks_and_consecutive(exp_intensities)
            has_dist = n_peaks >= 2

            if has_dist:
                sa = spectral_angle(theo_intensities, exp_intensities)
            else:
                sa = np.nan

            # Build result row
            result = {
                "RawFile": mzml_path,
                "Spectrum": spectrum_id,
                "PeptideSequence": sequence,
                "Modifications": modifications if pd.notna(modifications) else "None",
                "PeptideLength": len(sequence),
                "TheoreticalMonoIsotopicMass": mono_mass,
                "TheoreticalAverageMass": avg_mass,
                "ChargeState": charge,
                "TheoreticalMZ": theo_mz,
                "ObservedMZ": obs_mz,
                "RetentionTime": spec["rt"],
                "TIC": spec["tic"],
            }

            # Isotope peak m/z and intensities
            for i in range(n_isotopes):
                result[f"IsotopePeak{i+1}MZ"] = iso[i, 0]
                result[f"IsotopePeak{i+1}Intensity"] = iso[i, 1]

            # Atomic composition
            result["Carbons"] = comp["C"]
            result["Hydrogens"] = comp["H"]
            result["Nitrogens"] = comp["N"]
            result["Oxygens"] = comp["O"]
            result["Sulphurs"] = comp["S"]

            # BRAIN theoretical intensities
            for i in range(n_isotopes):
                result[f"BRAINRelativeIsotopePeak{i+1}Intensity"] = theo_intensities[i]

            # Quality metrics
            result["Distribution"] = has_dist
            result["SpectralAngle"] = sa
            result["NPeaks"] = n_peaks
            result["ConsecutivePeaks"] = consecutive

            all_results.append(result)

        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(input_list)} entries")

    if verbose:
        print(f"Done. Extracted {len(all_results)} isotope distributions.")

    return pd.DataFrame(all_results)

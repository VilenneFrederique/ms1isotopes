"""
Input/output handling for MS1 isotope distribution data.

Supports reading input lists (Excel, TSV, CSV) and writing results
to Excel and hierarchical JSON format.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def read_input_list(path: str) -> pd.DataFrame:
    """
    Read input list from Excel, TSV, or CSV.

    Auto-detects format from extension.
    """
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def results_to_excel(results: pd.DataFrame, output_path: str) -> None:
    """Write results DataFrame to Excel."""
    results.to_excel(output_path, index=False)


def results_to_json(results: pd.DataFrame, output_path: str, n_isotopes: int = 6) -> None:
    """
    Convert results DataFrame to hierarchical JSON.

    Structure: Peptides → Modifications → Charge States → Spectra
    """
    peptides_dict = {}

    for peptide in results["PeptideSequence"].unique():
        df_pep = results[results["PeptideSequence"] == peptide]
        pep_length = int(df_pep["PeptideLength"].iloc[0])

        modifications_dict = {}
        for mod in df_pep["Modifications"].unique():
            df_mod = df_pep[df_pep["Modifications"] == mod]

            charge_states_dict = {}
            for cs in df_mod["ChargeState"].unique():
                df_cs = df_mod[df_mod["ChargeState"] == cs]
                row0 = df_cs.iloc[0]

                # Ion metadata (shared across all spectra for this peptide/mod/charge)
                ion_meta = {
                    "TheoreticalMonoIsotopicMass": row0["TheoreticalMonoIsotopicMass"],
                    "TheoreticalAverageMass": row0["TheoreticalAverageMass"],
                    "TheoreticalMZ": row0["TheoreticalMZ"],
                    "Carbons": int(row0["Carbons"]),
                    "Hydrogens": int(row0["Hydrogens"]),
                    "Oxygens": int(row0["Oxygens"]),
                    "Nitrogens": int(row0["Nitrogens"]),
                    "Sulphurs": int(row0["Sulphurs"]),
                    "BRAINDistribution": [
                        row0[f"BRAINRelativeIsotopePeak{i+1}Intensity"]
                        for i in range(n_isotopes)
                    ],
                }

                # Per-spectrum data as a list (easier to iterate/batch in ML pipelines)
                spectra_list = []
                for _, spec_row in df_cs.iterrows():
                    spectra_list.append({
                        "id": spec_row["Spectrum"],
                        "ObservedMZ": spec_row["ObservedMZ"],
                        "RetentionTime": spec_row["RetentionTime"],
                        "TIC": spec_row["TIC"],
                        "Distribution": spec_row["Distribution"],
                        "NPeaks": int(spec_row["NPeaks"]),
                        "ConsecutivePeaks": spec_row["ConsecutivePeaks"],
                        "SpectralAngle": spec_row["SpectralAngle"],
                        "IsotopeDistribution": {
                            "MZ": [
                                spec_row[f"IsotopePeak{i+1}MZ"]
                                for i in range(n_isotopes)
                            ],
                            "Intensities": [
                                spec_row[f"IsotopePeak{i+1}Intensity"]
                                for i in range(n_isotopes)
                            ],
                        },
                    })

                charge_states_dict[int(cs)] = {
                    "IonMetadata": ion_meta,
                    "Spectra": spectra_list,
                }

            modifications_dict[str(mod)] = {"ChargeStates": charge_states_dict}

        peptides_dict[peptide] = {
            "PeptideMetadata": {"PeptideLength": pep_length},
            "Modifications": modifications_dict,
        }

    output = {"Peptides": peptides_dict}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


def results_to_tsv(results: pd.DataFrame, output_path: str) -> None:
    """Write results to TSV (tab-separated values)."""
    results.to_csv(output_path, sep="\t", index=False)

"""
Test suite for ms1isotopes.

Covers isotope distribution computation, mass calculation, spectral angle,
peak counting, and I/O serialisation. Extraction tests require pyopenms
and are skipped if unavailable.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ms1isotopes.isotopes import (
    sequence_to_composition,
    theoretical_isotope_distribution,
    spectral_angle,
    count_peaks_and_consecutive,
)
from ms1isotopes.mass import peptide_mass, mass_to_mz, mz_to_isotope_mz
from ms1isotopes.io import NumpyEncoder, results_to_json


# ===== Composition tests =====

class TestComposition:
    def test_glycine(self):
        comp = sequence_to_composition("G")
        assert comp == {"C": 2, "H": 5, "N": 1, "O": 2, "S": 0}

    def test_dipeptide(self):
        comp = sequence_to_composition("AG")
        assert comp == {"C": 5, "H": 10, "N": 2, "O": 3, "S": 0}

    def test_charge_adds_protons(self):
        comp0 = sequence_to_composition("AG", charge_state=0)
        comp2 = sequence_to_composition("AG", charge_state=2)
        assert comp2["H"] == comp0["H"] + 2

    def test_carbamidomethylation(self):
        plain = sequence_to_composition("C")
        cam = sequence_to_composition("C", n_carbamidomethyl=1)
        assert cam["C"] == plain["C"] + 2
        assert cam["H"] == plain["H"] + 3
        assert cam["N"] == plain["N"] + 1
        assert cam["O"] == plain["O"] + 1

    def test_oxidation(self):
        plain = sequence_to_composition("M")
        ox = sequence_to_composition("M", n_oxidation=1)
        assert ox["O"] == plain["O"] + 1

    def test_angiotensin_ii(self):
        """Angiotensin II: DRVYIHPF → C50H71N13O12 (neutral)."""
        comp = sequence_to_composition("DRVYIHPF", charge_state=0)
        assert comp["C"] == 50
        assert comp["H"] == 71
        assert comp["N"] == 13
        assert comp["O"] == 12
        assert comp["S"] == 0

    def test_unknown_amino_acid_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            sequence_to_composition("PEPTXDE")


# ===== Isotope distribution tests =====

class TestIsotopicDistribution:
    def test_small_peptide_mono_dominant(self):
        comp = sequence_to_composition("GGG", charge_state=2)
        _, dist = theoretical_isotope_distribution(comp, n_peaks=6)
        assert dist[0] > dist[1], "M+0 should be most abundant for small peptide"

    def test_large_peptide_mono_not_dominant(self):
        """For peptides >~2000 Da, M+1 overtakes M+0."""
        comp = sequence_to_composition("A" * 30, charge_state=2)
        _, dist = theoretical_isotope_distribution(comp, n_peaks=8)
        assert np.argmax(dist) > 0, "M+1 should overtake M+0 for large peptides"

    def test_distribution_sums_to_near_one(self):
        comp = sequence_to_composition("PEPTIDE", charge_state=2)
        _, dist = theoretical_isotope_distribution(comp, n_peaks=10)
        assert 0.99 < np.sum(dist) <= 1.0 + 1e-6

    def test_all_positive(self):
        comp = sequence_to_composition("YLGYLEQLLR", charge_state=2)
        _, dist = theoretical_isotope_distribution(comp, n_peaks=6)
        assert np.all(dist >= 0)

    def test_sulfur_enriches_m2(self):
        """Sulfur-34 (4.25%) causes elevated M+2 relative to M+0."""
        comp_s = sequence_to_composition("MMMMM", charge_state=2)
        comp_no_s = sequence_to_composition("LLLLL", charge_state=2)
        _, dist_s = theoretical_isotope_distribution(comp_s)
        _, dist_no_s = theoretical_isotope_distribution(comp_no_s)

        ratio_s = dist_s[2] / dist_s[0]
        ratio_no_s = dist_no_s[2] / dist_no_s[0]
        assert ratio_s > ratio_no_s

    def test_returns_mz_and_intensities(self):
        comp = sequence_to_composition("PEPTIDE", charge_state=2)
        mzs, intensities = theoretical_isotope_distribution(comp, n_peaks=6)
        assert len(mzs) == 6
        assert len(intensities) == 6
        assert mzs[0] > 0  # Should have real m/z values


# ===== Spectral angle tests =====

class TestSpectralAngle:
    def test_identical_gives_zero(self):
        a = np.array([0.6, 0.3, 0.1])
        assert abs(spectral_angle(a, a)) < 1e-6

    def test_orthogonal_gives_pi_over_2(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(spectral_angle(a, b) - np.pi / 2) < 1e-6

    def test_scale_invariant(self):
        a = np.array([0.6, 0.3, 0.1])
        b = np.array([6000.0, 3000.0, 1000.0])
        assert abs(spectral_angle(a, b)) < 1e-4

    def test_nan_handling(self):
        a = np.array([0.5, 0.3, 0.15, 0.05])
        b = np.array([5000.0, 2800.0, np.nan, np.nan])
        sa = spectral_angle(a, b)
        assert not np.isnan(sa)
        assert 0 <= sa <= np.pi / 2

    def test_all_nan_returns_nan(self):
        a = np.array([0.5, 0.3])
        b = np.array([np.nan, np.nan])
        assert np.isnan(spectral_angle(a, b))

    def test_single_peak_returns_nan(self):
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([100.0, np.nan, np.nan])
        assert np.isnan(spectral_angle(a, b))

    def test_range_zero_to_pi_half(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.random(4)
            b = rng.random(4) * 10000
            sa = spectral_angle(a, b)
            assert 0 <= sa <= np.pi / 2 + 1e-10


# ===== Peak counting tests =====

class TestPeakCounting:
    def test_all_detected(self):
        intensities = np.array([100.0, 50.0, 25.0, 10.0, 5.0, 1.0])
        n, consec = count_peaks_and_consecutive(intensities)
        assert n == 6
        assert consec is True

    def test_none_detected(self):
        intensities = np.array([np.nan] * 6)
        n, consec = count_peaks_and_consecutive(intensities)
        assert n == 0
        assert consec is False

    def test_single_peak(self):
        intensities = np.array([100.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        n, consec = count_peaks_and_consecutive(intensities)
        assert n == 1
        assert consec is False

    def test_gap_not_consecutive(self):
        intensities = np.array([100.0, np.nan, 25.0, np.nan, np.nan, np.nan])
        n, consec = count_peaks_and_consecutive(intensities)
        assert n == 2
        assert consec is False

    def test_three_consecutive(self):
        intensities = np.array([100.0, 50.0, 25.0, np.nan, np.nan, np.nan])
        n, consec = count_peaks_and_consecutive(intensities)
        assert n == 3
        assert consec is True


# ===== Mass calculation tests =====

class TestMass:
    def test_glycine_monoisotopic(self):
        mass = peptide_mass("G", "monoisotopic")
        assert abs(mass - 75.032) < 0.001

    def test_angiotensin_ii_mass(self):
        mass = peptide_mass("DRVYIHPF", "monoisotopic")
        assert abs(mass - 1045.535) < 0.01

    def test_average_heavier_than_mono(self):
        mono = peptide_mass("PEPTIDE", "monoisotopic")
        avg = peptide_mass("PEPTIDE", "average")
        assert avg > mono

    def test_cam_increases_mass(self):
        plain = peptide_mass("C", "monoisotopic")
        cam = peptide_mass("C", "monoisotopic", n_carbamidomethyl=1)
        assert abs(cam - plain - 57.021) < 0.001

    def test_mz_calculation(self):
        mass = peptide_mass("DRVYIHPF", "monoisotopic")
        mz = mass_to_mz(mass, charge=2, isotope=0)
        expected = (mass + 2 * 1.007276) / 2
        assert abs(mz - expected) < 0.001

    def test_isotope_mz_spacing(self):
        mz0 = mz_to_isotope_mz(500.0, isotope=0, charge=2)
        mz1 = mz_to_isotope_mz(500.0, isotope=1, charge=2)
        # Spacing should be ~1.003355 / charge = ~0.5017
        assert abs((mz1 - mz0) - 1.003355 / 2) < 0.001


# ===== I/O tests =====

class TestIO:
    def test_numpy_encoder_nan(self):
        result = json.dumps({"val": np.nan}, cls=NumpyEncoder)
        assert '"val": null' in result or '"val": NaN' in result

    def test_numpy_encoder_bool(self):
        result = json.dumps({"val": np.bool_(True)}, cls=NumpyEncoder)
        assert '"val": true' in result

    def test_numpy_encoder_int(self):
        result = json.dumps({"val": np.int64(42)}, cls=NumpyEncoder)
        assert '"val": 42' in result

    def test_results_to_json(self, tmp_path):
        """Test JSON output with minimal synthetic data."""
        df = pd.DataFrame([{
            "PeptideSequence": "PEPTIDE",
            "Modifications": "None",
            "PeptideLength": 7,
            "ChargeState": 2,
            "Spectrum": "scan=100",
            "TheoreticalMonoIsotopicMass": 799.36,
            "TheoreticalAverageMass": 799.85,
            "TheoreticalMZ": 400.69,
            "ObservedMZ": 400.69,
            "RetentionTime": 1200.0,
            "TIC": 1e6,
            "Carbons": 34, "Hydrogens": 53, "Nitrogens": 7,
            "Oxygens": 15, "Sulphurs": 0,
            "Distribution": True,
            "SpectralAngle": 0.08,
            "NPeaks": 4,
            "ConsecutivePeaks": True,
            **{f"IsotopePeak{i}MZ": 400.69 + i * 0.5 for i in range(1, 7)},
            **{f"IsotopePeak{i}Intensity": 1000.0 / i for i in range(1, 7)},
            **{f"BRAINRelativeIsotopePeak{i}Intensity": 0.5 / i for i in range(1, 7)},
        }])

        out_path = str(tmp_path / "test.json")
        results_to_json(df, out_path)

        with open(out_path) as f:
            data = json.load(f)

        assert "Peptides" in data
        assert "PEPTIDE" in data["Peptides"]
        pep = data["Peptides"]["PEPTIDE"]
        assert pep["PeptideMetadata"]["PeptideLength"] == 7

        # Check spectra is a list with id field
        spectra = pep["Modifications"]["None"]["ChargeStates"]["2"]["Spectra"]
        assert isinstance(spectra, list)
        assert spectra[0]["id"] == "scan=100"

        # Check isotope distributions are lists
        assert isinstance(spectra[0]["IsotopeDistribution"]["MZ"], list)
        assert isinstance(spectra[0]["IsotopeDistribution"]["Intensities"], list)
        assert len(spectra[0]["IsotopeDistribution"]["MZ"]) == 6

        # Check BRAIN distribution is a list
        brain = pep["Modifications"]["None"]["ChargeStates"]["2"]["IonMetadata"]["BRAINDistribution"]
        assert isinstance(brain, list)
        assert len(brain) == 6

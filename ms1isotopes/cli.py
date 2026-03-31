"""
Command-line interface for ms1isotopes.

Usage:
    ms1isotopes extract --mzml FILE --input FILE --output PREFIX [OPTIONS]
    ms1isotopes extract -m file.mzML -i psms.xlsx -o results

Options:
    --tolerance     Mass tolerance in ppm (default: 5)
    --n-isotopes    Number of isotope peaks (default: 6)
    --format        Output format: excel, json, tsv, all (default: all)
"""

import argparse
import sys
from pathlib import Path

from .io import read_input_list, results_to_excel, results_to_json, results_to_tsv
from .extraction import extract_ms1_isotope_distributions


def main():
    parser = argparse.ArgumentParser(
        prog="ms1isotopes",
        description="Extract MS1 isotope distributions from LC-MS/MS data.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- extract command ---
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract isotope distributions from mzML + input list",
    )
    extract_parser.add_argument(
        "-m", "--mzml", required=True,
        help="Path to the mzML file (MS1 spectra)",
    )
    extract_parser.add_argument(
        "-i", "--input", required=True, dest="input_list",
        help="Path to input list (Excel, TSV, or CSV)",
    )
    extract_parser.add_argument(
        "-o", "--output", required=True,
        help="Output file prefix (extensions added automatically)",
    )
    extract_parser.add_argument(
        "--tolerance", type=float, default=5.0,
        help="Mass tolerance in ppm (default: 5)",
    )
    extract_parser.add_argument(
        "--n-isotopes", type=int, default=6,
        help="Number of isotope peaks to extract (default: 6)",
    )
    extract_parser.add_argument(
        "--format", choices=["excel", "json", "tsv", "all"], default="all",
        help="Output format (default: all)",
    )
    extract_parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        run_extract(args)


def run_extract(args):
    """Execute the extraction pipeline."""
    input_list = read_input_list(args.input_list)
    print(f"Read {len(input_list)} entries from {args.input_list}")

    results = extract_ms1_isotope_distributions(
        mzml_path=args.mzml,
        input_list=input_list,
        tolerance_ppm=args.tolerance,
        n_isotopes=args.n_isotopes,
        verbose=not args.quiet,
    )

    output = Path(args.output)
    fmt = args.format

    if fmt in ("excel", "all"):
        out_path = str(output.with_suffix(".xlsx"))
        results_to_excel(results, out_path)
        print(f"Written: {out_path}")

    if fmt in ("json", "all"):
        out_path = str(output.with_suffix(".json"))
        results_to_json(results, out_path, n_isotopes=args.n_isotopes)
        print(f"Written: {out_path}")

    if fmt in ("tsv", "all"):
        out_path = str(output.with_suffix(".tsv"))
        results_to_tsv(results, out_path)
        print(f"Written: {out_path}")

    print(f"\nExtracted {len(results)} isotope distributions "
          f"for {results['PeptideSequence'].nunique()} unique peptides.")


if __name__ == "__main__":
    main()

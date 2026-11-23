"""Run the complete pipeline on every bundled demo file without the web app.

    python scripts/run_demo_cli.py                 # demo ML model
    python scripts/run_demo_cli.py --mode rule_based
    python scripts/run_demo_cli.py --out reports   # also write JSON/CSV/PDF reports

Useful for a quick check that everything works, and as a fallback
demonstration if the browser-based app cannot be opened.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEMO_DATA_DIR, load_advisory_rules, load_settings  # noqa: E402
from src.data.ingestion import read_csv_table, read_raster  # noqa: E402
from src.data.preprocessing import sample_from_band_files, sample_from_raster, samples_from_table  # noqa: E402
from src.data.synthetic import BANDS  # noqa: E402
from src.data.validation import ValidationError  # noqa: E402
from src.models.inference import build_classifier  # noqa: E402
from src.pipeline import analyze_sample  # noqa: E402
from src.reporting.report_generator import build_report, report_to_csv, report_to_json, report_to_pdf  # noqa: E402

DEMO_MAPPING = {"blue": 1, "green": 2, "red": 3, "red_edge": 4, "nir": 5}


def load_demo_samples(settings: dict) -> list:
    samples = []
    for path in sorted(DEMO_DATA_DIR.glob("*.tif")):
        scale = 10000 if "x10000" in path.name else 1.0
        samples.append(sample_from_raster(read_raster(path), DEMO_MAPPING, settings, scale, path.stem))
    band_files = {band: read_raster(DEMO_DATA_DIR / "bands" / f"water_stress_{band}.tif") for band in BANDS}
    samples.append(sample_from_band_files(band_files, settings))
    for name in ("demo_points.csv", "demo_points_no_red_edge.csv"):
        frame, source = read_csv_table(DEMO_DATA_DIR / name)
        samples.extend(samples_from_table(frame, source, settings))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", default="ml_demo", choices=["ml_demo", "cnn_deep_learning", "rule_based"]
    )
    parser.add_argument("--out", type=Path, help="folder for JSON/CSV/PDF reports")
    args = parser.parse_args()

    settings = load_settings()
    rules = load_advisory_rules()
    classifier, notices = build_classifier(args.mode, settings)
    for notice in notices:
        print("NOTICE:", notice)
    print(f"Classifier: {classifier.name} [{classifier.mode}, {classifier.version}]\n")

    results = []
    for sample in load_demo_samples(settings):
        result = analyze_sample(sample, classifier, settings, rules, sample.metadata.get("crop", "generic"))
        results.append(result)
        p = result.prediction
        confidence = f"{p.confidence:.2f}" if p.confidence is not None else "  - "
        print(f"{sample.source_name[:38]:38s} {sample.sample_id[:24]:24s} -> {p.category:16s} "
              f"(raw: {str(p.raw_category):16s} conf: {confidence}) advisories: "
              f"{', '.join(a.rule_id for a in result.advisories)}")
        for reason in p.reasons:
            print(f"{'':40s}reason: {reason}")

    print("\nValidation of deliberately broken inputs:")
    for path in sorted((DEMO_DATA_DIR / "invalid").iterdir()):
        try:
            if path.suffix == ".csv":
                frame, source = read_csv_table(path)
                samples_from_table(frame, source, settings)
            else:
                read_raster(path)
            print(f"  {path.name}: accepted (unexpected!)")
        except ValidationError as error:
            print(f"  {path.name}: rejected -> {error}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        report = build_report(results, classifier.info(), notices, settings)
        stem = args.out / report["report_id"]
        Path(f"{stem}.json").write_text(report_to_json(report), encoding="utf-8")
        Path(f"{stem}.csv").write_text(report_to_csv(report), encoding="utf-8")
        Path(f"{stem}.pdf").write_bytes(report_to_pdf(report))
        print(f"\nReports written: {stem}.json / .csv / .pdf")


if __name__ == "__main__":
    main()

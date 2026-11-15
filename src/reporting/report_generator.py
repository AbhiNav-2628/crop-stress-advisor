"""Building downloadable reports (JSON, CSV and PDF) from analysis results."""

import io
import json
import math
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import __version__
from src.features.feature_engineering import FEATURE_ORDER

DISCLAIMER = (
    "Decision-support output from a demonstration system. The active classifier was fitted to "
    "or tuned on synthetic data and has not been validated in the field. Results are not an "
    "agronomic diagnosis; confirm every finding by field inspection and consult a qualified "
    "agricultural expert before acting. No pesticide, fertiliser or treatment is prescribed."
)


def _clean(value):
    """Make a value JSON-safe: NaN/inf -> None, NumPy scalars -> Python types."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def new_report_id() -> str:
    return f"CSR-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:8]}"


def _sample_section(result) -> dict:
    sample = result.sample
    metadata = {k: v for k, v in sample.metadata.items() if k not in ("transform",)}
    explanation = {
        key: result.explanation[key]
        for key in ("method", "headline", "contributions", "rule_trace",
                    "inconclusive_reasons", "uncertainty_notes", "ndvi_reading", "limitations")
    }
    return {
        "sample_id": sample.sample_id,
        "input": {
            "source": sample.source_name,
            "kind": sample.kind,
            "shape": list(sample.shape),
            "bands_provided": list(sample.bands.keys()),
            "n_observations": sample.n_total,
            "n_valid": sample.n_valid,
            "valid_fraction": sample.valid_fraction,
            "crop": result.crop,
            "metadata": metadata,
        },
        "vegetation_indices": result.index_summary.to_dict(orient="records"),
        "indices_not_calculated": result.skipped_indices,
        "features": result.features.values,
        "prediction": result.prediction.to_dict(),
        "explanation": explanation,
        "data_quality_warnings": result.warnings,
        "advisories": [a.to_dict() for a in result.advisories],
        "analysed_at": result.analysed_at,
    }


def build_report(results: list, classifier_info: dict, notices: list[str] | None = None,
                 settings: dict | None = None) -> dict:
    project = (settings or {}).get("project", {})
    report = {
        "report_id": new_report_id(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": {
            "name": project.get("name", "Crop-Stress Diagnosis and Advisory System"),
            "problem_id": project.get("problem_id", "SIH25099"),
            "app_version": __version__,
        },
        "classifier": classifier_info,
        "notices": notices or [],
        "disclaimer": DISCLAIMER,
        "samples": [_sample_section(r) for r in results],
    }
    return _clean(report)


def report_to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


def report_to_csv(report: dict) -> str:
    """One row per sample with the main outputs flattened into columns."""
    rows = []
    for sample in report["samples"]:
        prediction = sample["prediction"]
        row = {
            "report_id": report["report_id"],
            "generated_at": report["generated_at"],
            "sample_id": sample["sample_id"],
            "source": sample["input"]["source"],
            "crop": sample["input"]["crop"],
            "classifier_mode": report["classifier"].get("mode"),
            "classifier_version": report["classifier"].get("version"),
            "category": prediction["category"],
            "raw_category": prediction["raw_category"],
            "confidence": prediction["confidence"],
            "is_inconclusive": prediction["is_inconclusive"],
            "inconclusive_reasons": " | ".join(prediction["inconclusive_reasons"]),
            "valid_fraction": sample["input"]["valid_fraction"],
        }
        for name in FEATURE_ORDER:
            row[name] = sample["features"].get(name)
        row["data_quality_warnings"] = " | ".join(sample["data_quality_warnings"])
        row["advisory_ids"] = "; ".join(a["rule_id"] for a in sample["advisories"])
        row["advisory_titles"] = "; ".join(a["title"] for a in sample["advisories"])
        row["disclaimer"] = report["disclaimer"]
        rows.append(row)
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# PDF (fpdf2, built-in Helvetica font)
# --------------------------------------------------------------------------

_REPLACEMENTS = {"−": "-", "–": "-", "—": "-", "≥": ">=", "≤": "<=",
                 "×": "x", "‘": "'", "’": "'", "“": '"', "”": '"'}


def _latin1(text) -> str:
    """The built-in PDF fonts only cover Latin-1, so replace anything else."""
    text = str(text)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report_to_pdf(report: dict) -> bytes:
    from fpdf import FPDF  # imported here so the rest of the module works without fpdf2

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text, size=13):
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, 7, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    def para(text, size=9, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 4.8, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    heading(report["system"]["name"], 14)
    para(f"Problem ID: {report['system']['problem_id']}   |   Report: {report['report_id']}   |   "
         f"Generated (UTC): {report['generated_at']}   |   App version: {report['system']['app_version']}")
    classifier = report["classifier"]
    para(f"Classifier: {classifier.get('name')} (mode: {classifier.get('mode')}, "
         f"version: {classifier.get('version')})")
    para(f"Provenance: {classifier.get('provenance')}")
    for notice in report["notices"]:
        para(f"Notice: {notice}", style="I")
    pdf.ln(2)
    para(report["disclaimer"], style="B")

    for sample in report["samples"]:
        pdf.ln(3)
        heading(f"Sample: {sample['sample_id']}", 12)
        info = sample["input"]
        para(f"Source: {info['source']} ({info['kind']}), crop: {info['crop']}, "
             f"valid observations: {info['n_valid']}/{info['n_observations']} "
             f"({info['valid_fraction']:.0%}), bands: {', '.join(info['bands_provided'])}")

        prediction = sample["prediction"]
        heading("Assessment", 11)
        para(f"Category: {prediction['category']}   (raw classifier output: {prediction['raw_category']}, "
             f"confidence: {_fmt(prediction['confidence'], 2)})", style="B")
        for reason in prediction["inconclusive_reasons"]:
            para(f"- Inconclusive because: {reason}")
        for note in prediction["uncertainty_notes"]:
            para(f"- {note}")

        heading("Vegetation indices (valid pixels)", 11)
        for row in sample["vegetation_indices"]:
            para(f"{row['index']}: mean {_fmt(row.get('mean'))}, std {_fmt(row.get('std'))}, "
                 f"min {_fmt(row.get('min'))}, max {_fmt(row.get('max'))}, "
                 f"undefined values {row['undefined_values']}")
        for name, reason in sample["indices_not_calculated"].items():
            para(f"{name.upper()}: not calculated - {reason}")

        heading("Explanation", 11)
        explanation = sample["explanation"]
        para(explanation["headline"])
        for row in explanation["contributions"][:5]:
            para(f"- {row['feature']} = {_fmt(row['value'])} (z {_fmt(row['zscore'], 2)}): "
                 f"contribution {row['contribution']:+.3f}")
        for rule in explanation["rule_trace"]:
            status = "FIRED/MATCHED" if rule["matched"] else "not matched"
            conditions = "; ".join(
                f"{c['text']} {'met' if c['passed'] else 'not met'}" for c in rule["all_of"] + rule["any_of"]
            )
            para(f"- {rule['rule_id']} -> {rule['category']} [{status}]: {conditions}")

        if sample["data_quality_warnings"]:
            heading("Data-quality warnings", 11)
            for warning in sample["data_quality_warnings"]:
                para(f"- {warning}")

        heading("Advisories (for human verification)", 11)
        for advisory in sample["advisories"]:
            para(f"[{advisory['severity'].upper()}] {advisory['rule_id']} - {advisory['title']}", style="B")
            para(advisory["message"])
            para(f"Verify: {advisory['verification']}")
            if advisory["limitations"] and advisory["limitations"] != "-":
                para(f"Limitations: {advisory['limitations']}", style="I")
            para(f"Source: {advisory['source']}", size=8)

    return bytes(pdf.output())

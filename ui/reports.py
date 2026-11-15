import streamlit as st

from src.reporting.report_generator import build_report, report_to_csv, report_to_json, report_to_pdf
from ui.common import get_classifier, get_settings, mode_banner, require_results


def render() -> None:
    st.title("Reports")
    mode_banner()
    results = require_results()
    classifier, notices = get_classifier(st.session_state.mode)

    st.write(
        f"The report covers **{len(results)} sample(s)** from *{st.session_state.input_label}*. "
        "It includes the input summary, vegetation indices, features, classification, model mode "
        "and version, data-quality warnings, explanation, matching advisories, a timestamp and a "
        "unique report ID."
    )

    if st.button("Generate report", type="primary") or "report" in st.session_state:
        if st.session_state.get("report_key") != st.session_state.results_key:
            st.session_state.report = build_report(results, classifier.info(), notices, get_settings())
            st.session_state.report_key = st.session_state.results_key
        report = st.session_state.report
        report_id = report["report_id"]
        st.success(f"Report **{report_id}** generated at {report['generated_at']} (UTC).")

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ JSON report", report_to_json(report), f"{report_id}.json",
                           "application/json", width="stretch")
        c2.download_button("⬇ CSV summary", report_to_csv(report), f"{report_id}.csv",
                           "text/csv", width="stretch")
        try:
            c3.download_button("⬇ PDF report", report_to_pdf(report), f"{report_id}.pdf",
                               "application/pdf", width="stretch")
        except Exception as error:  # PDF is a convenience; JSON/CSV remain available
            c3.warning(f"PDF export unavailable: {error}")

        with st.expander("Preview JSON"):
            st.json(report, expanded=False)
        st.caption("Generating again after changing the model mode or crop creates a new report ID.")

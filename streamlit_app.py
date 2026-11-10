"""Local Streamlit dashboard for single-image analyses and experiment history."""

from __future__ import annotations

import streamlit as st

from claimguard.config import get_settings
from claimguard.schemas import AnalysisReport
from claimguard.service import analyze_and_store
from claimguard.storage import ResearchStore

st.set_page_config(page_title="ClaimGuard Research", page_icon="🔎", layout="wide")
settings = get_settings()
store = ResearchStore(settings.database_path)

st.title("ClaimGuard Research")
st.caption("Research prototype · provenance and image-cue analysis · not a claim decision tool")

analyze_tab, history_tab, methods_tab = st.tabs(
    ["Analyze image", "Research history", "Method notes"]
)

with analyze_tab:
    st.warning(
        "Use public benchmark data only. Do not upload personal claim photos. "
        "No AI-manipulation probability is produced unless a calibrated fusion model is configured."
    )
    uploaded = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
        max_upload_size=max(
            1,
            (settings.max_upload_bytes + 1024 * 1024 - 1) // (1024 * 1024),
        ),
    )
    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width=420)
        if st.button("Run agents", type="primary"):
            try:
                report = analyze_and_store(
                    uploaded.getvalue(),
                    uploaded.name,
                    uploaded.type or "application/octet-stream",
                    settings,
                    store,
                )
                st.session_state["latest_report"] = report.model_dump(mode="json")
            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.exception(error)

    report_data = st.session_state.get("latest_report")
    if report_data:
        report = AnalysisReport.model_validate(report_data)
        st.subheader("Decision")
        if report.manipulation_probability is None:
            st.info("Abstained: no calibrated fusion model is configured.")
        else:
            st.metric(
                "Calibrated AI-manipulation probability", f"{report.manipulation_probability:.1%}"
            )
            st.caption(
                "This estimate is only applicable to the trained model's documented domain "
                "and is not a probability that a claim is fraudulent."
            )
        st.write(f"Decision status: `{report.decision_status}`")

        st.subheader("Agent outputs")
        agent_columns = st.columns(2)
        for index, (name, result) in enumerate(report.agents.items()):
            with agent_columns[index % 2]:
                st.markdown(f"**{name.replace('_', ' ').title()}** · `{result.status.value}`")
                if result.score is not None:
                    st.metric(result.score_name or "Score", f"{result.score:.3f}")
                    st.caption(result.score_semantics)
                for evidence in result.evidence:
                    st.write(f"- {evidence}")
                if result.error:
                    st.error(result.error)
                with st.expander("Features"):
                    st.json(result.features)
        with st.expander("Full JSON report"):
            st.json(report.model_dump(mode="json"))
        st.download_button(
            "Download report JSON",
            data=report.model_dump_json(indent=2),
            file_name=f"claimguard-{report.analysis_id}.json",
            mime="application/json",
        )

with history_tab:
    reports = store.list_analyses(100)
    if not reports:
        st.info("No analysis reports have been saved.")
    else:
        st.dataframe(
            [
                {
                    "Analysis ID": item.analysis_id,
                    "Created (UTC)": item.created_at,
                    "Filename": item.filename,
                    "SHA-256": item.sha256,
                    "Decision status": item.decision_status,
                    "AI manipulation probability": item.manipulation_probability,
                }
                for item in reports
            ],
            use_container_width=True,
            hide_index=True,
        )
    experiments = store.list_experiments(100)
    st.subheader("Benchmark experiments")
    if experiments:
        st.dataframe(
            [
                {
                    "Experiment": item.name,
                    "Dataset": item.dataset,
                    "Created (UTC)": item.created_at,
                    **item.metrics,
                }
                for item in experiments
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No experiment summaries have been stored yet.")

with methods_tab:
    st.markdown(
        """
        - **Metadata:** EXIF extraction and metadata completeness only. Missing fields are not
          treated as fraud evidence, and exact GPS coordinates are not retained.
        - **Forensics / visual:** image statistics are heuristic descriptors. They are not
          manipulation probabilities and are highly content- and encoding-dependent.
        - **AI detector:** requires a separately trained three-class Hugging Face model.
          Without it, the agent abstains.
        - **Decision:** requires a calibrated fusion artifact trained on labeled validation data.
          It abstains rather than averaging unlike or uncalibrated scores.
        - **Storage:** SQLite persists reports and image hashes, not uploaded image bytes.
        - **Privacy:** analyses can still contain filenames and EXIF metadata; use public,
          appropriately licensed benchmark images for research runs.
        """
    )

"""LangGraph-based execution of the specialized image-analysis agents."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from claimguard.agents.ai_detector import analyze_ai_image
from claimguard.agents.decision import create_decision
from claimguard.agents.forensics import analyze_forensics
from claimguard.agents.metadata import analyze_metadata
from claimguard.agents.visual import analyze_visual
from claimguard.imaging import decode_image, image_array
from claimguard.schemas import AgentResult, AgentStatus, AnalysisReport, current_time

LOGGER = logging.getLogger(__name__)


class AnalysisInput(TypedDict):
    image_bytes: bytes
    filename: str
    media_type: str
    model_path: str | None
    fusion_model_path: str


class AnalysisState(TypedDict):
    request: AnalysisInput
    results: dict[str, AgentResult]
    analysis_id: str
    created_at: str
    sha256: str
    image_width: int
    image_height: int
    manipulation_probability: float | None
    decision_status: str
    decision_model: str | None
    explanation: list[str]
    report: NotRequired[AnalysisReport]


def build_graph():
    """Compile the linear, inspectable multi-agent workflow."""
    graph = StateGraph(AnalysisState)
    graph.add_node("metadata", _metadata_node)
    graph.add_node("forensics", _forensics_node)
    graph.add_node("ai_detector", _ai_node)
    graph.add_node("visual", _visual_node)
    graph.add_node("decision", _decision_node)

    graph.add_edge(START, "metadata")
    graph.add_edge("metadata", "forensics")
    graph.add_edge("forensics", "ai_detector")
    graph.add_edge("ai_detector", "visual")
    graph.add_edge("visual", "decision")
    graph.add_edge("decision", END)
    return graph.compile()


def analyze_image(
    image_bytes: bytes,
    filename: str,
    media_type: str,
    model_path: str | None,
    fusion_model_path: Path,
) -> AnalysisReport:
    """Validate an image, execute the agent graph, and return a research report."""
    decoded = decode_image(image_bytes)
    request: AnalysisInput = {
        "image_bytes": image_bytes,
        "filename": filename,
        "media_type": media_type,
        "model_path": model_path,
        "fusion_model_path": str(fusion_model_path),
    }
    initial: AnalysisState = {
        "request": request,
        "results": {},
        "analysis_id": str(uuid.uuid4()),
        "created_at": current_time().isoformat(),
        "sha256": decoded.sha256,
        "image_width": decoded.width,
        "image_height": decoded.height,
        "manipulation_probability": None,
        "decision_status": "pending",
        "decision_model": None,
        "explanation": [],
    }
    final = build_graph().invoke(initial)
    if "report" not in final:
        raise RuntimeError("The analysis graph completed without producing a report.")
    return final["report"]


def _run_agent(name: str, operation) -> AgentResult:
    try:
        return operation()
    except Exception as error:
        LOGGER.exception("Analysis agent '%s' failed", name)
        return AgentResult(
            agent=name,
            status=AgentStatus.ERROR,
            score_name=None,
            score_semantics="Analysis failed; no score is available.",
            error=f"{type(error).__name__}: {error}",
            evidence=["This agent's output is unavailable because its analysis failed."],
        )


def _metadata_node(state: AnalysisState) -> dict[str, dict[str, AgentResult]]:
    result = _run_agent(
        "metadata",
        lambda: analyze_metadata(state["request"]["image_bytes"]),
    )
    return {"results": {**state["results"], "metadata": result}}


def _forensics_node(state: AnalysisState) -> dict[str, dict[str, AgentResult]]:
    decoded = decode_image(state["request"]["image_bytes"])
    result = _run_agent(
        "forensics",
        lambda: analyze_forensics(
            image_array(decoded.image),
            decoded.image.format or _format_from_media_type(state["request"]["media_type"]),
        ),
    )
    return {"results": {**state["results"], "forensics": result}}


def _ai_node(state: AnalysisState) -> dict[str, dict[str, AgentResult]]:
    result = _run_agent(
        "ai_detector",
        lambda: analyze_ai_image(
            state["request"]["image_bytes"],
            state["request"]["model_path"],
        ),
    )
    return {"results": {**state["results"], "ai_detector": result}}


def _visual_node(state: AnalysisState) -> dict[str, dict[str, AgentResult]]:
    decoded = decode_image(state["request"]["image_bytes"])
    result = _run_agent("visual", lambda: analyze_visual(image_array(decoded.image)))
    return {"results": {**state["results"], "visual": result}}


def _decision_node(state: AnalysisState) -> dict[str, object]:
    probability, decision_status, model_name, explanation, decision_agent = create_decision(
        agents=state["results"],
        model_path=Path(state["request"]["fusion_model_path"]),
    )
    results = {**state["results"], "decision": decision_agent}
    report = AnalysisReport(
        analysis_id=state["analysis_id"],
        created_at=current_time(),
        filename=state["request"]["filename"],
        sha256=state["sha256"],
        image_width=state["image_width"],
        image_height=state["image_height"],
        agents=results,
        manipulation_probability=probability,
        probability_target="AI-generated or AI-edited image, not claim fraud",
        decision_status=decision_status,
        decision_model=model_name,
        explanation=explanation,
        limitations=[
            "The output is a research aid, not a finding of fraud or a claim decision.",
            "Metadata completeness and handcrafted image cues are not calibrated "
            "manipulation probabilities.",
            "A trained AI detector or fusion model is unavailable unless explicitly configured.",
            "Model estimates may not generalize to unseen generators, devices, editing "
            "workflows, or claim domains.",
        ],
    )
    return {
        "results": results,
        "manipulation_probability": probability,
        "decision_status": decision_status,
        "decision_model": model_name,
        "explanation": explanation,
        "report": report,
    }


def _format_from_media_type(media_type: str) -> str:
    subtype = media_type.lower().split("/")[-1].split(";")[0]
    return "jpeg" if subtype == "jpg" else subtype

from __future__ import annotations

from sqlalchemy import select

from tame.models import GraphEdge, ProvenanceLink, SummaryNode


def test_consolidation_builds_summary_nodes_and_provenance(engine):
    namespace = "test.consolidation"
    engine.add_messages(
        [
            {"content": "We are researching temporal graphs for memory.", "source_type": "user_message"},
            {"content": "Paper note: provenance links explain source episodes.", "source_type": "imported_doc"},
        ],
        namespace_id=namespace,
        user_id="rina",
        agent_id="researcher",
        session_id="r1",
    )
    stats = engine.consolidate(namespace)
    assert stats["episodes_processed"] == 2

    with engine.session_factory() as session:
        summaries = session.scalars(select(SummaryNode).where(SummaryNode.namespace_id == namespace)).all()
        edges = session.scalars(select(GraphEdge).where(GraphEdge.namespace_id == namespace)).all()
        provenance = session.scalars(select(ProvenanceLink)).all()
        assert summaries
        assert edges
        assert provenance

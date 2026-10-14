"""In-memory financial knowledge graph and entity lineage engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multi_modal_financial.types import Document


@dataclass
class EntityNode:
    """Node in financial knowledge graph representing an entity or concept."""

    id: str
    node_type: str  # COMPANY, TICKER, PERIOD, STATEMENT, METRIC, SEGMENT, RISK_FACTOR
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityEdge:
    """Directed edge in financial knowledge graph."""

    source: str
    target: str
    relation: str  # HAS_METRIC, REPORTED_IN, SEGMENT_OF, EXPOSED_TO_RISK, VARIED_FROM
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


class FinancialKnowledgeGraph:
    """In-memory multi-relational knowledge graph for financial lineage and multi-hop reasoning."""

    def __init__(self) -> None:
        self.nodes: dict[str, EntityNode] = {}
        self.edges: list[EntityEdge] = []
        self._adj: dict[str, list[EntityEdge]] = {}
        self._rev_adj: dict[str, list[EntityEdge]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        **properties: Any,
    ) -> EntityNode:
        """Add or update an entity node in the graph."""
        if node_id not in self.nodes:
            node = EntityNode(
                id=node_id,
                node_type=node_type.upper(),
                label=label,
                properties=properties,
            )
            self.nodes[node_id] = node
            self._adj[node_id] = []
            self._rev_adj[node_id] = []
        else:
            self.nodes[node_id].properties.update(properties)
        return self.nodes[node_id]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        **properties: Any,
    ) -> EntityEdge:
        """Add a directed relational edge between two entity nodes."""
        if source_id not in self.nodes:
            self.add_node(source_id, "ENTITY", source_id)
        if target_id not in self.nodes:
            self.add_node(target_id, "ENTITY", target_id)

        edge = EntityEdge(
            source=source_id,
            target=target_id,
            relation=relation.upper(),
            weight=weight,
            properties=properties,
        )
        self.edges.append(edge)
        self._adj[source_id].append(edge)
        self._rev_adj[target_id].append(edge)
        return edge

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "out",
    ) -> list[EntityNode]:
        """Return adjacent entity nodes filtered by optional relation and direction."""
        if node_id not in self.nodes:
            return []

        rel_upper = relation.upper() if relation else None
        target_nodes: list[EntityNode] = []

        if direction in ("out", "both"):
            for edge in self._adj.get(node_id, []):
                if rel_upper is None or edge.relation == rel_upper:
                    if edge.target in self.nodes:
                        target_nodes.append(self.nodes[edge.target])

        if direction in ("in", "both"):
            for edge in self._rev_adj.get(node_id, []):
                if rel_upper is None or edge.relation == rel_upper:
                    if edge.source in self.nodes:
                        target_nodes.append(self.nodes[edge.source])

        return target_nodes

    def query_multi_hop(
        self,
        start_node_id: str,
        relation_path: list[str],
    ) -> list[EntityNode]:
        """Traverse graph along an ordered sequence of relation hops."""
        current_nodes = [self.nodes[start_node_id]] if start_node_id in self.nodes else []

        for rel in relation_path:
            next_nodes: list[EntityNode] = []
            for node in current_nodes:
                neighbors = self.get_neighbors(node.id, relation=rel, direction="out")
                next_nodes.extend(neighbors)
            # Deduplicate by node ID
            seen: set[str] = set()
            deduped: list[EntityNode] = []
            for n in next_nodes:
                if n.id not in seen:
                    seen.add(n.id)
                    deduped.append(n)
            current_nodes = deduped

        return current_nodes

    def build_from_document(self, document: Document) -> None:
        """Populate graph nodes and relationships directly from parsed financial document."""
        meta = document.metadata
        ticker = meta.ticker or "UNKNOWN"
        period = f"{meta.year or 2025}_{meta.period or 'FY'}"

        ticker_node_id = f"TICKER_{ticker}"
        self.add_node(ticker_node_id, "TICKER", ticker, ticker=ticker)

        period_node_id = f"PERIOD_{ticker}_{period}"
        self.add_node(
            period_node_id,
            "PERIOD",
            f"{ticker} {period}",
            year=meta.year,
            period=meta.period,
        )
        self.add_edge(ticker_node_id, period_node_id, "REPORTED_IN")

        # Add metric nodes from document
        for metric in document.get_all_metrics():
            m_id = f"METRIC_{ticker}_{period}_{metric.name}"
            self.add_node(
                m_id,
                "METRIC",
                metric.name,
                value=metric.normalized_value,
                unit=str(metric.unit.value if hasattr(metric.unit, "value") else metric.unit),
                scale=str(metric.scale.value if hasattr(metric.scale, "value") else metric.scale),
            )
            self.add_edge(period_node_id, m_id, "HAS_METRIC")

        # Add risk factors if present
        for chunk in document.chunks:
            if chunk.section_title and "risk" in chunk.section_title.lower():
                risk_id = f"RISK_{chunk.chunk_id}"
                snippet = chunk.content[:80].strip() + "..."
                self.add_node(risk_id, "RISK_FACTOR", snippet, chunk_id=chunk.chunk_id)
                self.add_edge(period_node_id, risk_id, "EXPOSED_TO_RISK")

    def to_dict(self) -> dict[str, Any]:
        """Serialize knowledge graph to primitive dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "label": n.label,
                    "properties": n.properties,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "weight": e.weight,
                    "properties": e.properties,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinancialKnowledgeGraph:
        """Reconstruct graph from serialized dictionary."""
        graph = cls()
        for nd in data.get("nodes", []):
            graph.add_node(nd["id"], nd["type"], nd["label"], **nd.get("properties", {}))
        for ed in data.get("edges", []):
            graph.add_edge(
                ed["source"],
                ed["target"],
                ed["relation"],
                weight=ed.get("weight", 1.0),
                **ed.get("properties", {}),
            )
        return graph

    def to_dot(self) -> str:
        """Export graph to Graphviz DOT diagram format."""
        lines = [
            "digraph FinancialKnowledgeGraph {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded];",
        ]
        for node in self.nodes.values():
            clean_label = node.label.replace('"', '\\"')
            lines.append(f'  "{node.id}" [label="{clean_label}\\n({node.node_type})"];')
        for edge in self.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.relation}"];')
        lines.append("}")
        return "\n".join(lines)

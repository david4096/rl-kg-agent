"""Internal Knowledge Graph for agent memory and learned knowledge."""

from typing import Dict, List, Optional, Any, Tuple, Set
import json
import logging
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a node in the internal knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance_score: float = 0.0

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


@dataclass
class KnowledgeEdge:
    """Represents an edge/relationship in the internal knowledge graph."""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any]
    created_at: datetime
    confidence: float = 1.0
    derived_from: Optional[str] = None  # Source query or action that created this edge

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)


@dataclass
class QueryContext:
    """Represents context from a query interaction."""
    query: str
    response: str
    entities_mentioned: List[str]
    relations_discovered: List[str]
    timestamp: datetime
    action_taken: str
    success: bool

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class InternalKnowledgeGraph:
    """Internal knowledge graph for storing agent's learned knowledge and memory."""

    def __init__(self, storage_path: str = "internal_kg.pkl"):
        """Initialize the internal knowledge graph.

        Args:
            storage_path: Path to store the knowledge graph
        """
        self.storage_path = storage_path
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.contexts: List[QueryContext] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity_type -> node_ids
        self.relation_index: Dict[str, List[KnowledgeEdge]] = defaultdict(list)

        self.load()

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
        """Add a node to the knowledge graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type/category of the node
            properties: Dictionary of node properties
        """
        now = datetime.now()

        if node_id in self.nodes:
            # Update existing node
            self.nodes[node_id].properties.update(properties)
            self.nodes[node_id].last_accessed = now
            self.nodes[node_id].access_count += 1
        else:
            # Create new node
            node = KnowledgeNode(
                id=node_id,
                type=node_type,
                properties=properties,
                created_at=now,
                last_accessed=now,
                access_count=1
            )
            self.nodes[node_id] = node
            self.entity_index[node_type].add(node_id)

        logger.debug(f"Added/updated node: {node_id} ({node_type})")

    def add_edge(self, source: str, target: str, relation: str,
                 properties: Optional[Dict[str, Any]] = None,
                 confidence: float = 1.0, derived_from: Optional[str] = None) -> None:
        """Add an edge to the knowledge graph.

        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            properties: Optional edge properties
            confidence: Confidence score for this relationship
            derived_from: Source information for this edge
        """
        if properties is None:
            properties = {}

        edge = KnowledgeEdge(
            source=source,
            target=target,
            relation=relation,
            properties=properties,
            created_at=datetime.now(),
            confidence=confidence,
            derived_from=derived_from
        )

        self.edges.append(edge)
        self.relation_index[relation].append(edge)
        logger.debug(f"Added edge: {source} --{relation}--> {target}")

    def add_context(self, query: str, response: str, entities: List[str],
                   relations: List[str], action: str, success: bool) -> None:
        """Add query context to memory.

        Args:
            query: Original user query
            response: Agent's response
            entities: Entities mentioned/discovered
            relations: Relations mentioned/discovered
            action: Action taken by agent
            success: Whether the action was successful
        """
        context = QueryContext(
            query=query,
            response=response,
            entities_mentioned=entities,
            relations_discovered=relations,
            timestamp=datetime.now(),
            action_taken=action,
            success=success
        )
        self.contexts.append(context)

        # Update importance scores for mentioned entities
        for entity in entities:
            if entity in self.nodes:
                self.nodes[entity].importance_score += 0.1 if success else -0.05

        logger.debug(f"Added context for query: {query[:50]}...")

    def get_relevant_nodes(self, query_entities: List[str],
                          node_types: Optional[List[str]] = None,
                          limit: int = 10) -> List[KnowledgeNode]:
        """Get nodes relevant to a query.

        Args:
            query_entities: Entities mentioned in the query
            node_types: Filter by node types
            limit: Maximum number of nodes to return

        Returns:
            List of relevant nodes sorted by relevance
        """
        relevant_nodes = []

        # Direct matches
        for entity in query_entities:
            if entity in self.nodes:
                relevant_nodes.append(self.nodes[entity])

        # Nodes connected to query entities
        connected_nodes = set()
        for edge in self.edges:
            if edge.source in query_entities:
                connected_nodes.add(edge.target)
            if edge.target in query_entities:
                connected_nodes.add(edge.source)

        for node_id in connected_nodes:
            if node_id in self.nodes:
                relevant_nodes.append(self.nodes[node_id])

        # Filter by type if specified
        if node_types:
            relevant_nodes = [n for n in relevant_nodes if n.type in node_types]

        # Sort by importance and recency
        relevant_nodes.sort(key=lambda n: (n.importance_score, n.last_accessed), reverse=True)

        return relevant_nodes[:limit]

    def get_node_neighbors(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Get neighboring nodes of a given node.

        Args:
            node_id: ID of the central node
            relation_types: Filter by relation types

        Returns:
            List of (neighbor_id, relation, direction) tuples
        """
        neighbors = []

        for edge in self.edges:
            if relation_types and edge.relation not in relation_types:
                continue

            if edge.source == node_id:
                neighbors.append((edge.target, edge.relation, "outgoing"))
            elif edge.target == node_id:
                neighbors.append((edge.source, edge.relation, "incoming"))

        return neighbors

    def get_path_between_nodes(self, start: str, end: str, max_hops: int = 3) -> Optional[List[str]]:
        """Find shortest path between two nodes.

        Args:
            start: Start node ID
            end: End node ID
            max_hops: Maximum number of hops to consider

        Returns:
            List of node IDs forming the path, or None if no path found
        """
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            neighbors = self.get_node_neighbors(current)
            for neighbor_id, relation, direction in neighbors:
                if neighbor_id == end:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_similar_contexts(self, current_query: str, limit: int = 5) -> List[QueryContext]:
        """Find contexts similar to the current query.

        Args:
            current_query: Current query text
            limit: Maximum number of contexts to return

        Returns:
            List of similar query contexts
        """
        # Handle empty query case
        if not current_query or not current_query.strip():
            return []

        # Simple similarity based on shared words (could be enhanced with embeddings)
        current_words = set(current_query.lower().split())

        # If current query has no words, return empty list
        if not current_words:
            return []

        scored_contexts = []
        for context in self.contexts:
            context_words = set(context.query.lower().split())

            # Calculate Jaccard similarity, handle division by zero
            union_size = len(current_words.union(context_words))
            if union_size == 0:
                similarity = 0.0
            else:
                similarity = len(current_words.intersection(context_words)) / union_size

            scored_contexts.append((similarity, context))

        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        return [context for _, context in scored_contexts[:limit]]

    def update_node_importance(self, node_id: str, delta: float) -> None:
        """Update importance score of a node.

        Args:
            node_id: Node to update
            delta: Change in importance score
        """
        if node_id in self.nodes:
            self.nodes[node_id].importance_score += delta
            self.nodes[node_id].last_accessed = datetime.now()

    def prune_low_importance_nodes(self, threshold: float = -1.0) -> int:
        """Remove nodes with very low importance scores.

        Args:
            threshold: Importance threshold below which nodes are removed

        Returns:
            Number of nodes removed
        """
        to_remove = [node_id for node_id, node in self.nodes.items()
                    if node.importance_score < threshold and node.access_count < 2]

        # Remove nodes
        for node_id in to_remove:
            del self.nodes[node_id]
            # Remove from entity index
            for entity_set in self.entity_index.values():
                entity_set.discard(node_id)

        # Remove associated edges
        self.edges = [e for e in self.edges if e.source not in to_remove and e.target not in to_remove]

        # Rebuild relation index
        self.relation_index = defaultdict(list)
        for edge in self.edges:
            self.relation_index[edge.relation].append(edge)

        logger.info(f"Pruned {len(to_remove)} low-importance nodes")
        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the internal knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.type] += 1

        relation_types = defaultdict(int)
        for edge in self.edges:
            relation_types[edge.relation] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_contexts": len(self.contexts),
            "node_types": dict(node_types),
            "relation_types": dict(relation_types),
            "avg_node_importance": sum(n.importance_score for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
        }

    def save(self) -> None:
        """Save the knowledge graph to disk."""
        data = {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "contexts": [asdict(c) for c in self.contexts]
        }

        with open(self.storage_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved internal KG to {self.storage_path}")

    def load(self) -> None:
        """Load the knowledge graph from disk."""
        if not Path(self.storage_path).exists():
            logger.info("No existing internal KG found, starting fresh")
            return

        try:
            with open(self.storage_path, "rb") as f:
                data = pickle.load(f)

            # Reconstruct nodes
            self.nodes = {k: KnowledgeNode(**v) for k, v in data["nodes"].items()}

            # Reconstruct edges
            self.edges = [KnowledgeEdge(**e) for e in data["edges"]]

            # Reconstruct contexts
            self.contexts = [QueryContext(**c) for c in data["contexts"]]

            # Rebuild indices
            self.entity_index = defaultdict(set)
            for node_id, node in self.nodes.items():
                self.entity_index[node.type].add(node_id)

            self.relation_index = defaultdict(list)
            for edge in self.edges:
                self.relation_index[edge.relation].append(edge)

            logger.info(f"Loaded internal KG from {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to load internal KG: {e}")
            # Reset to empty state
            self.nodes = {}
            self.edges = []
            self.contexts = []
            self.entity_index = defaultdict(set)
            self.relation_index = defaultdict(list)
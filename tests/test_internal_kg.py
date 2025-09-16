"""Tests for internal knowledge graph functionality."""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path

from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph, KnowledgeNode, KnowledgeEdge, QueryContext


class TestInternalKnowledgeGraph:
    """Test cases for InternalKnowledgeGraph."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def internal_kg(self, temp_storage_path):
        """Create an InternalKnowledgeGraph instance for testing."""
        return InternalKnowledgeGraph(temp_storage_path)

    def test_init(self, internal_kg):
        """Test initialization."""
        assert len(internal_kg.nodes) == 0
        assert len(internal_kg.edges) == 0
        assert len(internal_kg.contexts) == 0
        assert len(internal_kg.entity_index) == 0
        assert len(internal_kg.relation_index) == 0

    def test_add_node(self, internal_kg):
        """Test adding nodes."""
        # Add first node
        internal_kg.add_node("paris", "city", {"population": 2161000})

        assert "paris" in internal_kg.nodes
        assert internal_kg.nodes["paris"].type == "city"
        assert internal_kg.nodes["paris"].properties["population"] == 2161000
        assert internal_kg.nodes["paris"].access_count == 1

        # Add same node again (should update)
        internal_kg.add_node("paris", "city", {"country": "France"})

        assert internal_kg.nodes["paris"].access_count == 2
        assert "country" in internal_kg.nodes["paris"].properties
        assert "population" in internal_kg.nodes["paris"].properties  # Should keep old properties

    def test_add_edge(self, internal_kg):
        """Test adding edges."""
        # Add nodes first
        internal_kg.add_node("paris", "city", {})
        internal_kg.add_node("france", "country", {})

        # Add edge
        internal_kg.add_edge("paris", "france", "capital_of", {"strength": 1.0}, confidence=0.9)

        assert len(internal_kg.edges) == 1
        edge = internal_kg.edges[0]

        assert edge.source == "paris"
        assert edge.target == "france"
        assert edge.relation == "capital_of"
        assert edge.confidence == 0.9
        assert edge.properties["strength"] == 1.0

        # Check relation index
        assert "capital_of" in internal_kg.relation_index
        assert len(internal_kg.relation_index["capital_of"]) == 1

    def test_add_context(self, internal_kg):
        """Test adding query contexts."""
        query = "What is the capital of France?"
        response = "Paris is the capital of France."
        entities = ["Paris", "France"]
        relations = ["capital_of"]

        internal_kg.add_context(query, response, entities, relations, "sparql_query", True)

        assert len(internal_kg.contexts) == 1
        context = internal_kg.contexts[0]

        assert context.query == query
        assert context.response == response
        assert context.entities_mentioned == entities
        assert context.relations_discovered == relations
        assert context.action_taken == "sparql_query"
        assert context.success is True

    def test_get_relevant_nodes(self, internal_kg):
        """Test getting relevant nodes."""
        # Add some nodes
        internal_kg.add_node("paris", "city", {})
        internal_kg.add_node("france", "country", {})
        internal_kg.add_node("london", "city", {})

        # Add edges
        internal_kg.add_edge("paris", "france", "capital_of")

        # Get relevant nodes for entities
        relevant = internal_kg.get_relevant_nodes(["paris"])

        assert len(relevant) >= 1
        node_ids = [node.id for node in relevant]
        assert "paris" in node_ids

    def test_get_node_neighbors(self, internal_kg):
        """Test getting node neighbors."""
        # Add nodes and edges
        internal_kg.add_node("paris", "city", {})
        internal_kg.add_node("france", "country", {})
        internal_kg.add_edge("paris", "france", "capital_of")

        neighbors = internal_kg.get_node_neighbors("paris")

        assert len(neighbors) == 1
        neighbor_id, relation, direction = neighbors[0]
        assert neighbor_id == "france"
        assert relation == "capital_of"
        assert direction == "outgoing"

        # Test reverse direction
        neighbors_france = internal_kg.get_node_neighbors("france")
        assert len(neighbors_france) == 1
        neighbor_id, relation, direction = neighbors_france[0]
        assert neighbor_id == "paris"
        assert direction == "incoming"

    def test_get_path_between_nodes(self, internal_kg):
        """Test finding paths between nodes."""
        # Create a simple path: A -> B -> C
        internal_kg.add_node("a", "node", {})
        internal_kg.add_node("b", "node", {})
        internal_kg.add_node("c", "node", {})

        internal_kg.add_edge("a", "b", "connects_to")
        internal_kg.add_edge("b", "c", "connects_to")

        # Test direct connection
        path_direct = internal_kg.get_path_between_nodes("a", "b")
        assert path_direct == ["a", "b"]

        # Test 2-hop path
        path_indirect = internal_kg.get_path_between_nodes("a", "c")
        assert path_indirect == ["a", "b", "c"]

        # Test same node
        path_same = internal_kg.get_path_between_nodes("a", "a")
        assert path_same == ["a"]

        # Test no path
        internal_kg.add_node("isolated", "node", {})
        path_none = internal_kg.get_path_between_nodes("a", "isolated")
        assert path_none is None

    def test_get_similar_contexts(self, internal_kg):
        """Test getting similar contexts."""
        # Add some contexts
        internal_kg.add_context("What is the capital of France?", "Paris", [], [], "action1", True)
        internal_kg.add_context("What is the capital of Germany?", "Berlin", [], [], "action2", True)
        internal_kg.add_context("What is the weather like?", "Sunny", [], [], "action3", True)

        # Find similar contexts
        similar = internal_kg.get_similar_contexts("What is the capital of Italy?", limit=2)

        assert len(similar) <= 2
        # Should prioritize contexts with similar words ("capital")
        assert any("capital" in context.query for context in similar)

    def test_update_node_importance(self, internal_kg):
        """Test updating node importance."""
        internal_kg.add_node("test_node", "test", {})

        original_score = internal_kg.nodes["test_node"].importance_score

        internal_kg.update_node_importance("test_node", 0.5)

        new_score = internal_kg.nodes["test_node"].importance_score
        assert new_score == original_score + 0.5

    def test_prune_low_importance_nodes(self, internal_kg):
        """Test pruning low importance nodes."""
        # Add nodes with different importance scores
        internal_kg.add_node("important", "node", {})
        internal_kg.add_node("unimportant", "node", {})

        internal_kg.update_node_importance("important", 1.0)
        internal_kg.update_node_importance("unimportant", -2.0)

        # Add edge involving unimportant node
        internal_kg.add_edge("important", "unimportant", "test_relation")

        initial_node_count = len(internal_kg.nodes)
        initial_edge_count = len(internal_kg.edges)

        # Prune with threshold -1.0
        pruned_count = internal_kg.prune_low_importance_nodes(threshold=-1.0)

        assert pruned_count == 1
        assert "important" in internal_kg.nodes
        assert "unimportant" not in internal_kg.nodes

        # Edges should also be removed
        assert len(internal_kg.edges) < initial_edge_count

    def test_get_stats(self, internal_kg):
        """Test getting statistics."""
        # Add some data
        internal_kg.add_node("node1", "type1", {})
        internal_kg.add_node("node2", "type2", {})
        internal_kg.add_edge("node1", "node2", "relation1")
        internal_kg.add_context("test query", "test response", [], [], "test_action", True)

        stats = internal_kg.get_stats()

        assert isinstance(stats, dict)
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["total_contexts"] == 1
        assert "node_types" in stats
        assert "relation_types" in stats
        assert "avg_node_importance" in stats

    def test_save_and_load(self, temp_storage_path):
        """Test saving and loading the knowledge graph."""
        # Create and populate KG
        kg1 = InternalKnowledgeGraph(temp_storage_path)
        kg1.add_node("test_node", "test_type", {"key": "value"})
        kg1.add_edge("test_node", "test_node", "self_relation")
        kg1.add_context("test query", "test response", [], [], "test_action", True)

        # Save
        kg1.save()

        # Create new instance and load
        kg2 = InternalKnowledgeGraph(temp_storage_path)

        # Check data was loaded
        assert len(kg2.nodes) == 1
        assert len(kg2.edges) == 1
        assert len(kg2.contexts) == 1

        assert "test_node" in kg2.nodes
        assert kg2.nodes["test_node"].type == "test_type"
        assert kg2.nodes["test_node"].properties["key"] == "value"


class TestKnowledgeNode:
    """Test cases for KnowledgeNode."""

    def test_creation(self):
        """Test node creation."""
        now = datetime.now()
        node = KnowledgeNode(
            id="test",
            type="test_type",
            properties={"key": "value"},
            created_at=now,
            last_accessed=now
        )

        assert node.id == "test"
        assert node.type == "test_type"
        assert node.properties["key"] == "value"
        assert node.created_at == now
        assert node.last_accessed == now
        assert node.access_count == 0
        assert node.importance_score == 0.0

    def test_string_datetime_conversion(self):
        """Test conversion from string datetime."""
        now = datetime.now()
        node = KnowledgeNode(
            id="test",
            type="test_type",
            properties={},
            created_at=now.isoformat(),
            last_accessed=now.isoformat()
        )

        assert isinstance(node.created_at, datetime)
        assert isinstance(node.last_accessed, datetime)


class TestKnowledgeEdge:
    """Test cases for KnowledgeEdge."""

    def test_creation(self):
        """Test edge creation."""
        now = datetime.now()
        edge = KnowledgeEdge(
            source="a",
            target="b",
            relation="test_relation",
            properties={"weight": 1.0},
            created_at=now
        )

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.relation == "test_relation"
        assert edge.properties["weight"] == 1.0
        assert edge.created_at == now
        assert edge.confidence == 1.0
        assert edge.derived_from is None


class TestQueryContext:
    """Test cases for QueryContext."""

    def test_creation(self):
        """Test context creation."""
        now = datetime.now()
        context = QueryContext(
            query="test query",
            response="test response",
            entities_mentioned=["entity1"],
            relations_discovered=["relation1"],
            timestamp=now,
            action_taken="test_action",
            success=True
        )

        assert context.query == "test query"
        assert context.response == "test response"
        assert context.entities_mentioned == ["entity1"]
        assert context.relations_discovered == ["relation1"]
        assert context.timestamp == now
        assert context.action_taken == "test_action"
        assert context.success is True


if __name__ == "__main__":
    pytest.main([__file__])
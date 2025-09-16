"""Tests for knowledge graph loader functionality."""

import pytest
import tempfile
import os
from pathlib import Path

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator


@pytest.fixture
def simple_ttl_file():
    """Create a temporary TTL file for testing."""
    ttl_content = """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Paris rdfs:label "Paris" .
ex:Paris ex:isCapitalOf ex:France .
ex:France rdfs:label "France" .
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write(ttl_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


class TestKnowledgeGraphLoader:
    """Test cases for KnowledgeGraphLoader."""

    def test_init_empty(self):
        """Test initialization without TTL file."""
        loader = KnowledgeGraphLoader()
        assert len(loader.graph) == 0
        assert len(loader.namespaces) == 0

    def test_load_ttl(self, simple_ttl_file):
        """Test loading TTL file."""
        loader = KnowledgeGraphLoader()
        loader.load_ttl(simple_ttl_file)

        # Should have loaded some triples
        assert len(loader.graph) > 0

        # Should have extracted namespaces
        assert len(loader.namespaces) > 0

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = KnowledgeGraphLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_ttl("nonexistent.ttl")

    def test_execute_sparql(self, simple_ttl_file):
        """Test SPARQL query execution."""
        loader = KnowledgeGraphLoader(simple_ttl_file)

        # Simple SELECT query
        query = """
        SELECT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
        } LIMIT 5
        """

        results = loader.execute_sparql(query)
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be a dictionary
        for result in results:
            assert isinstance(result, dict)

    def test_validate_sparql_syntax(self, simple_ttl_file):
        """Test SPARQL syntax validation."""
        loader = KnowledgeGraphLoader(simple_ttl_file)

        # Valid query
        valid_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"
        assert loader.validate_sparql_syntax(valid_query) is True

        # Invalid query
        invalid_query = "INVALID SPARQL SYNTAX"
        assert loader.validate_sparql_syntax(invalid_query) is False

    def test_get_schema_info(self, simple_ttl_file):
        """Test schema information extraction."""
        loader = KnowledgeGraphLoader(simple_ttl_file)

        schema_info = loader.get_schema_info()

        assert isinstance(schema_info, dict)
        assert "total_triples" in schema_info
        assert "classes" in schema_info
        assert "predicates" in schema_info
        assert "namespaces" in schema_info

        assert schema_info["total_triples"] > 0
        assert isinstance(schema_info["classes"], list)
        assert isinstance(schema_info["predicates"], list)
        assert isinstance(schema_info["namespaces"], dict)


class TestSPARQLQueryGenerator:
    """Test cases for SPARQLQueryGenerator."""

    def test_entity_search_query(self):
        """Test entity search query generation."""
        generator = SPARQLQueryGenerator()

        query = generator.generate_entity_search_query("Paris")

        assert isinstance(query, str)
        assert "SELECT" in query
        assert "Paris" in query
        assert "FILTER" in query

    def test_entity_search_query_with_filters(self):
        """Test entity search query with property filters."""
        generator = SPARQLQueryGenerator()

        filters = {"http://example.org/type": "http://example.org/City"}
        query = generator.generate_entity_search_query("Paris", filters)

        assert isinstance(query, str)
        assert "Paris" in query
        assert "http://example.org/type" in query

    def test_relationship_query(self):
        """Test relationship query generation."""
        generator = SPARQLQueryGenerator()

        query = generator.generate_relationship_query(
            "http://example.org/Paris",
            "http://example.org/France"
        )

        assert isinstance(query, str)
        assert "SELECT" in query
        assert "predicate" in query
        assert "Paris" in query
        assert "France" in query

    def test_neighbors_query_one_hop(self):
        """Test neighbors query with 1 hop."""
        generator = SPARQLQueryGenerator()

        query = generator.generate_neighbors_query("http://example.org/Paris", hops=1)

        assert isinstance(query, str)
        assert "SELECT" in query
        assert "neighbor" in query
        assert "Paris" in query
        assert "UNION" in query

    def test_neighbors_query_two_hops(self):
        """Test neighbors query with 2 hops."""
        generator = SPARQLQueryGenerator()

        query = generator.generate_neighbors_query("http://example.org/Paris", hops=2)

        assert isinstance(query, str)
        assert "SELECT" in query
        assert "neighbor" in query
        assert "Paris" in query
        assert "intermediate" in query

    def test_neighbors_query_invalid_hops(self):
        """Test neighbors query with invalid hops raises error."""
        generator = SPARQLQueryGenerator()

        with pytest.raises(ValueError):
            generator.generate_neighbors_query("http://example.org/Paris", hops=3)


if __name__ == "__main__":
    pytest.main([__file__])
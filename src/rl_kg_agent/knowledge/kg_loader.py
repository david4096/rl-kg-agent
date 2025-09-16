"""Knowledge Graph loading and SPARQL query functionality."""

from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery
from SPARQLWrapper import SPARQLWrapper, JSON


logger = logging.getLogger(__name__)


class KnowledgeGraphLoader:
    """Loads and manages RDF knowledge graphs with SPARQL querying capabilities."""

    def __init__(self, ttl_file_path: Optional[str] = None):
        """Initialize the KG loader.

        Args:
            ttl_file_path: Path to TTL file to load initially
        """
        self.graph = Graph()
        self.namespaces: Dict[str, Namespace] = {}

        if ttl_file_path:
            self.load_ttl(ttl_file_path)

    def load_ttl(self, ttl_file_path: str) -> None:
        """Load RDF data from TTL file.

        Args:
            ttl_file_path: Path to the TTL file
        """
        file_path = Path(ttl_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"TTL file not found: {ttl_file_path}")

        logger.info(f"Loading TTL file: {ttl_file_path}")
        self.graph.parse(ttl_file_path, format="turtle")

        # Extract namespaces for easier querying
        for prefix, namespace in self.graph.namespaces():
            self.namespaces[prefix] = Namespace(namespace)

        logger.info(f"Loaded {len(self.graph)} triples from {ttl_file_path}")

    def execute_sparql(self, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against the loaded graph.

        Args:
            query: SPARQL query string

        Returns:
            List of result dictionaries
        """
        try:
            results = self.graph.query(query)

            # Convert results to list of dictionaries
            result_list = []
            for row in results:
                result_dict = {}
                for var in results.vars:
                    value = row[var]
                    if isinstance(value, URIRef):
                        result_dict[str(var)] = str(value)
                    elif isinstance(value, Literal):
                        result_dict[str(var)] = str(value)
                    elif value is not None:
                        result_dict[str(var)] = str(value)
                result_list.append(result_dict)

            logger.info(f"SPARQL query returned {len(result_list)} results")
            return result_list

        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    def validate_sparql_syntax(self, query: str) -> bool:
        """Validate SPARQL query syntax.

        Args:
            query: SPARQL query string

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            prepareQuery(query)
            return True
        except Exception as e:
            logger.warning(f"Invalid SPARQL syntax: {e}")
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get basic schema information about the loaded graph.

        Returns:
            Dictionary with schema information
        """
        # Query for classes
        classes_query = """
        SELECT DISTINCT ?class WHERE {
            ?s a ?class .
        }
        """
        classes = self.execute_sparql(classes_query)

        # Query for predicates
        predicates_query = """
        SELECT DISTINCT ?predicate WHERE {
            ?s ?predicate ?o .
        }
        """
        predicates = self.execute_sparql(predicates_query)

        return {
            "total_triples": len(self.graph),
            "classes": [c["class"] for c in classes],
            "predicates": [p["predicate"] for p in predicates],
            "namespaces": {prefix: str(ns) for prefix, ns in self.namespaces.items()}
        }

    def find_entities_by_type(self, entity_type: str, limit: int = 10) -> List[str]:
        """Find entities of a specific type.

        Args:
            entity_type: RDF type to search for
            limit: Maximum number of results

        Returns:
            List of entity URIs
        """
        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity a <{entity_type}> .
        }} LIMIT {limit}
        """

        results = self.execute_sparql(query)
        return [r["entity"] for r in results]

    def get_entity_properties(self, entity_uri: str) -> List[Dict[str, str]]:
        """Get all properties of an entity.

        Args:
            entity_uri: URI of the entity

        Returns:
            List of property-value pairs
        """
        query = f"""
        SELECT ?property ?value WHERE {{
            <{entity_uri}> ?property ?value .
        }}
        """

        return self.execute_sparql(query)


class SPARQLQueryGenerator:
    """Generates SPARQL queries for common patterns."""

    @staticmethod
    def generate_entity_search_query(entity_name: str, property_filters: Optional[Dict] = None) -> str:
        """Generate query to search for entities by name.

        Args:
            entity_name: Name or label to search for
            property_filters: Optional property filters

        Returns:
            SPARQL query string
        """
        query = f"""
        SELECT DISTINCT ?entity ?label WHERE {{
            ?entity rdfs:label ?label .
            FILTER(CONTAINS(LCASE(?label), LCASE("{entity_name}")))
        """

        if property_filters:
            for prop, value in property_filters.items():
                query += f"\n            ?entity <{prop}> <{value}> ."

        query += "\n        } LIMIT 20"
        return query

    @staticmethod
    def generate_relationship_query(subject: str, object_entity: str) -> str:
        """Generate query to find relationships between entities.

        Args:
            subject: Subject entity URI
            object_entity: Object entity URI

        Returns:
            SPARQL query string
        """
        return f"""
        SELECT ?predicate WHERE {{
            <{subject}> ?predicate <{object_entity}> .
        }}
        """

    @staticmethod
    def generate_neighbors_query(entity_uri: str, hops: int = 1) -> str:
        """Generate query to find neighboring entities.

        Args:
            entity_uri: Central entity URI
            hops: Number of hops (1 or 2 supported)

        Returns:
            SPARQL query string
        """
        if hops == 1:
            return f"""
            SELECT DISTINCT ?neighbor ?relation WHERE {{
                {{ <{entity_uri}> ?relation ?neighbor . }}
                UNION
                {{ ?neighbor ?relation <{entity_uri}> . }}
            }} LIMIT 50
            """
        elif hops == 2:
            return f"""
            SELECT DISTINCT ?neighbor ?path WHERE {{
                {{
                    <{entity_uri}> ?r1 ?intermediate .
                    ?intermediate ?r2 ?neighbor .
                    BIND(CONCAT(STR(?r1), " -> ", STR(?r2)) AS ?path)
                }}
                UNION
                {{
                    ?intermediate ?r1 <{entity_uri}> .
                    ?neighbor ?r2 ?intermediate .
                    BIND(CONCAT(STR(?r2), " -> ", STR(?r1)) AS ?path)
                }}
                FILTER(?neighbor != <{entity_uri}>)
            }} LIMIT 100
            """
        else:
            raise ValueError("Only 1 or 2 hops supported")
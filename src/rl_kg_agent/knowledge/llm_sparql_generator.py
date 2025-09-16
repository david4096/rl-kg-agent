"""LLM-powered SPARQL query generation."""

from typing import Dict, List, Any, Optional
import logging
import re
from .kg_loader import SPARQLQueryGenerator


logger = logging.getLogger(__name__)


class LLMSPARQLGenerator:
    """SPARQL query generator using LLM for intelligent query construction."""

    def __init__(self, llm_client, kg_loader):
        """Initialize LLM SPARQL generator.

        Args:
            llm_client: LLM client for query generation
            kg_loader: Knowledge graph loader for schema information
        """
        self.llm_client = llm_client
        self.kg_loader = kg_loader
        self.fallback_generator = SPARQLQueryGenerator()

    def generate_sparql_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate SPARQL query using LLM based on conversation context.

        Args:
            context: Conversation context including query, entities, history, etc.

        Returns:
            Generated SPARQL query string or None if generation fails
        """
        query = context.get("query", "")
        entities = context.get("entities", [])

        if not query:
            return None

        try:
            # Get knowledge graph schema information
            schema_info = self.kg_loader.get_schema_info()

            # Create LLM prompt for SPARQL generation
            sparql_prompt = self._create_sparql_generation_prompt(
                query, entities, schema_info, context
            )

            # Generate SPARQL using LLM
            llm_response = self.llm_client.generate_response([
                {"role": "system", "content": "You are an expert SPARQL query generator. Generate valid SPARQL queries based on natural language questions and knowledge graph schema."},
                {"role": "user", "content": sparql_prompt}
            ])

            # Extract SPARQL query from LLM response
            sparql_query = self._extract_sparql_from_response(llm_response)

            if sparql_query and self.kg_loader.validate_sparql_syntax(sparql_query):
                logger.info(f"Successfully generated SPARQL query via LLM")
                logger.debug(f"Generated query: {sparql_query}")
                return sparql_query
            else:
                logger.warning("LLM-generated SPARQL query is invalid, falling back to rule-based generation")
                return self._fallback_generation(query, entities)

        except Exception as e:
            logger.error(f"LLM SPARQL generation failed: {e}")
            return self._fallback_generation(query, entities)

    def _create_sparql_generation_prompt(self, query: str, entities: List[str],
                                       schema_info: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create a detailed prompt for SPARQL generation.

        Args:
            query: Natural language query
            entities: Detected entities
            schema_info: Knowledge graph schema information
            context: Additional context

        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You need to generate a SPARQL query to answer this natural language question:

Question: "{query}"

Detected entities: {entities if entities else "None detected"}

Knowledge Graph Schema Information:
- Total triples: {schema_info.get('total_triples', 0)}
- Available predicates: {schema_info.get('predicates', [])[:20]}  # Show first 20
- Available classes: {schema_info.get('classes', [])[:20]}  # Show first 20
- Namespaces: {schema_info.get('namespaces', {})}

Previous conversation context: {context.get('internal_knowledge', 'None')}

Instructions:
1. Generate a valid SPARQL SELECT query that would answer the natural language question
2. Use the available predicates and classes from the schema
3. Consider common RDF patterns like rdfs:label for entity names
4. Look for relationships that might connect the entities mentioned
5. Include appropriate FILTER clauses if needed for text matching
6. Limit results to a reasonable number (usually LIMIT 10 or 20)
7. Use prefixes when possible to make the query more readable

Common patterns to consider:
- For "What is the capital of X?": Look for patterns like "?capital ?relation ?country" where relation might be "isCapitalOf" or similar
- For "Who wrote X?": Look for patterns like "?book ?authorRelation ?author" where authorRelation might be "writtenBy", "hasAuthor", etc.
- For "Where is X located?": Look for location-related predicates

Please provide ONLY the SPARQL query without explanation, starting with SELECT or ASK."""

        return prompt

    def _extract_sparql_from_response(self, response: str) -> Optional[str]:
        """Extract SPARQL query from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted SPARQL query or None
        """
        if not response:
            return None

        # Clean up the response
        response = response.strip()

        # First look for PREFIX declarations to include them
        prefix_start = None
        prefix_match = re.search(r'PREFIX\s+\w+:', response, re.IGNORECASE)
        if prefix_match:
            prefix_start = prefix_match.start()

        # Find SPARQL queries by looking for keywords and proper brace matching
        sparql_keywords = ['SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE']

        for keyword in sparql_keywords:
            keyword_pattern = rf'{keyword}.*?WHERE\s*\{{'
            match = re.search(keyword_pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                # Use prefix start if available, otherwise use query start
                start_pos = prefix_start if prefix_start is not None else match.start()
                query = self._extract_complete_query(response, start_pos)
                if query:
                    return query

        # If no structured pattern found, check if the entire response looks like SPARQL
        if any(keyword in response.upper() for keyword in ['SELECT', 'WHERE', 'PREFIX']):
            # Clean and return the whole response
            return response.strip()

        return None

    def _extract_complete_query(self, text: str, start_pos: int) -> Optional[str]:
        """Extract complete SPARQL query with proper brace matching.

        Args:
            text: Full text containing the query
            start_pos: Starting position of the query

        Returns:
            Complete SPARQL query string or None
        """
        # Find the WHERE clause from the start position
        where_match = re.search(r'WHERE\s*\{', text[start_pos:], re.IGNORECASE)
        if not where_match:
            return None

        # Calculate absolute positions
        where_pos = start_pos + where_match.start()
        brace_start = start_pos + where_match.end() - 1  # Position of opening brace

        # Count braces to find the matching closing brace
        brace_count = 1
        pos = brace_start + 1

        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            # Found the matching brace, now look for LIMIT clause
            query_end = pos

            # Check for LIMIT clause after the closing brace
            limit_match = re.search(r'\s*(LIMIT\s+\d+)', text[pos:], re.IGNORECASE)
            if limit_match:
                query_end = pos + limit_match.end()

            # Extract the complete query from start_pos to query_end
            complete_query = text[start_pos:query_end].strip()
            return complete_query

        return None

    def _fallback_generation(self, query: str, entities: List[str]) -> Optional[str]:
        """Fallback to rule-based SPARQL generation.

        Args:
            query: Natural language query
            entities: Detected entities

        Returns:
            Generated SPARQL query or None
        """
        logger.info("Using fallback rule-based SPARQL generation")

        try:
            # Enhanced rule-based generation
            return self._enhanced_rule_based_generation(query, entities)
        except Exception as e:
            logger.error(f"Fallback SPARQL generation failed: {e}")
            return None

    def _enhanced_rule_based_generation(self, query: str, entities: List[str]) -> Optional[str]:
        """Enhanced rule-based SPARQL generation with better patterns.

        Args:
            query: Natural language query
            entities: Detected entities

        Returns:
            Generated SPARQL query or None
        """
        query_lower = query.lower()

        # Capital city queries
        if "capital" in query_lower:
            if entities:
                country = entities[0]  # Assume first entity is the country
                return f"""
                PREFIX ex: <http://example.org/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?capital ?capitalLabel WHERE {{
                    ?capital ex:isCapitalOf ?country .
                    ?country rdfs:label "{country}" .
                    OPTIONAL {{ ?capital rdfs:label ?capitalLabel }}
                }} LIMIT 10
                """

        # Author queries
        if any(word in query_lower for word in ["wrote", "author", "written"]):
            if entities:
                work = entities[0]  # Assume first entity is the work
                return f"""
                PREFIX ex: <http://example.org/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?author ?authorLabel WHERE {{
                    ?work rdfs:label "{work}" .
                    ?work ex:writtenBy ?author .
                    OPTIONAL {{ ?author rdfs:label ?authorLabel }}
                }} LIMIT 10
                """

        # General entity search
        if entities:
            entity = entities[0]
            return f"""
            PREFIX ex: <http://example.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?subject ?predicate ?object WHERE {{
                {{
                    ?subject rdfs:label "{entity}" .
                    ?subject ?predicate ?object
                }}
                UNION
                {{
                    ?object rdfs:label "{entity}" .
                    ?subject ?predicate ?object
                }}
                UNION
                {{
                    ?subject rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(?label), LCASE("{entity}")))
                    ?subject ?predicate ?object
                }}
            }} LIMIT 20
            """

        # Fallback: get all triples with any entity mentions
        if entities:
            entity = entities[0]
            return f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o .
                FILTER(
                    CONTAINS(LCASE(STR(?s)), LCASE("{entity}")) ||
                    CONTAINS(LCASE(STR(?o)), LCASE("{entity}"))
                )
            }} LIMIT 10
            """

        return None

    def generate_exploration_query(self, entities: List[str]) -> Optional[str]:
        """Generate a query to explore the knowledge graph around given entities.

        Args:
            entities: List of entities to explore

        Returns:
            SPARQL query for exploration
        """
        if not entities:
            return """
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o
            } LIMIT 20
            """

        entity = entities[0]
        return f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?s ?p ?o WHERE {{
            {{
                ?s rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{entity}")))
                ?s ?p ?o
            }}
            UNION
            {{
                ?o rdfs:label ?label .
                FILTER(CONTAINS(LCASE(?label), LCASE("{entity}")))
                ?s ?p ?o
            }}
        }} LIMIT 30
        """

    def explain_query_generation(self, query: str, sparql_query: str) -> str:
        """Generate an explanation of how the SPARQL query relates to the natural language query.

        Args:
            query: Original natural language query
            sparql_query: Generated SPARQL query

        Returns:
            Human-readable explanation
        """
        try:
            explanation_prompt = f"""Explain how this SPARQL query answers the natural language question:

Question: "{query}"

SPARQL Query:
{sparql_query}

Provide a brief, clear explanation of what the SPARQL query is looking for and how it relates to the question."""

            explanation = self.llm_client.generate_response([
                {"role": "system", "content": "You explain SPARQL queries in simple terms."},
                {"role": "user", "content": explanation_prompt}
            ])

            return explanation.strip()

        except Exception as e:
            logger.error(f"Query explanation generation failed: {e}")
            return f"Generated SPARQL query to search for information related to: {query}"
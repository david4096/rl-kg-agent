# MCP Integration for RL-KG-Agent

This document describes the Model Context Protocol (MCP) integration in RL-KG-Agent, which enables biomedical literature search and entity annotation capabilities.

## Overview

The MCP integration adds a new action type `QUERY_MCP_THEN_RESPOND` that allows the agent to:

- Annotate biomedical text with entity recognition (via TERMite API)
- Search biomedical literature for relevant sentences (via SciBite Search API)  
- Get comprehensive RAG-based answers (via SciBite Answer API)
- Look up detailed entity information

## Configuration

### Enable MCP in Agent Configuration

Add the following to your `rl_kg_agent_config.json`:

```json
{
  "mcp": {
    "enabled": true,
    "config_file": "configs/mcp_config.json",
    "default_server": "unified_biomedical",
    "connection_timeout": 30.0,
    "retry_attempts": 3,
    "prefer_rag_for_questions": true,
    "enable_biomedical_enhancement": true,
    "auto_detect_biomedical_queries": true,
    "fallback_on_error": true
  }
}
```

### MCP Server Configuration

The `configs/mcp_config.json` file defines available MCP servers and their tools:

```json
{
  "mcp_servers": {
    "unified_biomedical": {
      "name": "Unified Biomedical MCP Server",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/unified_mcp_server.py"]
      },
      "tools": [
        {
          "name": "medline_semantic_search",
          "description": "Semantic search with entity annotation"
        },
        {
          "name": "get_rag_answer", 
          "description": "Comprehensive RAG-based answers"
        },
        {
          "name": "get_entity_details",
          "description": "Detailed entity information"
        }
      ],
      "enabled": true
    }
  }
}
```

## Usage

### Command Line Interface

Enable MCP features with the CLI:

```bash
# Enable MCP for interactive mode
rl-kg-agent interactive --use-mcp --ttl-file my_kg.ttl

# Use custom MCP config file
rl-kg-agent interactive --mcp-config /path/to/mcp_config.json --ttl-file my_kg.ttl

# Training with MCP enabled
rl-kg-agent train --use-mcp --ttl-file my_kg.ttl --dataset biomedical_qa
```

### Programmatic Usage

```python
from rl_kg_agent.config import get_config, get_mcp_manager
from rl_kg_agent.actions.action_manager import ActionManager
from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.utils.llm_client import LLMClient

# Load configuration with MCP enabled
config = get_config()
config.mcp.enabled = True

# Initialize components
kg_loader = KnowledgeGraphLoader("my_kg.ttl")
sparql_generator = SPARQLQueryGenerator()
internal_kg = InternalKnowledgeGraph("internal_kg.pkl")
llm_client = LLMClient()

# Create MCP manager
mcp_manager = get_mcp_manager(config)

# Create action manager with MCP support
action_manager = ActionManager(
    kg_loader, sparql_generator, internal_kg, llm_client, mcp_manager
)

# The agent will now have access to MCP actions
# for biomedical queries
```

## Available MCP Tools

### 1. medline_semantic_search

Performs two-step semantic search:
1. Entity annotation with TERMite API
2. Literature search with SciBite Search API

**Usage:**
- Triggered by: Biomedical terms, research queries
- Returns: Annotated entities + relevant literature sentences

### 2. get_rag_answer

Comprehensive RAG-based question answering using SciBite's native conversation API.

**Usage:**
- Triggered by: Complex biomedical questions
- Returns: AI-generated answers with evidence from literature

### 3. get_entity_details

Look up detailed information about specific biomedical entities.

**Usage:**
- Triggered by: Entity-specific queries
- Returns: Comprehensive entity information

## Action Selection

The agent automatically selects MCP actions based on query characteristics:

- **Biomedical keywords**: gene, protein, disease, drug, medicine, clinical, etc.
- **Research patterns**: "papers about", "studies on", "research on", etc.
- **Question types**: What, how, why questions with biomedical context
- **Entity presence**: Detected biomedical entities

## Server Requirements

The MCP integration requires the unified MCP server to be running:

```bash
# Start the unified MCP server
cd /path/to/dbclshackathon
python unified_mcp_server.py
```

The server provides access to:
- TERMite API for biomedical entity annotation
- SciBite Search API for literature search
- SciBite Answer API for RAG-based Q&A

## Examples

### Basic Biomedical Query

```python
# This will trigger the MCP action
query = "What are the molecular mechanisms of diabetes?"

context = {
    "query": query,
    "entities": [],
    "internal_knowledge": ""
}

# Get action recommendations - MCP action should rank highly
recommendations = action_manager.get_action_recommendations(context)

# Execute the action
result = action_manager.execute_action(ActionType.QUERY_MCP_THEN_RESPOND, context)
print(result.response)
```

### Research Literature Search

```python
# This will use semantic search tool
query = "Find research papers about COVID-19 vaccines"

result = action_manager.execute_action(ActionType.QUERY_MCP_THEN_RESPOND, {
    "query": query,
    "entities": ["COVID-19", "vaccines"],
    "internal_knowledge": ""
})

# Response includes annotated entities and relevant sentences
print(result.response)
```

## Testing

Run the MCP integration tests:

```bash
cd rl-kg-agent
python tests/test_mcp_integration.py
```

Try the interactive demo:

```bash
python examples/mcp_integration_example.py
```

## Troubleshooting

### MCP Server Connection Issues

1. Ensure the unified MCP server is running
2. Check the server path in `configs/mcp_config.json`
3. Verify network connectivity and ports
4. Check server logs for errors

### Configuration Issues

1. Validate MCP config with: `python -m rl_kg_agent.config`
2. Check file paths are absolute and accessible  
3. Ensure JSON syntax is valid
4. Verify MCP is enabled in main config

### Action Selection Issues

1. Check if biomedical keywords are present in queries
2. Verify MCP manager is properly initialized
3. Review action confidence scores
4. Test with known biomedical queries

## Dependencies

The MCP integration requires:

- `asyncio` for async MCP communication
- `json` for configuration management  
- `subprocess` for server process management
- Access to the unified MCP server

No additional Python packages are required beyond the base RL-KG-Agent dependencies.
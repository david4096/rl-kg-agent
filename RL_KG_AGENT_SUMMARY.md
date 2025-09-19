# RL-KG-Agent: Reinforcement Learning for Knowledge Graph Reasoning

## Approach
**Multi-Action Reinforcement Learning Framework** that trains an intelligent agent to optimize question-answering strategies through trial and learning:

- **5 Strategic Actions**: Direct LLM response, Knowledge Graph queries, Planning workflows, Clarifying questions, and Biomedical MCP queries
- **PPO Training**: Agent learns optimal action selection based on semantic similarity rewards and success feedback
- **Multi-Modal Knowledge**: Combines static RDF knowledge graphs, dynamic internal memory, and external biomedical APIs

## Key Benefits
- **Adaptive Strategy Selection**: Agent learns when to use knowledge graphs vs. direct responses vs. external tools
- **Semantic Reward System**: Training optimizes for answer quality using sentence transformer similarity scoring  
- **Continuous Learning**: Internal knowledge graph grows and improves with each interaction
- **Biomedical Enhancement**: MCP integration provides specialized literature search and entity annotation capabilities

## Expected Outcomes
- **Improved Answer Quality**: 40-60% better semantic similarity scores through optimized action selection
- **Strategic Intelligence**: Agent develops nuanced understanding of when different approaches work best
- **Knowledge Accumulation**: System builds comprehensive internal knowledge base over time
- **Domain Expertise**: Enhanced biomedical question-answering through specialized tool integration

*Training creates an intelligent agent that doesn't just answer questions, but learns the best strategy for each type of query.*
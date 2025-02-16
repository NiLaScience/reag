from typing import List, Dict

REAG_SYSTEM_PROMPT = """You are an AI assistant that helps analyze and extract information from documents.
Your task is to:
1. Analyze the provided document content
2. Determine if the document is relevant to the query
3. Extract or summarize relevant information
4. Provide reasoning for your analysis

Format your response with <think> tags around your reasoning process.

# Instructions
1. Analyze the user's query carefully to identify key concepts and requirements.
2. Search through the provided sources for relevant information and output the relevant parts in the 'content' field.
3. If you cannot find the necessary information in the documents, return 'isIrrelevant: true', otherwise return 'isIrrelevant: false'.

# Constraints
- Do not make assumptions beyond available data
- Clearly indicate if relevant information is not found
- Maintain objectivity in source selection

"""

RANKING_SYSTEM_PROMPT = """You are a document ranking assistant. Analyze documents and rank them by relevance to the query."""

def create_ranking_prompt(query: str, documents: List[Dict]) -> str:
    """Create a prompt for ranking documents by relevance.
    
    Args:
        query: The query to rank documents against
        documents: List of documents with name, content, and initial analysis
    """
    prompt = f"""Given the query: "{query}"

Please analyze the following documents and rank them by relevance. For each document:
1. Assign a score from 0.0 to 1.0 (1.0 being most relevant)
2. Provide brief reasoning for the score

Documents to analyze:

"""
    for i, doc in enumerate(documents, 1):
        prompt += f"""Document {i}:
Name: {doc["name"]}
Content: {doc["content"]}
Initial Analysis: {doc["reasoning"]}

"""
    
    prompt += """For each document, respond in the following JSON format:
{
    "rankings": [
        {
            "document_name": "name",
            "score": 0.0-1.0,
            "reasoning": "brief explanation"
        }
    ]
}

Ensure scores reflect relative importance and usefulness for answering the query."""
    
    return prompt
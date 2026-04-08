#!/usr/bin/env python3
"""
Build POMDP priors from documents using LLM (MockLLM by default).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, field_validator
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# JSON Schema for priors
class PriorsSchema(BaseModel):
    """Schema for POMDP priors JSON."""
    priors_T_alpha: Optional[Dict[str, List[List[float]]]] = Field(
        None, description="Transition priors: action -> state x state matrix"
    )
    priors_Z_alpha: Optional[Dict[str, List[List[float]]]] = Field(
        None, description="Emission priors: action -> state x observation matrix"
    )
    action_costs: Optional[Dict[str, float]] = Field(
        None, description="Cost per action"
    )
    failure_penalty: float = Field(
        -100.0, description="Penalty for failure states"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"failure_prob": 0.5},
        description="Domain-specific thresholds"
    )
    
    @field_validator("priors_T_alpha", "priors_Z_alpha")
    @classmethod
    def validate_alpha_matrices(cls, v):
        if v is not None:
            for action, matrix in v.items():
                matrix_np = np.array(matrix)
                if np.any(matrix_np < 0):
                    raise ValueError(f"Alpha matrix for {action} contains negative values")
        return v


class MockLLM:
    """Mock LLM that returns valid priors JSON template."""
    
    def generate_priors(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate priors JSON from question and retrieved chunks.
        
        Returns a valid template with default/placeholder values.
        """
        logger.info("Using MockLLM to generate priors")
        logger.info(f"Question: {question}")
        logger.info(f"Using {len(retrieved_chunks)} retrieved chunks")
        
        # Return a template priors structure
        # In a real implementation, this would parse the chunks and question
        # to extract actual values
        
        priors = {
            "priors_T_alpha": None,  # Will be filled by user or remain None
            "priors_Z_alpha": None,
            "action_costs": {
                "ignore": 0.0,
                "calibrate": 5.0,
                "repair": 20.0,
            },
            "failure_penalty": -100.0,
            "thresholds": {
                "failure_prob": 0.5,
            },
            "_metadata": {
                "question": question,
                "sources": [chunk.get("source", "unknown") for chunk in retrieved_chunks[:5]],
                "generated_by": "MockLLM",
            },
        }
        
        return priors


class OpenAIClient:
    """OpenAI API client (stub, only used if API key is set)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info("OpenAI client initialized (stub implementation)")
    
    def generate_priors(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate priors using OpenAI API.
        
        This is a stub - in a real implementation, you would:
        1. Format question + chunks as a prompt
        2. Call OpenAI API
        3. Parse JSON response
        4. Validate against schema
        """
        logger.warning("OpenAI client is a stub, returning MockLLM output")
        mock_llm = MockLLM()
        return mock_llm.generate_priors(question, retrieved_chunks)


def retrieve_chunks(
    index_path: str,
    question: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks from index using TF-IDF.
    
    Args:
        index_path: Path to index JSON file
        question: Query question
        top_k: Number of chunks to retrieve
        
    Returns:
        List of chunk dicts
    """
    with open(index_path, "r") as f:
        index = json.load(f)
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Reconstruct vectorizer (simplified - in practice, use pickle)
    vocab = index["vectorizer"]["vocabulary"]
    idf = np.array(index["vectorizer"]["idf"])
    
    # Build TF-IDF matrix for query
    # Use CountVectorizer to get term frequencies, then multiply by IDF
    # Note: This is simplified - in production you'd use pickle to save/load the fitted vectorizer
    count_vectorizer = CountVectorizer(vocabulary=vocab)
    query_counts = count_vectorizer.transform([question]).toarray()[0]
    
    # Compute TF-IDF: tf * idf (l2 normalized)
    query_tfidf = query_counts * idf
    # L2 normalize
    norm = np.linalg.norm(query_tfidf)
    if norm > 0:
        query_tfidf = query_tfidf / norm
    query_vec = query_tfidf.reshape(1, -1)
    
    # Compute similarity
    chunks_tfidf = np.array(index["tfidf_matrix"])
    # Normalize chunks_tfidf (should already be normalized, but ensure)
    chunks_norms = np.linalg.norm(chunks_tfidf, axis=1, keepdims=True)
    chunks_norms[chunks_norms == 0] = 1.0
    chunks_tfidf_norm = chunks_tfidf / chunks_norms
    
    similarities = cosine_similarity(query_vec, chunks_tfidf_norm)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return chunks
    retrieved = []
    for idx in top_indices:
        chunk = index["chunks"][idx].copy()
        chunk["similarity"] = float(similarities[idx])
        retrieved.append(chunk)
    
    return retrieved


def main():
    parser = argparse.ArgumentParser(description="Build POMDP priors from documents")
    parser.add_argument("--question", type=str, required=True,
                       help="Question/prompt for extracting priors")
    parser.add_argument("--index", type=str, default="artifacts/knowledge/index.jsonl",
                       help="Path to knowledge index")
    parser.add_argument("--out", type=str, default="configs/pomdp_priors.json",
                       help="Output path for priors JSON")
    parser.add_argument("--enable-openai", action="store_true",
                       help="Enable OpenAI API (requires OPENAI_API_KEY)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Retrieve relevant chunks
    if not Path(args.index).exists():
        logger.warning(f"Index file not found: {args.index}, skipping retrieval")
        retrieved_chunks = []
    else:
        logger.info(f"Retrieving top-{args.top_k} chunks from {args.index}")
        try:
            retrieved_chunks = retrieve_chunks(args.index, args.question, top_k=args.top_k)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            retrieved_chunks = []
    
    # Choose LLM
    if args.enable_openai and os.getenv("OPENAI_API_KEY"):
        logger.info("Using OpenAI API")
        llm = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        if args.enable_openai:
            logger.warning("--enable-openai specified but OPENAI_API_KEY not set, using MockLLM")
        llm = MockLLM()
    
    # Generate priors
    logger.info("Generating priors from question and retrieved chunks")
    priors_dict = llm.generate_priors(args.question, retrieved_chunks)
    
    # Validate against schema
    try:
        validated_priors = PriorsSchema(**priors_dict)
        priors_dict = validated_priors.model_dump(exclude_none=False)
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        logger.info("Saving unvalidated priors anyway")
    
    # Save priors
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(priors_dict, f, indent=2)
    
    logger.info(f"Saved priors to {out_path}")
    print(f"\nPriors saved to: {out_path}")
    print(f"You can now use this file with: python -m scripts.build_pomdp --priors {out_path}")


if __name__ == "__main__":
    main()


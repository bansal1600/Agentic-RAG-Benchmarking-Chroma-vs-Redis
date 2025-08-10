#!/usr/bin/env python3
"""
Working LangSmith Evaluation for Chroma vs Redis
Simplified approach that focuses on core LangSmith integration
"""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example

# Vector store imports
from langchain_chroma import Chroma
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation"""
    vector_store: str
    query: str
    latency_ms: float
    documents_retrieved: int
    relevance_score: float
    success: bool
    error: Optional[str] = None

class LangSmithVectorEvaluator:
    """LangSmith-integrated vector store evaluator"""
    
    def __init__(self):
        # Initialize LangSmith client
        self.client = Client()
        self.embeddings = OpenAIEmbeddings()
        
        # Test queries for evaluation
        self.test_queries = [
            {
                "input": {"query": "What are the types of agent memory?"},
                "expected": {"topics": ["memory", "agent", "types"], "min_docs": 1}
            },
            {
                "input": {"query": "How does context work in agents?"},
                "expected": {"topics": ["context", "agent", "workflow"], "min_docs": 1}
            },
            {
                "input": {"query": "What is memory for agents?"},
                "expected": {"topics": ["memory", "agent", "definition"], "min_docs": 1}
            },
            {
                "input": {"query": "How to update agent memory?"},
                "expected": {"topics": ["memory", "update", "agent"], "min_docs": 1}
            }
        ]
    
    def setup_vector_stores(self):
        """Initialize vector stores"""
        print("🔧 Setting up vector stores...")
        
        # Setup Chroma
        try:
            self.chroma_store = Chroma(
                collection_name="rag-chroma",
                embedding_function=self.embeddings,
                persist_directory="../agentic-RAG-Redis/chroma_db",
            )
            print("✅ Chroma store initialized")
        except Exception as e:
            print(f"❌ Chroma initialization failed: {e}")
            self.chroma_store = None
        
        # Setup Redis
        try:
            self.redis_store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                index_name="rag-redis-index",
                embeddings=self.embeddings,
            )
            print("✅ Redis store initialized")
        except Exception as e:
            print(f"❌ Redis initialization failed: {e}")
            self.redis_store = None
    
    def create_langsmith_dataset(self) -> str:
        """Create LangSmith dataset for evaluation"""
        dataset_name = f"vector-store-comparison-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📊 Creating LangSmith dataset: {dataset_name}")
        
        # Create dataset
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description="Vector store comparison dataset for Chroma vs Redis"
        )
        
        # Add examples
        inputs = [item["input"] for item in self.test_queries]
        outputs = [item["expected"] for item in self.test_queries]
        
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id
        )
        
        print(f"✅ Dataset created with {len(self.test_queries)} examples")
        return dataset_name
    
    def evaluate_chroma_retrieval(self, example: dict) -> dict:
        """Evaluate Chroma retrieval for LangSmith"""
        if not self.chroma_store:
            return {"error": "Chroma store not available", "latency_ms": 0, "documents": []}
        
        query = example["query"]
        
        try:
            start_time = time.time()
            retriever = self.chroma_store.as_retriever(k=3, search_type="similarity")
            documents = retriever.invoke(query)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            return {
                "documents": [doc.page_content for doc in documents],
                "latency_ms": latency_ms,
                "num_documents": len(documents),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0,
                "documents": [],
                "num_documents": 0,
                "success": False
            }
    
    def evaluate_redis_retrieval(self, example: dict) -> dict:
        """Evaluate Redis retrieval for LangSmith"""
        if not self.redis_store:
            return {"error": "Redis store not available", "latency_ms": 0, "documents": []}
        
        query = example["query"]
        
        try:
            start_time = time.time()
            retriever = self.redis_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            documents = retriever.invoke(query)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            return {
                "documents": [doc.page_content for doc in documents],
                "latency_ms": latency_ms,
                "num_documents": len(documents),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0,
                "documents": [],
                "num_documents": 0,
                "success": False
            }
    
    def relevance_evaluator(self, run, example) -> dict:
        """Custom relevance evaluator for LangSmith"""
        try:
            # Get the expected topics from the example
            expected_topics = example.outputs.get("topics", [])
            min_docs = example.outputs.get("min_docs", 1)
            
            # Get the retrieved documents from the run output
            documents = run.outputs.get("documents", [])
            num_documents = run.outputs.get("num_documents", 0)
            success = run.outputs.get("success", False)
            
            if not success:
                return {
                    "key": "relevance",
                    "score": 0.0,
                    "reason": "Retrieval failed"
                }
            
            # Check if minimum documents were retrieved
            if num_documents < min_docs:
                return {
                    "key": "relevance",
                    "score": 0.2,
                    "reason": f"Only {num_documents} documents retrieved, expected at least {min_docs}"
                }
            
            # Calculate topic coverage
            combined_content = " ".join(documents).lower()
            topics_found = sum(1 for topic in expected_topics if topic.lower() in combined_content)
            relevance_score = topics_found / len(expected_topics) if expected_topics else 0.5
            
            return {
                "key": "relevance",
                "score": relevance_score,
                "reason": f"Found {topics_found}/{len(expected_topics)} expected topics"
            }
            
        except Exception as e:
            return {
                "key": "relevance",
                "score": 0.0,
                "reason": f"Evaluation error: {str(e)}"
            }
    
    def latency_evaluator(self, run, example) -> dict:
        """Custom latency evaluator for LangSmith"""
        try:
            latency_ms = run.outputs.get("latency_ms", float('inf'))
            success = run.outputs.get("success", False)
            
            if not success:
                return {
                    "key": "latency",
                    "score": 0.0,
                    "reason": "Retrieval failed"
                }
            
            # Score based on latency (lower is better)
            # Excellent: < 200ms, Good: < 500ms, Fair: < 1000ms, Poor: >= 1000ms
            if latency_ms < 200:
                score = 1.0
                grade = "Excellent"
            elif latency_ms < 500:
                score = 0.8
                grade = "Good"
            elif latency_ms < 1000:
                score = 0.6
                grade = "Fair"
            else:
                score = 0.3
                grade = "Poor"
            
            return {
                "key": "latency",
                "score": score,
                "reason": f"{latency_ms:.1f}ms - {grade}"
            }
            
        except Exception as e:
            return {
                "key": "latency",
                "score": 0.0,
                "reason": f"Evaluation error: {str(e)}"
            }
    
    def run_langsmith_evaluation(self, dataset_name: str):
        """Run LangSmith evaluation for both vector stores"""
        print("🚀 Running LangSmith evaluation...")
        
        # Custom evaluators
        evaluators = [self.relevance_evaluator, self.latency_evaluator]
        
        results = {}
        
        # Evaluate Chroma
        if self.chroma_store:
            print("📊 Evaluating Chroma with LangSmith...")
            chroma_results = evaluate(
                self.evaluate_chroma_retrieval,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="chroma-vector-store",
                description="Chroma vector store evaluation",
                metadata={"vector_store": "chroma", "version": "1.0"}
            )
            results["chroma"] = chroma_results
            print(f"✅ Chroma evaluation completed: {chroma_results}")
        
        # Evaluate Redis
        if self.redis_store:
            print("📊 Evaluating Redis with LangSmith...")
            redis_results = evaluate(
                self.evaluate_redis_retrieval,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="redis-vector-store",
                description="Redis vector store evaluation",
                metadata={"vector_store": "redis", "version": "1.0"}
            )
            results["redis"] = redis_results
            print(f"✅ Redis evaluation completed: {redis_results}")
        
        return results
    
    def analyze_results(self, results: dict):
        """Analyze and compare LangSmith evaluation results"""
        print("\n" + "="*80)
        print("📊 LANGSMITH EVALUATION RESULTS")
        print("="*80)
        
        for store_name, result in results.items():
            if result:
                print(f"\n🔍 {store_name.upper()} RESULTS:")
                print(f"  Experiment: {result}")
                # Note: The actual metrics would be available in the LangSmith dashboard
                print(f"  Check LangSmith dashboard for detailed metrics and traces")
        
        print(f"\n🔗 View detailed results in LangSmith dashboard:")
        print(f"   https://smith.langchain.com/")
        
        return results

def main():
    """Main evaluation execution"""
    print("🔬 LangSmith Vector Store Evaluation")
    print("=" * 50)
    
    # Check for LangSmith API key
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ LANGSMITH_API_KEY environment variable not set!")
        print("Please set your LangSmith API key to use this evaluation.")
        return
    
    evaluator = LangSmithVectorEvaluator()
    evaluator.setup_vector_stores()
    
    # Create evaluation dataset
    dataset_name = evaluator.create_langsmith_dataset()
    
    # Run LangSmith evaluation
    results = evaluator.run_langsmith_evaluation(dataset_name)
    
    # Analyze results
    evaluator.analyze_results(results)
    
    print(f"\n✅ LangSmith evaluation completed!")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔗 View results at: https://smith.langchain.com/")

if __name__ == "__main__":
    main()

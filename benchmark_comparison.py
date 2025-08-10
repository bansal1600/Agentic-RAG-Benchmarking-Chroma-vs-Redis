#!/usr/bin/env python3
"""
Comprehensive Benchmarking Framework: Chroma vs Redis Vector Stores
Integrates with LangSmith for detailed tracing and evaluation
"""

import time
import statistics
import json
import asyncio
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Vector store imports
from langchain_chroma import Chroma
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment
from dotenv import load_dotenv
load_dotenv()

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    vector_store: str
    query: str
    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    documents_retrieved: int
    relevance_scores: List[float]
    timestamp: str
    error: str = None

@dataclass
class AggregatedResults:
    """Aggregated benchmark results"""
    vector_store: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_memory_mb: float
    avg_cpu_percent: float
    total_queries: int
    error_rate: float
    avg_relevance_score: float

class VectorStoreBenchmark:
    """Comprehensive benchmarking framework for vector stores"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.langsmith_client = Client()
        
        # Initialize vector stores
        self.chroma_store = None
        self.redis_store = None
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the types of agent memory?",
            "How does context work in agents?",
            "What is memory for agents?",
            "How to update agent memory?",
            "What are the benefits of agent memory?",
            "How do agents use memory in decision making?",
            "What is the difference between short-term and long-term memory?",
            "How does memory affect agent performance?",
            "What are memory management strategies for agents?",
            "How do agents store and retrieve memories?"
        ]
        
        # Ground truth relevance (manually curated for evaluation)
        self.ground_truth = {
            "What are the types of agent memory?": ["memory", "types", "agent", "classification"],
            "How does context work in agents?": ["context", "agent", "workflow", "execution"],
            "What is memory for agents?": ["memory", "agent", "definition", "purpose"],
            # Add more ground truth as needed
        }
    
    def setup_vector_stores(self):
        """Initialize both vector stores"""
        print("🔧 Setting up vector stores...")
        
        # Setup Chroma
        try:
            self.chroma_store = Chroma(
                collection_name="rag-chroma",
                embedding_function=self.embeddings,
                persist_directory="../adaptive-RAG/chroma_db",
            )
            print("✅ Chroma store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Chroma: {e}")
        
        # Setup Redis
        try:
            self.redis_store = RedisVectorStore(
                redis_url="redis://localhost:6379",
                index_name="rag-redis-index",
                embeddings=self.embeddings,
            )
            print("✅ Redis store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Redis: {e}")
    
    def measure_system_resources(self) -> Tuple[float, float]:
        """Measure current system resource usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        return memory_mb, cpu_percent
    
    def calculate_relevance_score(self, query: str, documents: List[Document]) -> List[float]:
        """Calculate relevance scores for retrieved documents"""
        if query not in self.ground_truth:
            return [0.5] * len(documents)  # Default score if no ground truth
        
        ground_truth_terms = self.ground_truth[query]
        scores = []
        
        for doc in documents:
            content = doc.page_content.lower()
            matches = sum(1 for term in ground_truth_terms if term.lower() in content)
            relevance = matches / len(ground_truth_terms)
            scores.append(relevance)
        
        return scores
    
    def benchmark_single_query(self, vector_store, store_name: str, query: str) -> BenchmarkResult:
        """Benchmark a single query against a vector store"""
        try:
            # Measure initial resources
            memory_before, cpu_before = self.measure_system_resources()
            
            # Execute query with timing
            start_time = time.time()
            
            if store_name == "chroma":
                retriever = vector_store.as_retriever(k=2, search_type="similarity")
                documents = retriever.get_relevant_documents(query)
            else:  # redis
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                )
                documents = retriever.get_relevant_documents(query)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Measure final resources
            memory_after, cpu_after = self.measure_system_resources()
            
            # Calculate relevance scores
            relevance_scores = self.calculate_relevance_score(query, documents)
            
            return BenchmarkResult(
                vector_store=store_name,
                query=query,
                latency_ms=latency_ms,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_after - cpu_before,
                documents_retrieved=len(documents),
                relevance_scores=relevance_scores,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return BenchmarkResult(
                vector_store=store_name,
                query=query,
                latency_ms=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                documents_retrieved=0,
                relevance_scores=[],
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def run_benchmark_suite(self, iterations: int = 3) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark suite"""
        print(f"🚀 Running benchmark suite with {iterations} iterations...")
        
        results = {"chroma": [], "redis": []}
        
        # Test each vector store
        for store_name, store in [("chroma", self.chroma_store), ("redis", self.redis_store)]:
            if store is None:
                print(f"⚠️ Skipping {store_name} - not initialized")
                continue
                
            print(f"\n📊 Benchmarking {store_name.upper()}...")
            
            for iteration in range(iterations):
                print(f"  Iteration {iteration + 1}/{iterations}")
                
                for query in self.test_queries:
                    result = self.benchmark_single_query(store, store_name, query)
                    results[store_name].append(result)
                    
                    if result.error:
                        print(f"    ❌ Error with query '{query[:30]}...': {result.error}")
                    else:
                        print(f"    ✅ Query '{query[:30]}...' - {result.latency_ms:.1f}ms")
        
        return results
    
    def run_concurrent_benchmark(self, concurrent_users: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """Run concurrent load testing"""
        print(f"⚡ Running concurrent benchmark with {concurrent_users} users...")
        
        results = {"chroma": [], "redis": []}
        
        for store_name, store in [("chroma", self.chroma_store), ("redis", self.redis_store)]:
            if store is None:
                continue
                
            print(f"\n🔄 Load testing {store_name.upper()}...")
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                
                # Submit concurrent queries
                for i in range(concurrent_users):
                    query = self.test_queries[i % len(self.test_queries)]
                    future = executor.submit(self.benchmark_single_query, store, store_name, query)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    result = future.result()
                    results[store_name].append(result)
        
        return results
    
    def aggregate_results(self, results: List[BenchmarkResult]) -> AggregatedResults:
        """Aggregate benchmark results"""
        if not results:
            return None
        
        valid_results = [r for r in results if r.error is None]
        if not valid_results:
            return None
        
        latencies = [r.latency_ms for r in valid_results]
        relevance_scores = [score for r in valid_results for score in r.relevance_scores]
        
        return AggregatedResults(
            vector_store=valid_results[0].vector_store,
            avg_latency_ms=statistics.mean(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
            p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
            avg_memory_mb=statistics.mean([r.memory_usage_mb for r in valid_results]),
            avg_cpu_percent=statistics.mean([r.cpu_usage_percent for r in valid_results]),
            total_queries=len(valid_results),
            error_rate=(len(results) - len(valid_results)) / len(results),
            avg_relevance_score=statistics.mean(relevance_scores) if relevance_scores else 0.0
        )
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Convert to serializable format
        serializable_results = {}
        for store_name, store_results in results.items():
            serializable_results[store_name] = [asdict(result) for result in store_results]
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
    
    def print_comparison_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Print detailed comparison report"""
        print("\n" + "="*80)
        print("📊 CHROMA vs REDIS PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        for store_name, store_results in results.items():
            if not store_results:
                continue
                
            aggregated = self.aggregate_results(store_results)
            if not aggregated:
                continue
            
            print(f"\n🔍 {store_name.upper()} RESULTS:")
            print(f"  Average Latency: {aggregated.avg_latency_ms:.2f}ms")
            print(f"  P95 Latency: {aggregated.p95_latency_ms:.2f}ms")
            print(f"  P99 Latency: {aggregated.p99_latency_ms:.2f}ms")
            print(f"  Average Memory Usage: {aggregated.avg_memory_mb:.2f}MB")
            print(f"  Average CPU Usage: {aggregated.avg_cpu_percent:.2f}%")
            print(f"  Total Queries: {aggregated.total_queries}")
            print(f"  Error Rate: {aggregated.error_rate:.2%}")
            print(f"  Average Relevance Score: {aggregated.avg_relevance_score:.3f}")
        
        # Performance comparison
        if "chroma" in results and "redis" in results:
            chroma_agg = self.aggregate_results(results["chroma"])
            redis_agg = self.aggregate_results(results["redis"])
            
            if chroma_agg and redis_agg:
                print(f"\n⚡ PERFORMANCE COMPARISON:")
                latency_improvement = ((chroma_agg.avg_latency_ms - redis_agg.avg_latency_ms) / chroma_agg.avg_latency_ms) * 100
                print(f"  Latency Improvement (Redis vs Chroma): {latency_improvement:+.1f}%")
                
                relevance_diff = redis_agg.avg_relevance_score - chroma_agg.avg_relevance_score
                print(f"  Relevance Score Difference: {relevance_diff:+.3f}")
                
                print(f"\n🏆 WINNER:")
                if redis_agg.avg_latency_ms < chroma_agg.avg_latency_ms:
                    print(f"  Speed: Redis ({redis_agg.avg_latency_ms:.1f}ms vs {chroma_agg.avg_latency_ms:.1f}ms)")
                else:
                    print(f"  Speed: Chroma ({chroma_agg.avg_latency_ms:.1f}ms vs {redis_agg.avg_latency_ms:.1f}ms)")
                
                if redis_agg.avg_relevance_score > chroma_agg.avg_relevance_score:
                    print(f"  Quality: Redis ({redis_agg.avg_relevance_score:.3f} vs {chroma_agg.avg_relevance_score:.3f})")
                else:
                    print(f"  Quality: Chroma ({chroma_agg.avg_relevance_score:.3f} vs {redis_agg.avg_relevance_score:.3f})")

def main():
    """Main benchmarking execution"""
    print("🔬 Vector Store Benchmark Suite")
    print("=" * 50)
    
    benchmark = VectorStoreBenchmark()
    benchmark.setup_vector_stores()
    
    # Run standard benchmark
    print("\n1️⃣ Running standard benchmark...")
    standard_results = benchmark.run_benchmark_suite(iterations=3)
    
    # Run concurrent benchmark
    print("\n2️⃣ Running concurrent load test...")
    concurrent_results = benchmark.run_concurrent_benchmark(concurrent_users=5)
    
    # Combine results
    all_results = {}
    for store_name in ["chroma", "redis"]:
        all_results[store_name] = standard_results.get(store_name, []) + concurrent_results.get(store_name, [])
    
    # Generate reports
    benchmark.print_comparison_report(all_results)
    benchmark.save_results(all_results)
    
    print(f"\n✅ Benchmark completed! Check the generated JSON file for detailed results.")

if __name__ == "__main__":
    main()

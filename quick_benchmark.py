#!/usr/bin/env python3
"""
Quick benchmark demo to compare Chroma vs Redis performance
"""

import time
import statistics
from langchain_chroma import Chroma
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def quick_benchmark():
    """Run a quick performance comparison"""
    print("🔬 Quick Benchmark: Chroma vs Redis")
    print("=" * 50)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Test queries
    test_queries = [
        "What are the types of agent memory?",
        "How does context work in agents?",
        "What is memory for agents?"
    ]
    
    results = {}
    
    # Test Chroma
    try:
        print("\n📊 Testing Chroma...")
        chroma_store = Chroma(
            collection_name="rag-chroma",
            embedding_function=embeddings,
            persist_directory="../agentic-RAG-Redis/chroma_db",
        )
        chroma_retriever = chroma_store.as_retriever(k=2, search_type="similarity")
        
        chroma_times = []
        for query in test_queries:
            start_time = time.time()
            docs = chroma_retriever.get_relevant_documents(query)
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            chroma_times.append(latency)
            print(f"  ✅ Query: {latency:.1f}ms, {len(docs)} docs")
        
        results["chroma"] = {
            "avg_latency": statistics.mean(chroma_times),
            "min_latency": min(chroma_times),
            "max_latency": max(chroma_times),
            "total_queries": len(chroma_times)
        }
        
    except Exception as e:
        print(f"  ❌ Chroma error: {e}")
        results["chroma"] = None
    
    # Test Redis
    try:
        print("\n📊 Testing Redis...")
        redis_store = RedisVectorStore(
            redis_url="redis://localhost:6379",
            index_name="rag-redis-index",
            embeddings=embeddings,
        )
        redis_retriever = redis_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        redis_times = []
        for query in test_queries:
            start_time = time.time()
            docs = redis_retriever.get_relevant_documents(query)
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            redis_times.append(latency)
            print(f"  ✅ Query: {latency:.1f}ms, {len(docs)} docs")
        
        results["redis"] = {
            "avg_latency": statistics.mean(redis_times),
            "min_latency": min(redis_times),
            "max_latency": max(redis_times),
            "total_queries": len(redis_times)
        }
        
    except Exception as e:
        print(f"  ❌ Redis error: {e}")
        results["redis"] = None
    
    # Print comparison
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    if results["chroma"] and results["redis"]:
        chroma_avg = results["chroma"]["avg_latency"]
        redis_avg = results["redis"]["avg_latency"]
        
        print(f"\n🔍 CHROMA RESULTS:")
        print(f"  Average Latency: {chroma_avg:.2f}ms")
        print(f"  Min Latency: {results['chroma']['min_latency']:.2f}ms")
        print(f"  Max Latency: {results['chroma']['max_latency']:.2f}ms")
        
        print(f"\n🔍 REDIS RESULTS:")
        print(f"  Average Latency: {redis_avg:.2f}ms")
        print(f"  Min Latency: {results['redis']['min_latency']:.2f}ms")
        print(f"  Max Latency: {results['redis']['max_latency']:.2f}ms")
        
        improvement = ((chroma_avg - redis_avg) / chroma_avg) * 100
        print(f"\n⚡ PERFORMANCE IMPROVEMENT:")
        print(f"  Redis vs Chroma: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"  🏆 Redis is {improvement:.1f}% faster than Chroma!")
        else:
            print(f"  🏆 Chroma is {abs(improvement):.1f}% faster than Redis!")
    
    else:
        print("❌ Could not complete comparison - check vector store setup")

if __name__ == "__main__":
    quick_benchmark()

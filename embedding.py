##########ONE TIME RUN!!!! to create a Redis Vector DB from web pages##########

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
import os
import redis
from dotenv import load_dotenv
load_dotenv()

# Use the environment variable if set, otherwise default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")

# Redis connection
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Successfully connected to Redis!")
except redis.ConnectionError:
    print("❌ Failed to connect to Redis. Make sure Redis is running.")
    print("Run: brew services start redis")
    exit(1)

# Check if Redis index already exists
index_name = "rag-redis-index"
try:
    # Try to get info about existing index
    redis_client.ft(index_name).info()
    print(f"Loading existing Redis Vector DB with index: {index_name}")
    
    vectorstore = RedisVectorStore(
        redis_url=REDIS_URL,
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
    )
except:
    print(f"Creating new Redis Vector DB with index: {index_name}")
    
    # Load documents from URLs
    urls = [
        "https://blog.langchain.com/memory-for-agents/",
        "https://langchain-ai.github.io/langgraph/agents/context/",
    ]

    docs_lists = []
    for url in urls:
        print(f"Loading documents from: {url}")
        loader = WebBaseLoader(url)
        docs_lists.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_lists)

    print(f"Number of document chunks: {len(doc_splits)}")

    # Create Redis vector store and add documents
    vectorstore = RedisVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        redis_url=REDIS_URL,
        index_name=index_name,
    )
    
    print(f"✅ Successfully created Redis Vector DB with {len(doc_splits)} documents!")
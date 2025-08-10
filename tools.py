from langchain.tools.retriever import create_retriever_tool
from retriever import retriever
from langgraph.prebuilt import ToolNode

tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [tool]
tool_executor = ToolNode(tools)
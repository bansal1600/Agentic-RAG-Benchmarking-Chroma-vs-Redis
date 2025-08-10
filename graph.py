from langgraph.graph import END, StateGraph
import pprint
from langchain_core.messages import HumanMessage
from state import AgentState
from nodes import agent, retrieve, rewrite, generate
from edges import grade_documents, should_retrieve
from IPython.display import Image, display

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # retrieval
workflow.add_node("generate", generate)  # retrieval    


# Call agent node to decide to retrieve or not
workflow.set_entry_point("agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    should_retrieve,
    {
        # Call tool node
        "continue": "retrieve",
        "end": END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
    {
        "yes": "generate",
        "no": "rewrite",  
    },
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
app = workflow.compile()

inputs = {
    "messages": [
        HumanMessage(
            content="What are the types of agent memory?"
        )
    ]
}
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

for output in app.stream(inputs, stream_mode="values"):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")


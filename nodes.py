from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

from state import AgentState
from tools import tools, tool_executor

def agent(state : AgentState):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response apended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI()
    functions = [convert_to_openai_function(t) for t in tools]
    model = model.bind_tools(functions)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {
        "messages": [response]
    }

def retrieve(state):
    """
    Uses tool to execute retrieval.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with retrieved docs
    """
    print("---EXECUTE RETRIEVAL---")
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]

    # We call the tool_executor and get back a response
    response = tool_executor.invoke({"messages": [last_message]})

    # Extract the actual ToolMessage returned by tool_executor
    tool_message = response["messages"][0]

    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content

    last_message = messages[-1]
    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI()

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.
    
    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with re-phrased question
    """
    
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [HumanMessage(
        content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
    )]

    # Grader
    model = ChatOpenAI()
    response = model.invoke(msg)
    return {"messages": [response]}
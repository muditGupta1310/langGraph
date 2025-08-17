from langgraph.graph import START,END , StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated 
from langchain_core.messages import BaseMessage, HumanMessage

llm = ChatOllama(model='llama3:latest')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {'configurable':{'thread_id':'thread-01'}}


response = chatbot.invoke(
                {'messages':[HumanMessage(content='Hi how are you')]},
                config = CONFIG                
            )





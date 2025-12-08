# Dieser Code ist geschrieben Anlehung an
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
import asyncio
from typing import TypedDict, Annotated, List
import streamlit as st
import os
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END
# Import für Tools
from search_tool import SearchTool
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from operator import add

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-5-mini",
    temperature=0.0)

client = chromadb.PersistentClient(path="./chroma_db")
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jinaai/jina-embeddings-v2-base-de")

collection = client.get_or_create_collection(
        "verfahrenstechnik",
        embedding_function=emb)

search_tool = SearchTool(collection)
TOOLS = [search_tool]
tools_node = ToolNode(TOOLS)
llm_with_tools = llm.bind_tools(TOOLS)
system_prompt=("Du bist ein freundlicher Lern-Assistent. Wenn du das"
               "Such-Tool verwendest, formatiere die Quellenangaben aus den Metadaten (Feld"
               "*metadatas* im zurückgelieferten Objekt des SearchTools"
               "mit nummerierten Referenzen (z.B. [1]) im Text und der entsprechenden Quellenangabe"
               "am Ende (z.B. [1] Kapitel 3. Kuchenfiltration, S. 23)")


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add]


def chat_node(state: GraphState) -> dict:
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
    # AIMessage (könnte Ergebnisse von Tool calls beinhalten)
    ai = llm_with_tools.invoke(msgs)
    return {"messages": [ai]}


graph = StateGraph(GraphState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tools_node)
graph.set_entry_point("chat")
graph.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "chat")
app_graph = graph.compile()

st.title("Lern-Bot")

# Initialisiere Nachrichten
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeige, die Chat-Historie an, falls es eine gibt.
for role, content in st.session_state.messages:
    r = role if role in ("user", "assistant") else "assistant"
    with st.chat_message(r):
        st.write(content)

# RAG-Chat auf Basis von Nutzereingaben
if prompt := st.chat_input("Frag, für mehr Informationen!"):
    st.session_state.messages.append(("user", prompt))
    print(st.session_state.messages)
    content = st.session_state.messages[-1][1]
    with st.chat_message("user"):
        st.write(content)
    history_msgs: List[AnyMessage] = []
    for role, content in st.session_state.messages:
        history_msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    result_state = app_graph.invoke({"messages": history_msgs})
    last_ai = None
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break
    out_text = last_ai.content if last_ai else "(kein Text)"
    st.session_state.messages.append(("assistant", out_text))
    st.rerun()


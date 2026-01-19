# Dieser Code ist geschrieben in Anlehung an
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
#
# Update (2 Modi):
# - Lernmodus: beantwortet Fragen inhaltlich zur gewählten Lehr-/Lerneinheit
# - Prompt-Coach: bewertet User-Prompts nach RAFT + Kontext, gibt Feedback + Verbesserungsvorschlag

from __future__ import annotations

from operator import add
from typing import Annotated, Dict, List, Literal, TypedDict, Optional, Tuple
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage,
    AnyMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Optionales Tool-/RAG-Setup (bei euch aktuell FAQ=False)
from langgraph.prebuilt import ToolNode, tools_condition
import torch
import chromadb
from chromadb.utils import embedding_functions

from message_handler import MessageHandler
from search_tool import SearchTool


load_dotenv()


# -----------------------------
#  App Icon / Avatar (Custom)
#  Lege deine Bilder z.B. unter ./assets/ ab
#  Empfohlene Groessen:
#   - favicon/page_icon: 32-64px (quadratisch)
#   - chat avatar: 128-256px (quadratisch, gern transparent)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("ELISA_ASSETS_DIR", str(BASE_DIR / "assets")))


def _find_asset(filename: str) -> Optional[Path]:
    """Findet ein Asset robust:
    - zuerst in ASSETS_DIR
    - dann im BASE_DIR
    - dann case-insensitive Suche in ASSETS_DIR (Windows-Fehlerquelle)
    """
    candidates = [ASSETS_DIR / filename, BASE_DIR / filename]
    for p in candidates:
        if p.exists():
            return p

    if ASSETS_DIR.exists() and ASSETS_DIR.is_dir():
        target = filename.lower()
        for f in ASSETS_DIR.iterdir():
            if f.is_file() and f.name.lower() == target:
                return f

    return None


def _asset_or_fallback(filename: str, fallback: str) -> str:
    """Gibt einen existierenden Dateipfad (als String) oder ein Fallback (z.B. Emoji) zurueck."""
    p = _find_asset(filename)
    if not p:
        return fallback
    # Streamlit kommt mit Forward-Slashes i.d.R. am besten klar (auch unter Windows)
    return p.resolve().as_posix()


# Du kannst diese Dateinamen frei anpassen:
APP_ICON = _asset_or_fallback("elisa_favicon.png", "🤖")
BOT_AVATAR = _asset_or_fallback("elisa_avatar.png", "🤖")
USER_AVATAR = _asset_or_fallback("user_avatar.png", "🧑")


# Muss vor allen anderen st.* Aufrufen kommen
st.set_page_config(page_title="Elisa", page_icon=APP_ICON)

# Optional: Debug-Ausgabe, wenn Icons nicht gefunden werden
ICON_DEBUG = os.getenv("ELISA_ICON_DEBUG", "0") == "1"
if ICON_DEBUG:
    with st.sidebar.expander("Icon Debug", expanded=True):
        st.write("BASE_DIR:", str(BASE_DIR))
        st.write("ASSETS_DIR:", str(ASSETS_DIR), "| exists:", ASSETS_DIR.exists())
        if ASSETS_DIR.exists() and ASSETS_DIR.is_dir():
            st.write("assets files:", [p.name for p in ASSETS_DIR.iterdir() if p.is_file()])
        st.write("APP_ICON:", APP_ICON)
        st.write("BOT_AVATAR:", BOT_AVATAR)
        st.write("USER_AVATAR:", USER_AVATAR)

FAQ = False
MODEL_NAME = "openai/gpt-5-mini"
MAX_TOKEN = 24000

Mode = Literal["lesson", "coach"]

# -----------------------------
#  Lehrinheiten: Kontextbasis
#  (Hier bitte eure echten Inhalte/Bullets eintragen)
# -----------------------------
LESSON_UNITS: Dict[str, Dict[str, str]] = {
    "einfuehrung": {
        "title": "Einführung",
        "context": "Grundlagen: Was ist KI, wofür eignet sie sich, und welche Grenzen gibt es?",
    },
    "funktionsweise": {
        "title": "Funktionsweise",
        "context": "High-Level: Trainingsdaten, Wahrscheinlichkeiten, Halluzinationen, warum Modelle manchmal falsch liegen.",
    },
    "prompting-1": {
        "title": "Prompting 1",
        "context": "Fokus: Grundlagen. Du lernst, wie du Ziel, Kontext und Rolle so formulierst, dass die KI präziser und nutzbarer antwortet.",
    },
    "prompting-2": {
        "title": "Prompting 2",
        "context": "Fortgeschritten: Beispiele, Constraints, Output-Formate, Iteration, Qualitätskontrolle.",
    },
    "gefahren": {
        "title": "Gefahren",
        "context": "Risiken: Datenschutz, Bias, Desinformation, Urheberrecht, Abhängigkeit. Maßnahmen zur sicheren Nutzung.",
    },
}


def _get_query_param(key: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit-Kompatibilität: st.query_params (neu) vs experimental_get_query_params (alt)."""
    try:
        v = st.query_params.get(key)
        if v is None:
            return default
        # je nach Streamlit-Version kann es str oder list[str] sein
        if isinstance(v, list):
            return v[0] if v else default
        return str(v)
    except Exception:
        try:
            params = st.experimental_get_query_params()
            if key not in params:
                return default
            return params[key][0] if params[key] else default
        except Exception:
            return default


def _sanitize_unit(unit_id: Optional[str]) -> str:
    if not unit_id:
        return "prompting-1"
    unit_id = unit_id.strip().lower()
    return unit_id if unit_id in LESSON_UNITS else "prompting-1"


def build_lesson_system_prompt(unit_id: str) -> str:
    unit = LESSON_UNITS.get(unit_id, LESSON_UNITS["prompting-1"])
    return (
        "Du bist 'Elisa', ein Lernbot für ein E-Learning-Modul zum Umgang mit KI. "
        "Du befindest dich im LERNMODUS.\n\n"
        f"AKTUELLE LEHREINHEIT: {unit['title']}\n"
        f"KONTEXT (nur darauf stützen): {unit['context']}\n\n"
        "Regeln:\n"
        "- Antworte inhaltlich passend zur aktuellen Lehreinheit.\n"
        "- Wenn wichtige Infos fehlen oder etwas nicht im Kontext steht, stelle 1-2 Rückfragen oder sage klar, dass es in dieser Einheit nicht behandelt wird.\n"
        "- Gib kurze, praktische Beispiele (wenn sinnvoll).\n"
        "- Schreibe klar, freundlich, knapp.\n"
        "- NICHT nach RAFT bewerten (das ist nur im Prompt-Coach-Modus)."
    )


def build_coach_system_prompt(unit_id: str) -> str:
    unit = LESSON_UNITS.get(unit_id, LESSON_UNITS["prompting-1"])
    return (
        "Du bist 'Elisa', ein Prompt-Coach. Deine Aufgabe: bewerte Nutzer-Prompts nach dem RAFT-Framework und nach Kontextvollständigkeit. "
        "Du befindest dich im PROMPT-COACH-MODUS.\n\n"
        f"AKTUELLE LEHREINHEIT: {unit['title']}\n"
        f"KONTEXT (hilft beim Bewerten): {unit['context']}\n\n"
        "RAFT (Kriterien):\n"
        "R = Rolle (welche Perspektive/Expertise soll die KI annehmen?)\n"
        "A = Aufgabe (was genau soll getan werden, mit welchem Ziel?)\n"
        "F = Format (wie soll die Ausgabe aussehen: Struktur, Länge, Tabellen/JSON/etc.)\n"
        "T = Tonalität (welcher Stil: sachlich, locker, professionell, etc.)\n\n"
        "Zusätzlich bewerten: Kontext & Constraints (Zielgruppe, Randbedingungen, Beispiele, was vermieden werden soll).\n\n"
        "Output-Regeln (immer in diesem Format):\n"
        "1) Score: X/10\n"
        "2) Feedback: 3-6 Bulletpoints, jeweils konkret (was fehlt/was ist gut) und entlang RAFT + Kontext\n"
        "3) Verbesserter Prompt: ein optimierter Prompt als Textblock\n"
        "4) Nächster Schritt: eine kurze Aufforderung zum erneuten Versuch\n\n"
        "Wichtig:\n"
        "- Wenn der Nutzer KEINEN Prompt liefert, sondern nach Tipps/Erklärung fragt (z.B. 'Was ist RAFT?'), dann NICHT bewerten und KEINEN Score geben. Stattdessen kurz erklären und 1 Beispielprompt anbieten.\n"
        "- Keine langen Aufsätze. Keine Metadiskussion."
    )


# -----------------------------
# Session State init
# -----------------------------
if "bot_mode" not in st.session_state:
    st.session_state.bot_mode = "lesson"  # default

if "lesson_messages" not in st.session_state:
    st.session_state.lesson_messages = []

if "coach_messages" not in st.session_state:
    st.session_state.coach_messages = []


# -----------------------------
# LLM init
# -----------------------------
if "base_llm" not in st.session_state:
    st.session_state.base_llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=MODEL_NAME,
        temperature=0.0,
        streaming=True,
    )

# Optional: Suchtool, falls FAQ-Modus aktiv
if FAQ:
    if "tools_node" not in st.session_state:
        client = chromadb.PersistentClient(path="./chroma_db")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-de",
            device=device,
        )

        collection = client.get_or_create_collection(
            "verfahrenstechnik",
            embedding_function=emb,
        )

        search_tool = SearchTool(collection)
        TOOLS = [search_tool]
        st.session_state.tools_node = ToolNode(TOOLS)
        st.session_state.llm = st.session_state.base_llm.bind_tools(TOOLS)
else:
    st.session_state.llm = st.session_state.base_llm


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    llm: object
    system_prompt: str


def chat_node(state: GraphState) -> dict:
    system_prompt = state.get("system_prompt", "")
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
    llm = state.get("llm")
    ai = llm.invoke(msgs)
    return {"messages": [ai]}


if "app_graph" not in st.session_state:
    graph = StateGraph(GraphState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")

    if FAQ:
        graph.add_node("tools", st.session_state.tools_node)
        graph.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": END})
        graph.add_edge("tools", "chat")
    else:
        graph.add_edge("chat", END)

    st.session_state.app_graph = graph.compile()


# -----------------------------
# UI
# -----------------------------
unit_id = _sanitize_unit(_get_query_param("unit", "prompting-1"))
unit_title = LESSON_UNITS[unit_id]["title"]

mode: Mode = st.session_state.bot_mode
messages_key = "lesson_messages" if mode == "lesson" else "coach_messages"
messages: List[Tuple[str, str]] = st.session_state[messages_key]

# Header + Mode Switch Button
left, right = st.columns([3, 1])
with left:
    st.title("Elisa")
    st.caption(f"Einheit: {unit_title} | Modus: {'Lernmodus' if mode == 'lesson' else 'Prompt-Coach'}")
with right:
    switch_label = "Zu Prompt-Coach wechseln" if mode == "lesson" else "Zu Lernmodus wechseln"
    if st.button(switch_label, use_container_width=True):
        st.session_state.bot_mode = "coach" if mode == "lesson" else "lesson"
        st.rerun()

# Optional: Chat leeren
if st.button("Chat leeren", type="secondary"):
    st.session_state[messages_key] = []
    st.rerun()

# Chat-Historie anzeigen
for role, content in messages:
    r = role if role in ("user", "assistant") else "assistant"
    avatar = USER_AVATAR if r == "user" else BOT_AVATAR
    with st.chat_message(r, avatar=avatar):
        st.write(content)

# Systemprompt je Modus
system_prompt = build_lesson_system_prompt(unit_id) if mode == "lesson" else build_coach_system_prompt(unit_id)

# Eingabe
input_hint = (
    "Stelle eine Frage zur aktuellen Lehreinheit …"
    if mode == "lesson"
    else "Schreibe deinen Prompt, den ich nach RAFT bewerten soll …"
)

if prompt := st.chat_input(input_hint):
    st.session_state[messages_key].append(("user", prompt))

    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)

    # Historie in LangChain Messages umwandeln (inkl. Token-Handling über euren MessageHandler)
    history_msgs = MessageHandler(model=MODEL_NAME.split("/")[-1], max_tokens=MAX_TOKEN)
    for role, content in st.session_state[messages_key]:
        history_msgs.add_message(
            HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        )

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        full_response = ""
        message_placeholder = st.empty()

        for event in st.session_state.app_graph.stream(
            {"messages": history_msgs.get_conversation(), "llm": st.session_state.llm, "system_prompt": system_prompt},
            stream_mode="messages",
        ):
            if isinstance(event[0], AIMessageChunk):
                chunk_content = event[0].content
                if chunk_content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + " ")

        message_placeholder.markdown(full_response)

    st.session_state[messages_key].append(("assistant", full_response))
    st.rerun()

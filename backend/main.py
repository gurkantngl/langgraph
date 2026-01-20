import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

load_dotenv()

app = FastAPI(title="Support Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent / "data"

KB_DEFAULT = [
    {
        "id": "kb-refund-1",
        "title": "Iade sureci ve kosullari",
        "category": "refund",
        "body": "14 gun icinde iade hakki bulunur. Urun hasarsiz olmalidir.",
    },
    {
        "id": "kb-payment-1",
        "title": "Odeme hatalari ve cozumleri",
        "category": "payment",
        "body": "Kart limiti, banka onayi ve 3D Secure adimlari kontrol edilir.",
    },
    {
        "id": "kb-shipping-1",
        "title": "Kargo ve teslimat bilgilendirmesi",
        "category": "shipping",
        "body": "Kargo cikisi 1-3 is gunu. Takip kodu e-posta ile paylasilir.",
    },
    {
        "id": "kb-general-1",
        "title": "Sik sorulan sorular",
        "category": "general",
        "body": "Hesap, siparis ve teslimatla ilgili temel bilgiler.",
    },
]

TICKETS_DEFAULT = [
    {
        "id": "t-1001",
        "message": "Siparisim hala kargoya verilmedi, acil yardim rica ederim.",
    },
    {
        "id": "t-1002",
        "message": "Iade sureci nasil isliyor? Urunu geri gondermek istiyorum.",
    },
]


def load_json_data(filename: str, fallback: list) -> list:
    path = DATA_DIR / filename
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


KB = load_json_data("kb.json", KB_DEFAULT)
TICKETS = load_json_data("tickets.json", TICKETS_DEFAULT)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_gemini_model():
    if genai is None:
        return None
    if not GEMINI_API_KEY:
        return None
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)


GEMINI_MODEL = get_gemini_model()
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")


def get_delay_ms(env_key: str, fallback: float) -> float:
    value = os.getenv(env_key)
    if value is None or value.strip() == "":
        return fallback
    try:
        return max(0.0, float(value))
    except ValueError:
        return fallback


def get_delay_map() -> dict:
    return {
        "refund": get_delay_ms("TOOL_DELAY_REFUND_MS", 40),
        "payment": get_delay_ms("TOOL_DELAY_PAYMENT_MS", 30),
        "shipping": get_delay_ms("TOOL_DELAY_SHIPPING_MS", 25),
        "general": get_delay_ms("TOOL_DELAY_GENERAL_MS", 20),
    }


def get_top_k() -> int:
    value = os.getenv("RAG_TOP_K", "3")
    try:
        return max(1, int(value))
    except ValueError:
        return 3


class TicketRequest(BaseModel):
    message: str


class AgentState(TypedDict, total=False):
    message: str
    request_id: str
    trace_id: str
    trace_nodes: List[str]
    category: str
    priority: str
    sentiment: str
    plan: List[str]
    sources: List[dict]
    response: str
    tool_logs: List[dict]


KB_TEXTS = [f"{item['title']}. {item['body']}" for item in KB]
KB_INDEX = None
KB_VECTORS = None
LATEST_TRACE: dict | None = None


def embed_texts(texts: list[str], task_type: str) -> "np.ndarray | None":
    if genai is None or np is None or faiss is None or not GEMINI_API_KEY:
        return None
    model_name = GEMINI_EMBED_MODEL
    if not model_name.startswith(("models/", "tunedModels/")):
        model_name = f"models/{model_name}"
    vectors = []
    for text in texts:
        try:
            result = genai.embed_content(
                model=model_name, content=text, task_type=task_type
            )
            vector = np.array(result["embedding"], dtype="float32")
            vectors.append(vector)
        except Exception:
            return None
    return np.vstack(vectors)


def build_kb_index():
    global KB_INDEX, KB_VECTORS
    if faiss is None or np is None:
        return
    embeddings = embed_texts(KB_TEXTS, "retrieval_document")
    if embeddings is None:
        return
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    KB_INDEX = index
    KB_VECTORS = embeddings


build_kb_index()


def classify_text(message: str) -> dict:
    normalized = message.strip().lower()
    if any(word in normalized for word in ["iade", "refund", "return"]):
        category = "refund"
    elif any(word in normalized for word in ["odeme", "payment", "kart"]):
        category = "payment"
    elif any(word in normalized for word in ["kargo", "shipping", "teslimat"]):
        category = "shipping"
    else:
        category = "general"

    if any(word in normalized for word in ["acil", "hemen", "iptal"]):
        priority = "high"
    else:
        priority = "normal"

    if any(word in normalized for word in ["kotu", "sikayet", "magdur"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
    }


def extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def llm_classify(message: str) -> dict | None:
    if GEMINI_MODEL is None:
        return None
    prompt = (
        "Asagidaki destek mesajini siniflandir.\n"
        "JSON formatinda yanit ver:\n"
        '{ "category": "refund|payment|shipping|general",'
        ' "priority": "high|normal", "sentiment": "negative|neutral" }\n'
        f"Mesaj: {message}"
    )
    try:
        response = GEMINI_MODEL.generate_content(prompt)
    except Exception:
        return None
    parsed = extract_json(response.text or "")
    if not parsed:
        return None
    if parsed.get("category") not in {"refund", "payment", "shipping", "general"}:
        return None
    if parsed.get("priority") not in {"high", "normal"}:
        parsed["priority"] = "normal"
    if parsed.get("sentiment") not in {"negative", "neutral"}:
        parsed["sentiment"] = "neutral"
    return parsed


def llm_plan(message: str, category: str, sources: list[dict]) -> list | None:
    if GEMINI_MODEL is None:
        return None
    sources_text = "\n".join(f"- {item['title']}: {item['body']}" for item in sources)
    prompt = (
        "Bir destek ajani olarak 3-5 adimlik plan olustur.\n"
        "Sadece JSON array dondur (ornek: [\"adim1\", \"adim2\"]).\n"
        f"Kategori: {category}\n"
        f"Mesaj: {message}\n"
        f"Kaynaklar:\n{sources_text}"
    )
    try:
        response = GEMINI_MODEL.generate_content(prompt)
    except Exception:
        return None
    text = response.text or ""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        plan = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, list):
        return None
    return [str(item) for item in plan if str(item).strip()]


def llm_response(
    message: str, category: str, priority: str, plan: list, sources: list, tool_logs: list
) -> str | None:
    if GEMINI_MODEL is None:
        return None
    sources_text = "\n".join(f"- {item['title']}: {item['body']}" for item in sources)
    tools_text = "\n".join(
        f"- {item['tool']} ({item['status']}): {item['detail']}" for item in tool_logs
    )
    plan_text = "\n".join(f"- {item}" for item in plan)
    prompt = (
        "Turkce, nazik ve net bir destek yaniti yaz.\n"
        "Kisa ve eyleme yonelik olsun. Madde listesi kullanabilirsin.\n"
        f"Mesaj: {message}\n"
        f"Kategori: {category}\n"
        f"Oncelik: {priority}\n"
        f"Plan:\n{plan_text}\n"
        f"Tool loglari:\n{tools_text}\n"
        f"Kaynaklar:\n{sources_text}"
    )
    try:
        response = GEMINI_MODEL.generate_content(prompt)
    except Exception:
        return None
    return (response.text or "").strip() or None


def node_classify(state: AgentState) -> AgentState:
    details = llm_classify(state["message"]) or classify_text(state["message"])
    return {**details}


def node_retrieve(state: AgentState) -> AgentState:
    message = state["message"].lower()
    if KB_INDEX is not None and np is not None:
        query_vec = embed_texts([message], "retrieval_query")
        if query_vec is not None:
            norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
            query_vec = query_vec / np.clip(norms, 1e-12, None)
            top_k = min(get_top_k(), len(KB))
            scores, indices = KB_INDEX.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(KB):
                    continue
                item = dict(KB[idx])
                item["score"] = float(score)
                results.append(item)
            if results:
                return {"sources": results}

    category = state["category"]
    candidates = [item for item in KB if item["category"] == category]
    if not candidates:
        candidates = [item for item in KB if item["category"] == "general"]
    if any(word in message for word in ["kargo", "teslimat"]):
        candidates = [item for item in KB if item["category"] == "shipping"]
    return {"sources": candidates[:3]}


def node_plan(state: AgentState) -> AgentState:
    category = state["category"]
    llm_plan_result = llm_plan(state["message"], category, state.get("sources", []))
    if llm_plan_result:
        return {"plan": llm_plan_result}
    if category == "refund":
        plan = [
            "Siparis bilgilerini dogrula",
            "Iade kosullarini kontrol et",
            "Iade talebi kaydi olustur",
        ]
    elif category == "payment":
        plan = [
            "Odeme adimini yeniden denetle",
            "Banka onayi ve 3D Secure kontrolu",
            "Basarisiz odeme icin tekrar dene bilgisi ver",
        ]
    elif category == "shipping":
        plan = [
            "Siparis durumunu kontrol et",
            "Kargo cikis tarihini paylas",
            "Takip kodu varsa ilet",
        ]
    else:
        plan = [
            "Talebi dogru kategoriye esle",
            "Gerekli bilgileri iste",
            "Cozum adimlarini paylas",
        ]
    return {"plan": plan}


def node_tools(state: AgentState) -> AgentState:
    category = state["category"]
    logs = []
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    request_id = state.get("request_id", "unknown")
    delay_map = get_delay_map()

    def make_log(tool: str, status: str, detail: str, delay_ms: float) -> dict:
        start = time.perf_counter()
        time.sleep(delay_ms / 1000)
        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "tool": tool,
            "status": status,
            "detail": detail,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "request_id": request_id,
        }

    if category == "refund":
        logs.append(
            make_log(
                "refund_case_creator",
                "ok",
                "Iade talebi icin mock kayit acildi (RF-2104).",
                delay_map["refund"],
            )
        )
    elif category == "payment":
        logs.append(
            make_log(
                "payment_retry_advisor",
                "ok",
                "3D Secure ve banka onayi kontrolu icin mock adim olustu.",
                delay_map["payment"],
            )
        )
    elif category == "shipping":
        logs.append(
            make_log(
                "shipping_status_checker",
                "ok",
                "Kargo takibi icin mock kontrol yapildi (TK-9921).",
                delay_map["shipping"],
            )
        )
    else:
        logs.append(
            make_log(
                "customer_info_requester",
                "needs_info",
                "Eksik bilgi tespiti yapildi, ek bilgi istenmeli.",
                delay_map["general"],
            )
        )

    return {"tool_logs": logs}


def node_respond(state: AgentState) -> AgentState:
    sources = state.get("sources", [])
    plan = state.get("plan", [])
    llm_text = llm_response(
        state["message"],
        state["category"],
        state["priority"],
        plan,
        sources,
        state.get("tool_logs", []),
    )
    if llm_text:
        return {"response": llm_text}
    source_titles = ", ".join(item["title"] for item in sources)
    response_lines = [
        "Merhaba, talebinizi inceledim.",
        f"Kategori: {state['category']}, Oncelik: {state['priority']}.",
        "Onerecegim adimlar:",
        *[f"- {item}" for item in plan],
    ]
    if source_titles:
        response_lines.append(f"Ilgili kaynaklar: {source_titles}.")
    response_lines.append("Yardimci olabilecegim baska bir konu var mi?")
    return {"response": "\n".join(response_lines)}


def node_qa(state: AgentState) -> AgentState:
    response = state.get("response", "").strip()
    if len(response) > 900:
        response = response[:900].rstrip() + "..."
    if not response.endswith("?"):
        response = response + " Yardimci olabilecegim baska bir konu var mi?"
    return {"response": response}


def wrap_traced_node(name: str, handler):
    def _wrapped(state: AgentState) -> AgentState:
        trace_nodes = list(state.get("trace_nodes", []))
        trace_nodes.append(name)
        update = handler(state) or {}
        if isinstance(update, dict) and "trace_nodes" in update:
            extra_nodes = update.get("trace_nodes", [])
            if isinstance(extra_nodes, list):
                trace_nodes.extend(extra_nodes)
        return {**(update or {}), "trace_nodes": trace_nodes}

    return _wrapped


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", wrap_traced_node("classify", node_classify))
    graph.add_node("retrieve", wrap_traced_node("retrieve", node_retrieve))
    graph.add_node("plan", wrap_traced_node("plan", node_plan))
    graph.add_node("tools", wrap_traced_node("tools", node_tools))
    graph.add_node("respond", wrap_traced_node("respond", node_respond))
    graph.add_node("qa", wrap_traced_node("qa", node_qa))
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "tools")
    graph.add_edge("tools", "respond")
    graph.add_edge("respond", "qa")
    graph.add_edge("qa", END)
    return graph.compile()


agent_graph = build_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tickets/triage")
def triage_ticket(payload: TicketRequest) -> dict:
    return classify_text(payload.message)


@app.post("/agent/run")
def run_agent(payload: TicketRequest) -> dict:
    request_id = str(uuid4())
    trace_id = str(uuid4())
    final_state = agent_graph.invoke(
        {
            "message": payload.message,
            "request_id": request_id,
            "trace_id": trace_id,
            "trace_nodes": [],
        }
    )
    global LATEST_TRACE
    LATEST_TRACE = {
        "trace_id": trace_id,
        "nodes": final_state.get("trace_nodes", []),
        "finished_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "category": final_state.get("category"),
        "priority": final_state.get("priority"),
        "sentiment": final_state.get("sentiment"),
        "plan": final_state.get("plan", []),
        "sources": final_state.get("sources", []),
        "tool_logs": final_state.get("tool_logs", []),
        "response": final_state.get("response", ""),
    }


@app.get("/mock/tickets")
def get_mock_tickets() -> dict:
    return {"tickets": TICKETS}


@app.get("/agent/graph")
def get_agent_graph() -> dict:
    try:
        mermaid = agent_graph.get_graph().draw_mermaid()
    except Exception:
        mermaid = ""
    return {"mermaid": mermaid}


@app.get("/agent/trace/latest")
def get_latest_trace() -> dict:
    if not LATEST_TRACE:
        return {"trace_id": None, "nodes": [], "finished_at": None}
    return LATEST_TRACE

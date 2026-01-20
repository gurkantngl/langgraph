# Support Agent (LangGraph + Gemini + RAG)

Agentic destek asistani demo projesi. LangGraph akisi, Gemini LLM ve FAISS tabanli RAG ile calisir. Frontend React (Vite), backend FastAPI.

## Ozellikler
- LangGraph akisi: classify -> retrieve -> plan -> tools -> respond -> qa
- Gemini LLM ile siniflandirma, plan ve yanit uretimi
- Gemini Embeddings + FAISS ile gercek RAG
- Tool call simulasyonu + log ekranı (timestamp, duration, request_id)
- Mermaid ile akis grafigi ve dinamik trace highlight

## Gereksinimler
- Python 3.11+
- Node.js 18+

## Kurulum
### Backend
```
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```
cd frontend
npm install
```

## Ortam Degiskenleri (Backend)
Zorunlu:
- GEMINI_API_KEY veya GOOGLE_API_KEY

Opsiyonel:
- GEMINI_MODEL (ornek: gemini-1.5-flash)
- GEMINI_EMBED_MODEL (ornek: text-embedding-004)
- RAG_TOP_K (varsayilan 3)
- TOOL_DELAY_REFUND_MS / TOOL_DELAY_PAYMENT_MS / TOOL_DELAY_SHIPPING_MS / TOOL_DELAY_GENERAL_MS

## Calistirma
### Backend
```
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```
cd frontend
npm run dev
```

Tarayici: http://localhost:5173

## Notlar
- RAG verisi `backend/data/kb.json` icinden okunur.
- Ornek talepler `backend/data/tickets.json` icindedir.

# Backend (FastAPI + LangGraph)

Basit bir iskelet API ve LangGraph agent akisi. Tamamen mock veri uzerinden calisir.
Mock veriler `data/kb.json` ve `data/tickets.json` altinda tutulur.

## Kurulum
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

## Ortam Degiskenleri
Gemini kullanmak icin asagidaki degiskenleri ayarla:
- GEMINI_API_KEY veya GOOGLE_API_KEY
- GEMINI_MODEL (opsiyonel, ornek: gemini-1.5-flash)
- GEMINI_EMBED_MODEL (opsiyonel, ornek: text-embedding-004)

RAG ayarlari (opsiyonel):
- RAG_TOP_K (varsayilan 3)

Tool gecikmelerini ms cinsinden ayarlamak icin (opsiyonel):
- TOOL_DELAY_REFUND_MS
- TOOL_DELAY_PAYMENT_MS
- TOOL_DELAY_SHIPPING_MS
- TOOL_DELAY_GENERAL_MS

## Calistirma
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## Endpoint'ler
- GET /health
- POST /tickets/triage
- POST /agent/run
- GET /mock/tickets
- GET /agent/graph

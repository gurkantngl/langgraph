import React, { useEffect, useRef, useState } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tickets, setTickets] = useState([]);
  const [graph, setGraph] = useState("");
  const [traceNodes, setTraceNodes] = useState([]);
  const graphRef = useRef(null);
  const [theme, setTheme] = useState("dark");
  const [copied, setCopied] = useState(false);

  const loadTickets = async () => {
    try {
      const response = await fetch(`${API_BASE}/mock/tickets`);
      const data = await response.json();
      setTickets(data.tickets || []);
    } catch (err) {
      setTickets([]);
    }
  };

  const loadGraph = async () => {
    try {
      const response = await fetch(`${API_BASE}/agent/graph`);
      const data = await response.json();
      setGraph(data.mermaid || "");
    } catch (err) {
      setGraph("");
    }
  };

  const loadTrace = async () => {
    try {
      const response = await fetch(`${API_BASE}/agent/trace/latest`);
      const data = await response.json();
      setTraceNodes(Array.isArray(data.nodes) ? data.nodes : []);
    } catch (err) {
      setTraceNodes([]);
    }
  };

  useEffect(() => {
    loadTickets();
    loadGraph();
    loadTrace();
  }, []);

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!loading) return;
    const interval = setInterval(loadTrace, 600);
    return () => clearInterval(interval);
  }, [loading]);

  const buildTraceGraph = (baseGraph, nodes) => {
    if (!baseGraph) return "";
    if (!nodes || nodes.length === 0) return baseGraph;
    const uniqueNodes = Array.from(new Set(nodes));
    const classLines = uniqueNodes
      .map((node) => `class ${node} trace-node;`)
      .join("\n");
    return `${baseGraph}\n${classLines}\n`;
  };

  useEffect(() => {
    const container = graphRef.current;
    if (!container) return;
    const tracedGraph = buildTraceGraph(graph, traceNodes);
    if (!tracedGraph) {
      container.textContent = "Grafik bilgisi bulunamadi.";
      return;
    }

    if (!window.mermaid || !window.mermaid.render) {
      container.textContent = tracedGraph;
      return;
    }

    const mermaidTheme = theme === "light" ? "default" : "dark";
    window.mermaid.initialize({ startOnLoad: false, theme: mermaidTheme });
    window.mermaid
      .render(`graph-${Date.now()}`, tracedGraph)
      .then(({ svg }) => {
        container.innerHTML = svg;
      })
      .catch(() => {
        container.textContent = tracedGraph;
      });
  }, [graph, traceNodes, theme]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error("Sunucudan yanit alinamadi.");
      }

      const data = await response.json();
      setResult(data);
      loadTrace();
    } catch (err) {
      setError(err.message || "Bir hata olustu.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async (value) => {
    if (!value) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }
    } catch (err) {
      setCopied(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-row">
          <div>
            <h1>Support Agent Demo</h1>
            <p>Ticket metnini gir ve LangGraph akisinin sonucunu gor.</p>
          </div>
          <button
            type="button"
            className="ghost"
            onClick={() =>
              setTheme((prev) => (prev === "dark" ? "light" : "dark"))
            }
          >
            {theme === "dark" ? "Light Moda Gec" : "Dark Moda Gec"}
          </button>
        </div>
      </header>

      <form className="form" onSubmit={handleSubmit}>
        <label htmlFor="message">Ticket mesaji</label>
        <textarea
          id="message"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="Ornek: Siparisim hala kargoya verilmedi."
          rows={6}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Isleniyor..." : "Gonder"}
        </button>
      </form>

      {tickets.length > 0 && (
        <div className="panel">
          <h3>Ornek ticket'lar</h3>
          <div className="samples">
            {tickets.map((ticket) => (
              <button
                type="button"
                key={ticket.id}
                className="sample"
                onClick={() => setMessage(ticket.message)}
              >
                {ticket.message}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result">
          <h2>Agent Sonucu</h2>
          <div className="meta">
            <span>Kategori: {result.category}</span>
            <span>Oncelik: {result.priority}</span>
            <span>Duygu: {result.sentiment}</span>
            {result.request_id && (
              <span className="request-row">
                <span>Request ID: {result.request_id}</span>
                <button
                  type="button"
                  className="ghost copy-btn"
                  onClick={() => handleCopy(result.request_id)}
                >
                  {copied ? "Kopyalandi" : "Kopyala"}
                </button>
              </span>
            )}
          </div>

          <section className="panel">
            <h3>Plan</h3>
            <ul>
              {(result.plan || []).map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <h3>Kaynaklar</h3>
            <ul>
              {(result.sources || []).map((item) => (
                <li key={item.id}>
                  <div className="source-row">
                    <span>{item.title}</span>
                    <span className="source-score">
                      {typeof item.score === "number"
                        ? item.score.toFixed(3)
                        : "-"}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <h3>Tool Call Loglari</h3>
            <div className="logs">
              {(result.tool_logs || []).map((log, index) => (
                <div key={`${log.tool}-${index}`} className="log-item">
                  <div className="log-title">
                    <span className="log-tool">{log.tool}</span>
                    <span className={`log-status status-${log.status}`}>
                      {log.status}
                    </span>
                  </div>
                  <div className="log-meta">
                    <span>{log.timestamp || "-"}</span>
                    <span>{log.duration_ms ? `${log.duration_ms} ms` : "-"}</span>
                    <span>{log.request_id || "-"}</span>
                  </div>
                  <p>{log.detail}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <h3>Yanıt</h3>
            <pre>{result.response}</pre>
          </section>
        </div>
      )}

      <section className="panel">
        <div className="panel-head">
          <h3>Akis Grafigi (Mermaid)</h3>
          <div className="panel-actions">
            <button type="button" className="ghost" onClick={loadTrace}>
              Trace Yenile
            </button>
            <button type="button" onClick={loadGraph}>
              Grafik Yenile
            </button>
          </div>
        </div>
        <div ref={graphRef} className="graph" />
      </section>
    </div>
  );
}

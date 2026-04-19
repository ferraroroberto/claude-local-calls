"""HTML landing page served at GET /."""

LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>claude-local-calls</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --border: #242a35;
    --text: #e6e8eb;
    --muted: #8a94a6;
    --accent: #d97757;
    --ok: #4ade80;
    --code-bg: #0b0d12;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }
  h1 { margin: 0; font-size: 26px; font-weight: 600; }
  .tag { color: var(--muted); font-size: 13px; }
  .lede { color: var(--muted); margin: 0 0 28px; }
  h2 { font-size: 16px; font-weight: 600; margin: 28px 0 10px; letter-spacing: .02em; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
  }
  .status { display: flex; align-items: center; gap: 10px; }
  .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--muted); display: inline-block;
  }
  .dot.ok { background: var(--ok); box-shadow: 0 0 0 4px rgba(74,222,128,.12); }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    display: inline-block;
    background: #1e2430;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .pill b { color: var(--text); font-weight: 600; margin-right: 6px; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  ul.endpoints { list-style: none; padding: 0; margin: 0; }
  ul.endpoints li { padding: 8px 0; border-bottom: 1px solid var(--border); }
  ul.endpoints li:last-child { border-bottom: 0; }
  code.method {
    display: inline-block; min-width: 46px; text-align: center;
    font-size: 11px; font-weight: 700; letter-spacing: .04em;
    padding: 2px 6px; border-radius: 4px; margin-right: 10px;
    background: #2a3342; color: #cdd6e4;
  }
  code.method.post { background: #3a2a1a; color: #ffc596; }
  .path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color: #d5dae3;
  }
  pre .c { color: #6b7486; }
  pre .s { color: #a7d99a; }
  pre .k { color: #e6a95b; }
  footer { color: var(--muted); font-size: 12px; margin-top: 28px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>claude-local-calls</h1>
    <span class="tag">v0.1.0 &middot; local Anthropic-compatible API</span>
  </header>
  <p class="lede">
    Drop-in replacement for the Anthropic Messages API, backed by the
    <code>claude -p</code> CLI and your local Claude Code auth &mdash;
    so any client pointed at this server uses your subscription instead of API credits.
  </p>

  <div class="panel status">
    <span id="dot" class="dot"></span>
    <strong>server</strong>
    <span id="status-text" class="tag">checking&hellip;</span>
    <span style="flex:1"></span>
    <span class="pill"><b>host</b><span id="host">127.0.0.1:8000</span></span>
  </div>

  <h2>Endpoints</h2>
  <div class="panel">
    <ul class="endpoints">
      <li><code class="method">GET</code><span class="path">/health</span> &mdash; liveness check</li>
      <li><code class="method post">POST</code><span class="path">/v1/messages</span> &mdash; Anthropic-compatible messages</li>
      <li><code class="method">GET</code><span class="path"><a href="/docs">/docs</a></span> &mdash; Swagger UI (interactive)</li>
      <li><code class="method">GET</code><span class="path"><a href="/redoc">/redoc</a></span> &mdash; ReDoc (reference)</li>
      <li><code class="method">GET</code><span class="path"><a href="/info">/info</a></span> &mdash; machine-readable index</li>
    </ul>
  </div>

  <h2>Use from Python (official Anthropic SDK)</h2>
  <pre><span class="k">from</span> anthropic <span class="k">import</span> Anthropic

client = Anthropic(api_key=<span class="s">"local-dummy"</span>, base_url=<span class="s">"http://127.0.0.1:8000"</span>)
msg = client.messages.create(
    model=<span class="s">"claude-haiku-4-5"</span>,
    max_tokens=<span class="k">128</span>,
    messages=[{<span class="s">"role"</span>: <span class="s">"user"</span>, <span class="s">"content"</span>: <span class="s">"Hello"</span>}],
)
<span class="k">print</span>(msg.content[<span class="k">0</span>].text)</pre>

  <h2>Use from the shell</h2>
  <pre>curl -s http://127.0.0.1:8000/v1/messages \\
  -H <span class="s">"Content-Type: application/json"</span> \\
  -d <span class="s">'{"model":"claude-haiku-4-5","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'</span></pre>

  <h2>Request shape</h2>
  <pre>{
  <span class="s">"model"</span>:       <span class="s">"claude-haiku-4-5"</span>,   <span class="c">// passed to --model</span>
  <span class="s">"max_tokens"</span>:  <span class="k">1024</span>,                  <span class="c">// accepted, not enforced</span>
  <span class="s">"system"</span>:      <span class="s">"You are helpful."</span>,    <span class="c">// optional, passed to --system-prompt</span>
  <span class="s">"messages"</span>: [
    {<span class="s">"role"</span>: <span class="s">"user"</span>, <span class="s">"content"</span>: <span class="s">"Hello"</span>}
  ]
}</pre>

  <h2>Caveats</h2>
  <div class="panel">
    <div class="row">
      <span class="pill">no streaming</span>
      <span class="pill">multi-turn flattened</span>
      <span class="pill">no tool use</span>
      <span class="pill">no images</span>
      <span class="pill">no thinking blocks</span>
    </div>
  </div>

  <footer>
    Requires the <code>claude</code> CLI on <code>PATH</code>. Shells out to
    <code>claude -p --output-format json</code> for every request.
  </footer>
</main>

<script>
  fetch("/health").then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(j => {
      document.getElementById("dot").classList.add("ok");
      document.getElementById("status-text").textContent = j.status;
    })
    .catch(() => {
      document.getElementById("status-text").textContent = "unreachable";
    });
  document.getElementById("host").textContent = location.host;
</script>
</body>
</html>
"""

"""Generates context_graph.html from paper_input_context_graph.json."""
import json, pathlib
from collections import Counter
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
args = parser.parse_args()

OUTPUTS_DIR = pathlib.Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

input_name = os.path.splitext(os.path.basename(args.filename))[0].removesuffix("_context_graph")
data = json.loads((OUTPUTS_DIR / f"{input_name}_context_graph.json").read_text(encoding="utf-8"))

deg = Counter()
for a, b in data["edges"]:
    deg[a] += 1
    deg[b] += 1

nodes_js = json.dumps([{"id": n, "degree": deg.get(n, 0)} for n in data["nodes"]], ensure_ascii=False)
edges_js = json.dumps([{"source": e[0], "target": e[1]} for e in data["edges"]], ensure_ascii=False)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>SoG Context Graph</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
    height: 100vh;
  }}

  /* ── header bar ── */
  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 20;
    height: 52px;
    background: rgba(15,17,23,0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #1e2535;
    display: flex; align-items: center; gap: 16px; padding: 0 20px;
  }}
  #header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: .02em; color: #7dd3fc; white-space: nowrap; }}
  #stats {{ font-size: 12px; color: #64748b; white-space: nowrap; }}

  #search {{
    flex: 1; max-width: 320px;
    background: #1e2535; border: 1px solid #2d3748;
    border-radius: 8px; padding: 6px 12px;
    color: #e2e8f0; font-size: 13px; outline: none;
  }}
  #search:focus {{ border-color: #38bdf8; }}

  #clear-btn {{
    background: #1e2535; border: 1px solid #2d3748;
    border-radius: 6px; padding: 5px 12px;
    color: #94a3b8; font-size: 12px; cursor: pointer;
  }}
  #clear-btn:hover {{ border-color: #38bdf8; color: #38bdf8; }}

  /* ── legend ── */
  #legend {{
    position: fixed; bottom: 18px; left: 18px; z-index: 20;
    background: rgba(15,17,23,0.88); border: 1px solid #1e2535;
    border-radius: 10px; padding: 12px 16px; font-size: 12px;
    line-height: 1.9;
  }}
  .leg-row {{ display: flex; align-items: center; gap: 8px; }}
  .leg-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}

  /* ── tooltip ── */
  #tooltip {{
    position: fixed; z-index: 30; pointer-events: none;
    background: rgba(15,17,23,0.95); border: 1px solid #2d3748;
    border-radius: 10px; padding: 10px 14px;
    font-size: 12.5px; line-height: 1.6;
    max-width: 280px; display: none;
  }}
  #tooltip .tt-name {{ font-weight: 600; color: #38bdf8; margin-bottom: 4px; }}

  /* ── SVG ── */
  svg {{ display: block; width: 100vw; height: 100vh; cursor: grab; }}
  svg:active {{ cursor: grabbing; }}

  .link {{ stroke: #2d3748; stroke-opacity: 0.55; stroke-width: 1px; }}
  .link.highlighted {{ stroke: #38bdf8; stroke-opacity: 1; stroke-width: 1.8px; }}

  .node circle {{
    stroke-width: 1.5px;
    transition: r .15s, stroke-opacity .15s;
    cursor: pointer;
  }}
  .node circle:hover {{ stroke-opacity: 1 !important; }}
  .node.dimmed circle {{ opacity: 0.12; }}
  .node.dimmed text {{ opacity: 0.08; }}

  .node text {{
    font-size: 10px; fill: #cbd5e1;
    pointer-events: none; user-select: none;
    text-shadow: 0 0 4px #0f1117, 0 0 4px #0f1117;
  }}

  /* ── controls ── */
  #controls {{
    position: fixed; right: 18px; bottom: 18px; z-index: 20;
    display: flex; flex-direction: column; gap: 6px;
  }}
  .ctrl-btn {{
    width: 36px; height: 36px;
    background: rgba(15,17,23,0.9); border: 1px solid #2d3748;
    border-radius: 8px; color: #94a3b8; font-size: 18px;
    cursor: pointer; display: grid; place-items: center;
  }}
  .ctrl-btn:hover {{ border-color: #38bdf8; color: #38bdf8; }}
</style>
</head>
<body>

<div id="header">
  <h1>⬡ SoG Context Graph</h1>
  <span id="stats"></span>
  <input id="search" type="text" placeholder="Search entity…" autocomplete="off"/>
  <button id="clear-btn">Reset</button>
</div>

<div id="legend">
  <div class="leg-row"><div class="leg-dot" style="background:#ef4444"></div> Hub (≥ 20 connections)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#f97316"></div> High (10–19)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#38bdf8"></div> Medium (5–9)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#818cf8"></div> Low (1–4)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#475569"></div> Isolated</div>
</div>

<div id="tooltip">
  <div class="tt-name" id="tt-name"></div>
  <div id="tt-degree"></div>
  <div id="tt-neighbors"></div>
</div>

<div id="controls">
  <button class="ctrl-btn" id="zoom-in"  title="Zoom in">+</button>
  <button class="ctrl-btn" id="zoom-out" title="Zoom out">−</button>
  <button class="ctrl-btn" id="zoom-fit" title="Fit view" style="font-size:13px">⛶</button>
</div>

<svg id="graph"></svg>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const RAW_NODES = {nodes_js};
const RAW_EDGES = {edges_js};

// ── colour by degree ──────────────────────────────────────────────────────
function nodeColor(d) {{
  if (d.degree >= 20) return "#ef4444";
  if (d.degree >= 10) return "#f97316";
  if (d.degree >=  5) return "#38bdf8";
  if (d.degree >=  1) return "#818cf8";
  return "#475569";
}}
function nodeRadius(d) {{
  return Math.max(5, Math.min(22, 5 + d.degree * 0.55));
}}

// ── SVG & zoom ────────────────────────────────────────────────────────────
const svgEl = document.getElementById("graph");
const svg   = d3.select(svgEl);
const W = () => window.innerWidth;
const H = () => window.innerHeight;

const zoomBehavior = d3.zoom().scaleExtent([0.05, 6])
  .on("zoom", e => container.attr("transform", e.transform));
svg.call(zoomBehavior);

const container = svg.append("g");
const linkGroup = container.append("g").attr("class", "links");
const nodeGroup = container.append("g").attr("class", "nodes");

// ── adjacency for fast neighbour lookup ──────────────────────────────────
const adjMap = new Map();
RAW_NODES.forEach(n => adjMap.set(n.id, new Set()));
RAW_EDGES.forEach(e => {{
  adjMap.get(e.source)?.add(e.target);
  adjMap.get(e.target)?.add(e.source);
}});

// ── simulation ────────────────────────────────────────────────────────────
const simulation = d3.forceSimulation(RAW_NODES)
  .force("link", d3.forceLink(RAW_EDGES).id(d => d.id).distance(60).strength(0.5))
  .force("charge", d3.forceManyBody().strength(-180).distanceMax(350))
  .force("center", d3.forceCenter(W() / 2, H() / 2))
  .force("collide", d3.forceCollide(d => nodeRadius(d) + 4))
  .alphaDecay(0.025);

// ── links ─────────────────────────────────────────────────────────────────
const link = linkGroup.selectAll("line")
  .data(RAW_EDGES)
  .join("line")
  .attr("class", "link");

// ── nodes ─────────────────────────────────────────────────────────────────
const node = nodeGroup.selectAll("g")
  .data(RAW_NODES)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag",  (e, d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on("end",   (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));

node.append("circle")
  .attr("r", nodeRadius)
  .attr("fill", nodeColor)
  .attr("stroke", d => d3.color(nodeColor(d)).brighter(0.8))
  .attr("stroke-opacity", 0.6);

node.append("text")
  .attr("dy", d => -nodeRadius(d) - 3)
  .attr("text-anchor", "middle")
  .text(d => d.id.length > 22 ? d.id.slice(0, 20) + "…" : d.id);

// stat bar
document.getElementById("stats").textContent =
  `${{RAW_NODES.length}} entities · ${{RAW_EDGES.length}} co-occurrence edges`;

// ── tick ──────────────────────────────────────────────────────────────────
simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

// ── tooltip ───────────────────────────────────────────────────────────────
const tooltip = document.getElementById("tooltip");
node
  .on("mousemove", (e, d) => {{
    const nbrs = [...(adjMap.get(d.id) || [])].slice(0, 8);
    document.getElementById("tt-name").textContent = d.id;
    document.getElementById("tt-degree").textContent = `Connections: ${{d.degree}}`;
    document.getElementById("tt-neighbors").textContent =
      nbrs.length ? "Neighbours: " + nbrs.join(", ") + (adjMap.get(d.id).size > 8 ? "…" : "") : "";
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top  = (e.clientY - 10) + "px";
  }})
  .on("mouseleave", () => {{ tooltip.style.display = "none"; }});

// ── click highlight ───────────────────────────────────────────────────────
let selected = null;
node.on("click", (e, d) => {{
  e.stopPropagation();
  if (selected === d.id) {{ clearHighlight(); return; }}
  selected = d.id;
  highlight(d.id);
}});
svg.on("click", () => clearHighlight());

function highlight(id) {{
  const nbrs = adjMap.get(id) || new Set();
  node.classed("dimmed", n => n.id !== id && !nbrs.has(n.id));
  link
    .classed("highlighted", l => l.source.id === id || l.target.id === id)
    .style("stroke", l => (l.source.id===id||l.target.id===id) ? "#38bdf8" : null)
    .style("stroke-opacity", l => (l.source.id===id||l.target.id===id) ? 1 : 0.1);
}}
function clearHighlight() {{
  selected = null;
  node.classed("dimmed", false);
  link.classed("highlighted", false).style("stroke", null).style("stroke-opacity", null);
  document.getElementById("search").value = "";
}}

// ── search ────────────────────────────────────────────────────────────────
document.getElementById("search").addEventListener("input", e => {{
  const q = e.target.value.trim().toLowerCase();
  if (!q) {{ clearHighlight(); return; }}
  const match = RAW_NODES.find(n => n.id.toLowerCase().includes(q));
  if (match) {{
    selected = match.id;
    highlight(match.id);
    // pan to node
    const t = d3.zoomTransform(svgEl);
    const x = W()/2 - t.k * match.x;
    const y = H()/2 - t.k * match.y;
    svg.transition().duration(500)
      .call(zoomBehavior.transform, d3.zoomIdentity.translate(x, y).scale(t.k));
  }}
}});
document.getElementById("clear-btn").addEventListener("click", clearHighlight);

// ── zoom controls ─────────────────────────────────────────────────────────
document.getElementById("zoom-in") .addEventListener("click", () => svg.transition().call(zoomBehavior.scaleBy, 1.4));
document.getElementById("zoom-out").addEventListener("click", () => svg.transition().call(zoomBehavior.scaleBy, 0.7));
document.getElementById("zoom-fit").addEventListener("click", fitView);

function fitView() {{
  const bounds = container.node().getBBox();
  const pad = 40;
  const scaleX = (W() - pad*2) / bounds.width;
  const scaleY = (H() - pad*2) / bounds.height;
  const scale  = Math.min(scaleX, scaleY, 2);
  const tx = W()/2 - scale*(bounds.x + bounds.width/2);
  const ty = H()/2 - scale*(bounds.y + bounds.height/2);
  svg.transition().duration(700)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}}

window.addEventListener("resize", () => simulation.force("center", d3.forceCenter(W()/2, H()/2)).alpha(0.05).restart());
</script>
</body>
</html>
"""

out = OUTPUTS_DIR / f"{input_name}_context_graph.html"
out.write_text(HTML, encoding="utf-8")
print(f"Written: {out.resolve()}  ({out.stat().st_size:,} bytes)")

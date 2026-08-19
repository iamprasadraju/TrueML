"""computation graph builder + renderer. traces the UOp DAG and draws it as
self-contained interactive HTML/SVG. no bloat."""

import math, json, os, tempfile, textwrap, webbrowser
from pathlib import Path

import networkx as nx
import numpy as np

# ── visual constants ── micrograd-style, academic look ──────────────────────
EDGE = dict(color="#3a3a3a", width=1.4, rad=0.15, max_rad_dist=3.5,
            hover_color="#0a0a0a", hover_width=2.6)
OP = dict(fill="white", edge="black", lw=1.6, font=10, min_d=1.5, char_k=0.11, caption_pad=0.18)
TEN = dict(fill="#fafafa", edge="black", lw=1.6, w=3.4, header=0.52, line_h=0.32,
           pad=0.2, min_h=1.05, text_pad=0.24)
LAYOUT = dict(hgap=1.35, vgap=0.95, meta_chars=22, upi=1.35, dpi=150, margin=0.05)
STYLE = dict(label=10.5, meta=8.5, text="#111", muted="#555", sep="#999")
ZOOM = dict(factor=0.9, lo=0.5, hi=10.0)

BROADCAST_OPS = {"ADD","SUB","SUBTRACT","MUL","MULTIPLY","DIV","DIVIDE","POW","RELU","SIGMOID","EXP","LOG","NEG"}

# ── shape inference helpers ─────────────────────────────────────────────────
def _matmul_shape(a, b):
    if len(a) == 1 and len(b) == 1: return () if a[0] == b[0] else None
    if len(a) == 1:
        return None if a[0] != b[-2] else tuple(np.broadcast_shapes(b[:-2])) + (b[-1],)
    if len(b) == 1:
        return None if a[-1] != b[0] else tuple(np.broadcast_shapes(a[:-2])) + (a[-2],)
    if a[-1] != b[-2]: return None
    return tuple(np.broadcast_shapes(a[:-2], b[:-2])) + (a[-2], b[-1])

# ── geometry: clip segments to shapes ───────────────────────────────────────
def _clip_box(p, q, box):
    x0, y0, x1, y1 = box
    dx, dy = q[0]-p[0], q[1]-p[1]
    inside = lambda t: x0 <= p[0]+t*dx <= x1 and y0 <= p[1]+t*dy <= y1
    hits = []
    if dx: hits += [t for t in ((x0-p[0])/dx, (x1-p[0])/dx) if inside(t)]
    if dy: hits += [t for t in ((y0-p[1])/dy, (y1-p[1])/dy) if inside(t)]
    return (max(0., min(hits)), min(1., max(hits))) if hits else (0., 1.)

def _clip_ellipse(p, q, center, rx, ry):
    cx, cy = center
    dx, dy = q[0]-p[0], q[1]-p[1]
    px, py = p[0]-cx, p[1]-cy
    a = (dx/rx)**2 + (dy/ry)**2
    b = 2*(px*dx/rx**2 + py*dy/ry**2)
    c = (px/rx)**2 + (py/ry)**2 - 1
    disc = b*b - 4*a*c
    if a == 0 or disc < 0: return 0., 1.
    s = math.sqrt(disc)
    t1, t2 = (-b-s)/(2*a), (-b+s)/(2*a)
    return max(0., min(t1, t2)), min(1., max(t1, t2))

# ── the graph ───────────────────────────────────────────────────────────────
class ComputationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._reg, self._keys, self._nk = {}, {}, 0
        self._opd = OP["min_d"]

    def build(self, tensor):
        """walks the UOp graph rooted at tensor.uop (post-order), deduplicates by id().
        _reg pins uops alive so id() can't get reused under us."""
        g, reg, keys = self.graph, self._reg, self._keys
        order = tensor.uop.toposort()
        for node in order:
            reg[id(node)] = node
            nid = id(node)
            key = keys.get(nid)
            if key is None:
                key = f"n{self._nk}"; self._nk += 1; keys[nid] = key
            g.add_node(key, label=node.op,
                       metadata={"shape": node.shape, "ndim": node.ndim, "dtype": node.dtype},
                       operation=node.op not in (None, "LOAD"))
            for child in node.src:
                g.add_edge(keys[id(child)], key)

    # ── layout engine ───────────────────────────────────────────────────
    def _propagate_shapes(self):
        g = self.graph
        try: order = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible: order = list(g.nodes)
        for n in order:
            d = g.nodes[n]
            if not d["operation"] or d.get("metadata"): continue
            shapes = [g.nodes[p].get("metadata",{}).get("shape") for p in g.predecessors(n)]
            dtypes = [g.nodes[p].get("metadata",{}).get("dtype") for p in g.predecessors(n)]
            shapes, dtypes = [s for s in shapes if s], [dt for dt in dtypes if dt]
            if not shapes: d["metadata"] = {"shape":"?","ndim":"?","dtype":"?"}; continue
            op = d["label"]
            try:
                if op == "TRANSPOSE": shape = shapes[0][::-1]
                elif op == "MATMUL": shape = _matmul_shape(*shapes)
                elif op in BROADCAST_OPS: shape = np.broadcast_shapes(*shapes)
                else: shape = shapes[0]
            except Exception: shape = None
            if shape is None: d["metadata"] = {"shape":"?","ndim":"?","dtype":"?"}; continue
            try: dtype = np.result_type(*[np.dtype(dt) for dt in dtypes])
            except Exception: dtype = dtypes[0] if dtypes else "?"
            d["metadata"] = {"shape": tuple(shape), "ndim": len(shape), "dtype": dtype}

    @staticmethod
    def _meta_lines(meta):
        lines = []
        for k, v in meta.items():
            wrapped = textwrap.wrap(str(v), width=LAYOUT["meta_chars"]) or [str(v)]
            lines.append(f"{k}={wrapped[0]}")
            lines.extend(f"  {p}" for p in wrapped[1:])
        return lines

    def _lines(self, d):
        if "_lines" not in d: d["_lines"] = self._meta_lines(d.get("metadata", {}))
        return d["_lines"]

    def _node_size(self, d):
        if "_size" in d: return d["_size"]
        if d["operation"]:
            lines = self._lines(d)
            sz = (self._opd, self._opd + TEN["line_h"]*len(lines) + 0.3) if lines else (self._opd, self._opd)
        else:
            lines = self._lines(d)
            sz = (TEN["w"], TEN["header"] + TEN["pad"] + len(lines)*TEN["line_h"]) if lines else (TEN["w"], TEN["min_h"])
        d["_size"] = sz
        return sz

    def _layers(self):
        g = self.graph
        try: order = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible: order = list(g.nodes)
        depth = {}
        for _ in range(g.number_of_nodes() + 1):
            changed = False
            for n in order:
                preds = list(g.predecessors(n))
                if not preds:
                    if depth.setdefault(n, 0) != 0: changed = True
                elif all(p in depth for p in preds):
                    d = max(depth[p] for p in preds) + 1
                    if depth.get(n) != d: depth[n] = d; changed = True
            if not changed: break
        if len(depth) != g.number_of_nodes(): depth = {n: 0 for n in g.nodes}
        mx = max(depth.values(), default=0)
        layers = [[] for _ in range(mx + 1)]
        for n, d in depth.items(): layers[d].append(n)
        return layers

    def _order_layers(self, layers):
        g = self.graph
        for layer in layers: layer.sort(key=lambda n: g.nodes[n]["label"])
        def bary(layer, idx):
            def pos(n):
                hits = [idx[nb] for nb in g.predecessors(n) if nb in idx]
                return sum(hits)/len(hits) if hits else 0.
            layer.sort(key=pos)
        for _ in range(10):
            changed = False
            idx = {}
            for d in range(1, len(layers)):
                idx.update({n: i for i, n in enumerate(layers[d-1])})
                if len(layers[d]) > 1:
                    before = list(layers[d]); bary(layers[d], idx)
                    if layers[d] != before: changed = True
            idx = {}
            for d in range(len(layers)-2, -1, -1):
                idx.update({n: i for i, n in enumerate(layers[d+1])})
                if len(layers[d]) > 1:
                    before = list(layers[d]); bary(layers[d], idx)
                    if layers[d] != before: changed = True
            if not changed: break
        return layers

    def _layout(self):
        self._propagate_shapes()
        self._opd = max(OP["min_d"], 0.95 + OP["char_k"] * max(
            (len(d["label"]) for _, d in self.graph.nodes(data=True) if d["operation"]), default=0))
        layers = self._order_layers(self._layers())
        sizes = [[self._node_size(self.graph.nodes[n]) for n in L] for L in layers]
        col_w = [max((w for w,_ in col), default=0) for col in sizes]
        row_h = [max((h for _,h in row), default=0) for row in sizes]
        col_x, x = [], 0.
        for w in col_w: col_x.append(x + w/2); x += w + LAYOUT["hgap"]
        pos = {}
        for d, L in enumerate(layers):
            step = row_h[d] + LAYOUT["vgap"]
            for i, n in enumerate(L): pos[n] = (col_x[d], (i - (len(L)-1)/2) * step)
        tw = x - LAYOUT["hgap"]
        th = max((len(L)-1)*(row_h[d]+LAYOUT["vgap"]) + row_h[d] for d, L in enumerate(layers)) if layers else 0.
        return pos, tw, th

    # ── clipping ────────────────────────────────────────────────────────
    def _clip(self, p, q, node):
        d = self.graph.nodes[node]
        if d["operation"]:
            r = self._opd / 2
            return _clip_ellipse(p, q, p, r, r)
        w, h = self._node_size(d)
        return _clip_box(p, q, (p[0]-w/2, p[1]-h/2, p[0]+w/2, p[1]+h/2))

    def _edge_endpoints(self, pos, u, v):
        pu, pv = pos[u], pos[v]
        _, bu = self._clip(pu, pv, u)
        _, bv = self._clip(pv, pu, v)
        sx, sy = pu[0]+bu*(pv[0]-pu[0]), pu[1]+bu*(pv[1]-pu[1])
        ex, ey = pv[0]+bv*(pu[0]-pv[0]), pv[1]+bv*(pu[1]-pv[1])
        return (sx, sy), (ex, ey)

    # ── HTML/SVG interactive viewer ─────────────────────────────────────
    def _svg_edge(self, pos, u, v):
        s, e = self._edge_endpoints(pos, u, v)
        dx, dy = e[0]-s[0], e[1]-s[1]
        L = math.hypot(dx, dy)
        rad = EDGE["rad"] * min(1., EDGE["max_rad_dist"]/L) if L else 0.
        sag = rad * L
        mx, my = (s[0]+e[0])/2, (s[1]+e[1])/2
        nx_, ny_ = (-dy/L, dx/L) if L else (0., 0.)
        cx, cy = mx + nx_*sag, my + ny_*sag
        return f'M {s[0]:.3f} {-s[1]:.3f} Q {cx:.3f} {-cy:.3f} {e[0]:.3f} {-e[1]:.3f}'

    def _svg_op(self, node, d, x, y):
        r = self._opd / 2
        lines = self._lines(d)
        esc = lambda s: str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        
        lbl = str(d.get("label", ""))
        palette = [
            ("#e0f2fe", "#0284c7", "#0c4a6e", "#bae6fd"), # blue
            ("#dcfce7", "#16a34a", "#14532d", "#bbf7d0"), # green
            ("#f3e8ff", "#9333ea", "#581c87", "#e9d5ff"), # purple
            ("#ffedd5", "#ea580c", "#7c2d12", "#fed7aa"), # orange
            ("#fce7f3", "#db2777", "#831843", "#fbcfe8"), # pink
            ("#e0e7ff", "#4f46e5", "#312e81", "#c7d2fe"), # indigo
            ("#ccfbf1", "#0d9488", "#134e4a", "#99f6e4"), # teal
        ]
        bg, border, text, hover = palette[sum(ord(c) for c in lbl) % len(palette)]
        
        parts = [f'<g class="node op" data-id="{node}" style="--bg:{bg}; --border:{border}; --text:{text}; --hover:{hover};" transform="translate({x:.3f} {-y:.3f})">',
                 f'<circle r="{r:.3f}"/>', f'<text class="op-label" y="0">{esc(lbl)}</text>']
        for i, line in enumerate(lines):
            ly = r + OP["caption_pad"] + TEN["line_h"]*(i+.6)
            parts.append(f'<text class="caption" x="0" y="{ly:.3f}">{esc(line)}</text>')
        parts.append("</g>")
        return "\n".join(parts)

    def _svg_tensor(self, node, d, x, y):
        w, h = self._node_size(d)
        lines = self._lines(d)
        hw, hh = w/2, h/2
        esc = lambda s: str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts = [f'<g class="node tensor" data-id="{node}" transform="translate({x:.3f} {-y:.3f})">',
                 f'<rect x="{-hw-.02:.3f}" y="{-hh-.02:.3f}" width="{w+.04:.3f}" height="{h+.04:.3f}" rx="0.02"/>']
        if not lines:
            parts.append(f'<text class="tensor-label" x="0" y="0" text-anchor="middle">{esc(d["label"])}</text>')
        else:
            px = -hw + TEN["text_pad"]
            parts.append(f'<text class="tensor-label" x="{px:.3f}" y="{-hh+TEN["header"]/2:.3f}">{esc(d["label"])}</text>')
            parts.append(f'<line class="rule" x1="{-hw+.18:.3f}" y1="{-hh+TEN["header"]:.3f}" x2="{hw-.18:.3f}" y2="{-hh+TEN["header"]:.3f}"/>')
            for i, line in enumerate(lines):
                my = -hh + TEN["header"] + TEN["line_h"]*(i+.6)
                parts.append(f'<text class="meta" x="{px:.3f}" y="{my:.3f}">{esc(line)}</text>')
        parts.append("</g>")
        return "\n".join(parts)

    def _build_html(self):
        pos, tw, th = self._layout()
        pad_x, pad_y = LAYOUT["margin"]*tw, LAYOUT["margin"]*th
        vw, vh = tw + 2*pad_x, th + 2*pad_y

        # BUG FIX: layout Y is centered around 0 (-th/2 to +th/2), negated for SVG.
        # world transform must shift by th/2 so content sits inside viewBox [0, vh].
        oy = pad_y + th/2

        edge_svg = "\n".join(f'<path class="edge" d="{self._svg_edge(pos, u, v)}" marker-end="url(#arrow)"/>'
                             for u, v in self.graph.edges())
        node_svg = "\n".join(self._svg_op(n, d, *pos[n]) if d["operation"] else self._svg_tensor(n, d, *pos[n])
                             for n, d in self.graph.nodes(data=True))

        def _clean(v):
            if isinstance(v, np.dtype): return str(v)
            if isinstance(v, tuple): return [_clean(x) for x in v]
            if isinstance(v, (np.integer, np.floating, np.bool_)): return v.item()
            return v

        payload = json.dumps({
            "nodes": [{"id": str(n), "label": d["label"],
                       "metadata": {k: _clean(v) for k,v in d.get("metadata",{}).items()}}
                      for n, d in self.graph.nodes(data=True)],
            "edges": [{"u": str(u), "v": str(v)} for u,v in self.graph.edges()],
            "size": [vw, vh],
        }).replace("</", "<\\/")

        marker = f'''<marker id="arrow" viewBox="0 0 1 1" refX="1" refY="0.5"
            markerWidth="0.25" markerHeight="0.25" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M 0 0 L 1 0.5 L 0 1 Z" fill="{EDGE['color']}"/></marker>'''

        return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Computation Graph</title>
<style>{_CSS}</style></head><body>
<div id="toolbar">
  <button id="fit">Fit</button><button id="zoomin">+</button>
  <button id="zoomout">&minus;</button><button id="captions">Captions</button>
  <button id="export">Export SVG</button>
</div>
<div id="viewport">
  <svg id="svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.2f} {vh:.2f}"
       preserveAspectRatio="xMidYMid meet">
    <defs>{marker}</defs>
    <g id="world" transform="translate(0 0) scale(1)">
      <g transform="translate({pad_x:.4f} {oy:.4f})">
        <g id="edges">{edge_svg}</g>
        <g id="nodes">{node_svg}</g>
      </g>
    </g>
  </svg>
  <div id="tooltip" hidden></div>
</div>
<script type="application/json" id="graph-data">{payload}</script>
<script>{_JS}</script></body></html>'''

    def view(self, save=None, open_browser=False, notebook=None):
        """interactive HTML/SVG viewer. jupyter inline or standalone browser."""
        html = self._build_html()
        if save:
            with open(save, "w") as f: f.write(html)
        if notebook is None:
            try:
                from IPython import get_ipython
                notebook = get_ipython() is not None
            except Exception: notebook = False
        if open_browser:
            path = save
            if path is None:
                fd, path = tempfile.mkstemp(suffix=".html"); os.close(fd)
                with open(path, "w") as f: f.write(html)
            webbrowser.open("file://" + os.path.abspath(path))
        if notebook:
            from IPython.display import IFrame, display
            import base64
            # Embed the full interactive HTML inside an iframe to prevent CSS leaking
            # and to preserve all JS interactivity (tooltips, dragging, zooming).
            b64_html = base64.b64encode(html.encode('utf-8')).decode('utf-8')
            src = f"data:text/html;base64,{b64_html}"
            display(IFrame(src=src, width="100%", height="600px"))
            if not open_browser: print("Interactive version: view(save='graph.html', open_browser=True)")
        elif not save and not open_browser:
            print("view(): pass save='graph.html' or open_browser=True to open in a browser.")
        return None

# ── inline CSS ──────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box}
body{margin:0;font-family:"MS Sans Serif","Segoe UI",Helvetica,Arial,sans-serif;background:#fff;overflow:hidden}
#toolbar{display:flex;gap:6px;align-items:center;padding:8px 12px;border-bottom:2px solid #404040;font-size:13px;color:#000;height:48px;flex-shrink:0;background:#c0c0c0;box-shadow:inset 0 -1px 0 #fff}
#toolbar button{font:inherit;font-weight:bold;padding:4px 12px;background:#c0c0c0;border-top:2px solid #fff;border-left:2px solid #fff;border-bottom:2px solid #000;border-right:2px solid #000;color:#000;cursor:pointer;box-shadow:inset -1px -1px 0 #808080, inset 1px 1px 0 #dfdfdf;outline:none}
#toolbar button:active{border-top:2px solid #000;border-left:2px solid #000;border-bottom:2px solid #fff;border-right:2px solid #fff;box-shadow:inset 1px 1px 0 #808080, inset -1px -1px 0 #dfdfdf;padding:5px 11px 3px 13px}
#viewport{position:relative;width:100%;height:calc(100vh - 48px);display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:grab;user-select:none;background:#808080}
#viewport.panning{cursor:grabbing}
#svg{width:100%;height:100%;display:block;background:#fff;box-shadow:inset 2px 2px 5px rgba(0,0,0,0.5)}
.edge{fill:none;stroke:#9ca3af;stroke-width:0.026;transition:stroke .15s,stroke-width .15s}
.edge.active{stroke:#4b5563;stroke-width:0.049}
.node{cursor:pointer}
.node.op circle{fill:var(--bg, #e0f2fe);stroke:var(--border, #0284c7);stroke-width:0.03;transition:fill .15s}
.node.op:hover circle,.node.op.selected circle{fill:var(--hover, #bae6fd)}
.node.tensor rect{fill:#fef9c3;stroke:#ca8a04;stroke-width:0.03;transition:fill .15s}
.node.tensor:hover rect,.node.tensor.selected rect{fill:#fef08a}
.op-label{text-anchor:middle;dominant-baseline:central;font-weight:700;fill:var(--text, #0c4a6e);font-size:0.1875px}
.tensor-label{text-anchor:start;dominant-baseline:central;font-weight:700;fill:#713f12;font-size:0.1969px}
.meta,.caption{font-family:"DejaVu Sans Mono",Menlo,Consolas,monospace;fill:#6b7280;font-size:0.1594px;text-anchor:start;dominant-baseline:central}
.caption{text-anchor:middle}
.rule{stroke:#fde047;stroke-width:0.015}
#svg.no-captions .caption{display:none}
#tooltip{position:absolute;z-index:10;max-width:320px;background:#ffffe1;border:1px solid #000;padding:4px 6px;font-size:12px;line-height:1.4;box-shadow:2px 2px 0 rgba(0,0,0,0.2);pointer-events:none}
#tooltip div:first-child{font-weight:700}
#tooltip .mono{font-family:Menlo,Consolas,monospace;font-size:11px;color:#333}
#tooltip hr{border:none;border-top:1px solid #ccc;margin:4px 0}
#tooltip .edges{color:#555}
"""

# ── inline JS ───────────────────────────────────────────────────────────────
_JS = """(function(){
"use strict";
var D=JSON.parse(document.getElementById("graph-data").textContent),
    svg=document.getElementById("svg"),W=document.getElementById("world"),
    VP=document.getElementById("viewport"),TT=document.getElementById("tooltip"),
    edges=[].slice.call(document.querySelectorAll(".edge")),
    nodes=[].slice.call(document.querySelectorAll(".node")),
    SZ=D.size,vb=svg.viewBox.baseVal,
    MX=10,adj={},sc=1,tx=0,ty=0,pin=null,dragging=false,ds=null,dtx=0,dty=0;

D.edges.forEach(function(e,i){(adj[e.u]=adj[e.u]||[]).push(i);(adj[e.v]=adj[e.v]||[]).push(i)});

function toU(cx,cy){
  var pt=svg.createSVGPoint();pt.x=cx;pt.y=cy;
  var ctm=svg.getScreenCTM();
  if(!ctm)return [cx,cy];
  var u=pt.matrixTransform(ctm.inverse());
  return [u.x,u.y];
}

function apply(){W.setAttribute("transform","translate("+tx+" "+ty+") scale("+sc+")")}

function fit(){
  var r=VP.getBoundingClientRect(),ar=r.width/r.height,vr=vb.width/vb.height;
  sc=1;tx=0;ty=0;
  clearA();hideTT();apply();
}

function zoomAt(px,py,f){
  var ns=Math.min(MX,Math.max(0.2,sc*f)),k=ns/sc;
  tx=px-(px-tx)*k;ty=py-(py-ty)*k;sc=ns;apply();
}

function setA(id){edges.forEach(function(el,i){var e=D.edges[i];el.classList.toggle("active",e.u===id||e.v===id)})}
function clearA(){edges.forEach(function(el){el.classList.remove("active")})}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function meta(m){return Object.keys(m).map(function(k){var v=m[k];if(Array.isArray(v))v="("+v.join(", ")+")";return'<span class="mono">'+k+"="+v+"</span>"}).join("<br>")}
function ttHtml(id){var n=null;D.nodes.forEach(function(x){if(x.id===id)n=x});if(!n)return"";var deg=(adj[id]||[]).length;return"<div>"+esc(n.label)+"</div>"+meta(n.metadata)+'<hr><span class="edges">'+deg+" connection"+(deg===1?"":"s")+"</span>"}

function showTT(id,cx,cy){TT.innerHTML=ttHtml(id);TT.hidden=false;moveTT(cx,cy)}
function moveTT(cx,cy){
  var r=VP.getBoundingClientRect(),x=cx-r.left+14,y=cy-r.top+14;
  if(x+240>r.width)x=cx-r.left-250;
  if(y+120>r.height)y=cy-r.top-100;
  TT.style.left=Math.max(0,x)+"px";TT.style.top=Math.max(0,y)+"px";
}
function hideTT(){TT.hidden=true;TT.innerHTML=""}

function isNode(el){while(el&&el!==VP){if(el.classList&&el.classList.contains("node"))return true;el=el.parentNode}return false}

nodes.forEach(function(el){
  el.addEventListener("mouseenter",function(ev){
    var id=el.getAttribute("data-id");setA(id);
    if(pin!==id)showTT(id,ev.clientX,ev.clientY);
  });
  el.addEventListener("mouseleave",function(){
    var id=el.getAttribute("data-id");
    if(pin!==id){clearA();hideTT()}
  });
  el.addEventListener("mousemove",function(ev){
    if(pin!==el.getAttribute("data-id"))moveTT(ev.clientX,ev.clientY);
  });
  el.addEventListener("click",function(ev){
    ev.stopPropagation();var id=el.getAttribute("data-id"),was=pin===id;
    nodes.forEach(function(x){x.classList.remove("selected")});
    if(was){pin=null;clearA();hideTT();return}
    pin=id;el.classList.add("selected");setA(id);showTT(id,ev.clientX,ev.clientY);
  });
});

VP.addEventListener("click",function(ev){
  if(isNode(ev.target))return;
  pin=null;nodes.forEach(function(x){x.classList.remove("selected")});clearA();hideTT();
});

VP.addEventListener("wheel",function(ev){
  ev.preventDefault();
  var u=toU(ev.clientX,ev.clientY);
  var f=Math.exp(-ev.deltaY*0.005);
  zoomAt(u[0],u[1],f);
},{passive:false});

VP.addEventListener("mousedown",function(ev){
  if(ev.button!==0||isNode(ev.target))return;
  dragging=true;VP.classList.add("panning");
  ds=toU(ev.clientX,ev.clientY);dtx=tx;dty=ty;
});

document.addEventListener("mousemove",function(ev){
  if(!dragging)return;
  var u=toU(ev.clientX,ev.clientY);
  tx=dtx+(u[0]-ds[0]);ty=dty+(u[1]-ds[1]);apply();
});

document.addEventListener("mouseup",function(){
  if(dragging){dragging=false;VP.classList.remove("panning")}
});

document.getElementById("fit").addEventListener("click",fit);
document.getElementById("zoomin").addEventListener("click",function(){zoomAt(vb.width/2,vb.height/2,1.25)});
document.getElementById("zoomout").addEventListener("click",function(){zoomAt(vb.width/2,vb.height/2,1/1.25)});
document.getElementById("captions").addEventListener("click",function(){svg.classList.toggle("no-captions")});

document.getElementById("export").addEventListener("click",function(){
  var clone=svg.cloneNode(true);
  clone.getElementById("world").setAttribute("transform", "translate(0 0) scale(1)");
  clone.setAttribute("width", Math.round(vb.width*100) + "px");
  clone.setAttribute("height", Math.round(vb.height*100) + "px");
  var style = document.createElement("style");
  style.textContent = document.querySelector("style").textContent;
  clone.insertBefore(style, clone.firstChild);
  var xml=new XMLSerializer().serializeToString(clone);
  var url=URL.createObjectURL(new Blob([xml],{type:"image/svg+xml;charset=utf-8"}));
  var a=document.createElement("a");
  a.download="graph.svg";
  a.href=url;
  a.click();
  URL.revokeObjectURL(url);
});

fit();
})();"""
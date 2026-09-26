from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from .config import JOBS_ROOT

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def jobs_snapshot() -> dict:
    rows = []
    if JOBS_ROOT.exists():
        for root in JOBS_ROOT.iterdir():
            if not root.is_dir():
                continue
            try:
                status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
                request = json.loads((root / "request.json").read_text()) if (root / "request.json").exists() else {}
            except (OSError, json.JSONDecodeError):
                continue
            renders = []
            render_dir = root / "renders"
            if render_dir.exists():
                paths = [p for p in render_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
                for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:32]:
                    renders.append({"name": path.name, "mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()})
            rows.append({"job_id": root.name, "prompt": request.get("prompt", "Untitled"), "state": status.get("state", "unknown"), "stage": status.get("stage", "unknown"), "updated_at": status.get("updated_at"), "renders": renders})
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    active = next((row for row in rows if row["state"] == "running"), rows[0] if rows else None)
    return {"generated_at": datetime.now(UTC).isoformat(), "active": active, "jobs": rows[:12], "counts": {"total": len(rows), "running": sum(row["state"] == "running" for row in rows), "failed": sum(row["state"] == "failed" for row in rows)}}


def public_render(job_id: str, filename: str) -> Path:
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid filename")
    root = (JOBS_ROOT / job_id).resolve()
    path = (root / "renders" / filename).resolve()
    if JOBS_ROOT not in root.parents or root not in path.parents or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(404, "Render not found")
    return path


def dashboard_page() -> HTMLResponse:
    return HTMLResponse("""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3D Modeling AI // Observation Deck</title>
<style>
:root{--bg:#05080d;--panel:#09131e;--line:#173b52;--cyan:#42e8ff;--green:#61ffb0;--muted:#718da1;--text:#daf5ff;--red:#ff5577}*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:radial-gradient(circle at 50% -20%,#12334a 0,#07111a 34%,var(--bg) 70%);color:var(--text);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}
body:after{content:"";position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(66,232,255,.022) 4px)}
.wrap{max-width:1500px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:18px}
.brand{font-size:clamp(24px,4vw,42px);font-weight:900;letter-spacing:.08em}.brand b,h2{color:var(--cyan);text-shadow:0 0 16px #42e8ff66}.sub,.muted{color:var(--muted)}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green);margin-right:8px}.grid{display:grid;grid-template-columns:1.5fr .75fr;gap:18px;margin-top:20px}
.panel{background:linear-gradient(145deg,rgba(11,25,38,.96),rgba(5,11,18,.97));border:1px solid var(--line);padding:18px;box-shadow:inset 0 0 30px #0a263622}
h2{font-size:12px;letter-spacing:.17em;margin:0 0 14px;text-transform:uppercase}.hero{font-size:24px;font-weight:800;margin:4px 0 9px}.stage{color:var(--green);text-transform:uppercase}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{border:1px solid #16374d;padding:13px;background:#07111b}.num{font-size:25px;color:var(--cyan)}
.renders{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.shot{position:relative;aspect-ratio:1;background:#020609;border:1px solid #19445d;overflow:hidden}.shot img{width:100%;height:100%;object-fit:cover}
.shot span{position:absolute;bottom:0;left:0;right:0;padding:7px;background:#02060bdb;color:#a6f0ff;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wide{grid-column:1/-1}
.job{padding:10px 0;border-bottom:1px solid #142b3a}.state{float:right;color:var(--green)}.failed .state{color:var(--red)}.empty{padding:35px;text-align:center;color:#547184;border:1px dashed #19445d}
footer{margin-top:18px;color:#46677c;font-size:11px}@media(max-width:850px){.grid{grid-template-columns:1fr}.renders{grid-template-columns:repeat(2,1fr)}.wrap{padding:15px}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">3D MODELING <b>AI</b></div><div class="sub">AUTONOMOUS BLENDER // OBSERVATION DECK</div></div><div><span class="pulse"></span>LIVE TELEMETRY</div></div>
<div class="grid"><section class="panel"><h2>Active fabrication</h2><div id="active"></div></section><section class="panel"><h2>System telemetry</h2><div class="stats"><div class="stat"><div class="num" id="total">0</div><div class="muted">JOBS</div></div><div class="stat"><div class="num" id="running">0</div><div class="muted">ACTIVE</div></div><div class="stat"><div class="num" id="failed">0</div><div class="muted">FAILED</div></div></div><p class="muted" id="stamp"></p></section>
<section class="panel wide"><h2>Optical capture array // latest angles</h2><div class="renders" id="renders"></div></section><section class="panel wide"><h2>Execution history</h2><div id="jobs"></div></section></div>
<footer>READ-ONLY PUBLIC VIEW // EXECUTION CONTROLS REMAIN TOKEN PROTECTED // AUTO REFRESH 3s</footer></div>
<script>
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function tick(){try{const r=await fetch("/dashboard/api",{cache:"no-store"}),d=await r.json();total.textContent=d.counts.total;running.textContent=d.counts.running;failed.textContent=d.counts.failed;stamp.textContent="LAST SYNC "+new Date(d.generated_at).toLocaleTimeString();const a=d.active;
active.innerHTML=a?'<div class="hero">'+esc(a.prompt)+'</div><div class="stage">◉ '+esc(a.stage)+'</div><p class="muted">JOB '+esc(a.job_id)+' // '+esc(a.state)+'</p>':'<div class="empty">NO FABRICATION JOBS YET</div>';
renders.innerHTML=a&&a.renders.length?a.renders.map(x=>'<div class="shot"><img src="/dashboard/renders/'+encodeURIComponent(a.job_id)+'/'+encodeURIComponent(x.name)+'?v='+encodeURIComponent(x.mtime)+'"><span>'+esc(x.name)+'</span></div>').join(""):'<div class="empty" style="grid-column:1/-1">CAMERAS STANDING BY // CHECKPOINT RENDERS APPEAR HERE</div>';
jobs.innerHTML=d.jobs.length?d.jobs.map(j=>'<div class="job '+esc(j.state)+'"><span class="state">'+esc(j.state)+'</span><b>'+esc(j.prompt)+'</b><br><span class="muted">'+esc(j.stage)+' // '+esc(j.updated_at||"")+'</span></div>').join(""):'<div class="empty">NO HISTORY</div>';}catch(e){stamp.textContent="TELEMETRY LINK LOST"}}
tick();setInterval(tick,3000);
</script></body></html>""")

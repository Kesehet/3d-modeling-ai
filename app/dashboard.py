from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from .config import JOBS_ROOT

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PUBLIC_ARTIFACT_CATEGORIES = {"renders", "scene", "exports"}


def _safe_job_root(job_id: str) -> Path:
    if not job_id or any(ch not in "0123456789abcdef-" for ch in job_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid job id")
    root = (JOBS_ROOT / job_id).resolve()
    if JOBS_ROOT not in root.parents or not root.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    return root


def _artifact_rows(root: Path, category: str) -> list[dict]:
    folder = root / category
    if not folder.exists():
        return []
    rows = []
    for path in sorted(folder.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_file():
            rows.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                }
            )
    return rows


def jobs_snapshot() -> dict:
    rows = []
    if JOBS_ROOT.exists():
        for root in JOBS_ROOT.iterdir():
            if not root.is_dir():
                continue
            try:
                status_path = root / "status.json"
                request_path = root / "request.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                request = json.loads(request_path.read_text()) if request_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                continue

            artifacts = {category: _artifact_rows(root, category) for category in PUBLIC_ARTIFACT_CATEGORIES}
            renders = [
                item
                for item in artifacts["renders"]
                if Path(item["name"]).suffix.lower() in IMAGE_SUFFIXES
            ][:32]
            rows.append(
                {
                    "job_id": root.name,
                    "prompt": request.get("prompt", "Untitled"),
                    "intended_use": request.get("intended_use", "unknown"),
                    "state": status.get("state", "unknown"),
                    "stage": status.get("stage", "unknown"),
                    "updated_at": status.get("updated_at"),
                    "renders": renders,
                    "artifacts": artifacts,
                }
            )

    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    active = next((row for row in rows if row["state"] == "running"), rows[0] if rows else None)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "active": active,
        "jobs": rows[:20],
        "counts": {
            "total": len(rows),
            "running": sum(row["state"] == "running" for row in rows),
            "failed": sum(row["state"] == "failed" for row in rows),
        },
    }


def public_artifact(job_id: str, category: str, filename: str) -> Path:
    if category not in PUBLIC_ARTIFACT_CATEGORIES or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    root = _safe_job_root(job_id)
    path = (root / category / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return path


def public_render(job_id: str, filename: str) -> Path:
    path = public_artifact(job_id, "renders", filename)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Render not found")
    return path


def dashboard_page() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Modeling AI // Observation Deck</title>
<style>
:root{--bg:#05080d;--panel:#09131e;--line:#173b52;--cyan:#42e8ff;--green:#61ffb0;--muted:#718da1;--text:#daf5ff;--red:#ff5577;--amber:#ffc85a}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 50% -20%,#12334a 0,#07111a 34%,var(--bg) 70%);color:var(--text);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}
body:after{content:"";position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(66,232,255,.022) 4px);z-index:99}
.wrap{max-width:1500px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:18px}
.brand{font-size:clamp(24px,4vw,42px);font-weight:900;letter-spacing:.08em}.brand b,h2{color:var(--cyan);text-shadow:0 0 16px #42e8ff66}.sub,.muted{color:var(--muted)}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green);margin-right:8px}.grid{display:grid;grid-template-columns:1.5fr .75fr;gap:18px;margin-top:20px}
.panel{background:linear-gradient(145deg,rgba(11,25,38,.96),rgba(5,11,18,.97));border:1px solid var(--line);padding:18px;box-shadow:inset 0 0 30px #0a263622}
h2{font-size:12px;letter-spacing:.17em;margin:0 0 14px;text-transform:uppercase}.hero{font-size:22px;font-weight:800;margin:4px 0 9px}.stage{color:var(--green);text-transform:uppercase}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{border:1px solid #16374d;padding:13px;background:#07111b}.num{font-size:25px;color:var(--cyan)}
.renders{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.shot{position:relative;aspect-ratio:1;background:#020609;border:1px solid #19445d;overflow:hidden}.shot img{width:100%;height:100%;object-fit:cover;transition:.2s}.shot:hover img{transform:scale(1.03)}
.shot .cap{position:absolute;bottom:0;left:0;right:0;padding:7px;background:#02060be6;color:#a6f0ff;font-size:10px}.shot .cap a{float:right;color:var(--green);text-decoration:none}
.wide{grid-column:1/-1}.job{padding:11px;border:1px solid transparent;border-bottom-color:#142b3a;cursor:pointer}.job:hover,.job.selected{border-color:#23506a;background:#07131f}.state{float:right;color:var(--green)}.failed .state{color:var(--red)}
.empty{padding:35px;text-align:center;color:#547184;border:1px dashed #19445d}.formgrid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px}.field label{display:block;color:var(--muted);font-size:10px;letter-spacing:.12em;margin-bottom:6px}
input,select,textarea{width:100%;background:#040b12;border:1px solid #1b425a;color:var(--text);padding:11px;font:inherit;outline:none}textarea{min-height:74px;resize:vertical}input:focus,select:focus,textarea:focus{border-color:var(--cyan);box-shadow:0 0 0 1px #42e8ff33}
.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}.btn{border:1px solid #24627c;background:#092032;color:var(--cyan);padding:10px 14px;font:inherit;font-weight:800;cursor:pointer;letter-spacing:.05em}.btn:hover{background:#0c2a40}.btn.primary{color:#031118;background:var(--cyan);border-color:var(--cyan)}.btn.green{color:#03140d;background:var(--green);border-color:var(--green)}.btn:disabled{opacity:.45;cursor:not-allowed}
.artgroups{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.artgroup{border:1px solid #16364a;background:#06101a;padding:12px}.artgroup h3{font-size:11px;color:var(--cyan);letter-spacing:.12em}.artifact{display:flex;justify-content:space-between;gap:8px;padding:8px 0;border-top:1px solid #102838}.artifact a{color:var(--green);text-decoration:none}.notice{margin-top:10px;min-height:18px;color:var(--amber)}
footer{margin-top:18px;color:#46677c;font-size:11px}@media(max-width:850px){.grid{grid-template-columns:1fr}.renders{grid-template-columns:repeat(2,1fr)}.artgroups,.formgrid{grid-template-columns:1fr}.wrap{padding:15px}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">3D MODELING <b>AI</b></div><div class="sub">AUTONOMOUS BLENDER // OBSERVATION DECK</div></div><div><span class="pulse"></span>LIVE TELEMETRY</div></div>
<div class="grid">
<section class="panel wide"><h2>New fabrication job</h2>
<div class="formgrid"><div class="field"><label>PROMPT</label><textarea id="prompt" placeholder="Describe the 3D object you want to create...">Create a stylized Pikachu test model</textarea></div>
<div class="field"><label>INTENDED USE</label><select id="intended"><option value="rendering">Rendering</option><option value="3d_printing">3D printing</option><option value="game_asset">Game asset</option></select></div>
<div class="field"><label>TARGET WIDTH MM (OPTIONAL)</label><input id="width" type="number" min="0.01" max="10000" step="0.1" placeholder="e.g. 120"></div></div>
<div class="actions"><button class="btn primary" id="create">CREATE JOB</button><button class="btn green" id="pikachu">CREATE + RUN PIKACHU TEST</button><button class="btn" id="rerun" disabled>RUN PIKACHU ON SELECTED JOB</button></div><div class="notice" id="notice"></div></section>

<section class="panel"><h2>Active fabrication</h2><div id="active"></div></section>
<section class="panel"><h2>System telemetry</h2><div class="stats"><div class="stat"><div class="num" id="total">0</div><div class="muted">JOBS</div></div><div class="stat"><div class="num" id="running">0</div><div class="muted">ACTIVE</div></div><div class="stat"><div class="num" id="failed">0</div><div class="muted">FAILED</div></div></div><p class="muted" id="stamp"></p></section>

<section class="panel wide"><h2>Optical capture array // selected job</h2><div class="renders" id="renders"></div></section>
<section class="panel wide"><h2>Generated files // gallery & downloads</h2><div class="artgroups" id="artifacts"></div></section>
<section class="panel wide"><h2>Recent jobs // click to inspect</h2><div id="jobs"></div></section>
</div><footer>OPEN PROTOTYPE // JOB CREATION + TEST EXECUTION + GENERATED FILE DOWNLOADS ENABLED // AUTO REFRESH 3s</footer></div>
<script>
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=n=>n<1024?n+" B":n<1048576?(n/1024).toFixed(1)+" KB":(n/1048576).toFixed(1)+" MB";
let selectedJob=null,lastData=null,busy=false;
function jobFromData(d){return d.jobs.find(j=>j.job_id===selectedJob)||d.active||d.jobs[0]||null}
function fileUrl(job,cat,name){return "/dashboard/artifacts/"+encodeURIComponent(job)+"/"+encodeURIComponent(cat)+"/"+encodeURIComponent(name)}
function renderUI(d){lastData=d;total.textContent=d.counts.total;running.textContent=d.counts.running;failed.textContent=d.counts.failed;stamp.textContent="LAST SYNC "+new Date(d.generated_at).toLocaleTimeString();
const current=jobFromData(d);if(current&&!selectedJob)selectedJob=current.job_id;rerun.disabled=!current||busy;
active.innerHTML=d.active?'<div class="hero">'+esc(d.active.prompt)+'</div><div class="stage">◉ '+esc(d.active.stage)+'</div><p class="muted">JOB '+esc(d.active.job_id)+' // '+esc(d.active.state)+'</p>':'<div class="empty">NO JOBS RUNNING</div>';
renders.innerHTML=current&&current.renders.length?current.renders.map(x=>'<div class="shot"><a target="_blank" href="/dashboard/renders/'+encodeURIComponent(current.job_id)+'/'+encodeURIComponent(x.name)+'"><img loading="lazy" src="/dashboard/renders/'+encodeURIComponent(current.job_id)+'/'+encodeURIComponent(x.name)+'?v='+encodeURIComponent(x.mtime)+'"></a><div class="cap">'+esc(x.name)+'<a download href="'+fileUrl(current.job_id,"renders",x.name)+'">DOWNLOAD</a></div></div>').join(""):'<div class="empty" style="grid-column:1/-1">CAMERAS STANDING BY // RENDERS APPEAR HERE</div>';
if(current){const groups=["renders","scene","exports"];artifacts.innerHTML=groups.map(cat=>{const files=current.artifacts[cat]||[];return '<div class="artgroup"><h3>'+cat.toUpperCase()+' // '+files.length+'</h3>'+(files.length?files.map(f=>'<div class="artifact"><span>'+esc(f.name)+' <small class="muted">'+fmt(f.bytes)+'</small></span><a download href="'+fileUrl(current.job_id,cat,f.name)+'">DOWNLOAD</a></div>').join(""):'<div class="muted">No files yet</div>')+'</div>'}).join("")}else{artifacts.innerHTML='<div class="empty" style="grid-column:1/-1">NO GENERATED FILES YET</div>'}
jobs.innerHTML=d.jobs.length?d.jobs.map(j=>'<div data-job="'+esc(j.job_id)+'" class="job '+esc(j.state)+(j.job_id===selectedJob?' selected':'')+'"><span class="state">'+esc(j.state)+'</span><b>'+esc(j.prompt)+'</b><br><span class="muted">'+esc(j.stage)+' // '+esc(j.intended_use)+' // '+esc(j.updated_at||"")+'</span></div>').join(""):'<div class="empty">NO HISTORY</div>';
document.querySelectorAll("[data-job]").forEach(el=>el.onclick=()=>{selectedJob=el.dataset.job;renderUI(lastData)})}
async function tick(){try{const r=await fetch("/dashboard/api",{cache:"no-store"});renderUI(await r.json())}catch(e){stamp.textContent="TELEMETRY LINK LOST"}}
async function newJob(runTest){if(busy)return;const p=prompt.value.trim();if(!p){notice.textContent="Prompt is required.";return}busy=true;create.disabled=pikachu.disabled=rerun.disabled=true;notice.textContent="CREATING JOB...";
try{const body={prompt:p,intended_use:intended.value};if(width.value)body.target_width_mm=Number(width.value);let r=await fetch("/dashboard/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());const d=await r.json();selectedJob=d.job_id;notice.textContent="JOB "+d.job_id+" CREATED";await tick();if(runTest){notice.textContent="PIKACHU BUILD RUNNING // WATCH THE TELEMETRY + GALLERY...";r=await fetch("/dashboard/jobs/"+encodeURIComponent(d.job_id)+"/pikachu",{method:"POST"});if(!r.ok)throw new Error(await r.text());notice.textContent="PIKACHU BUILD COMPLETE // FILES READY"}}
catch(e){notice.textContent="ERROR // "+e.message}finally{busy=false;create.disabled=pikachu.disabled=false;rerun.disabled=!selectedJob;await tick()}}
async function runSelected(){if(!selectedJob||busy)return;busy=true;create.disabled=pikachu.disabled=rerun.disabled=true;notice.textContent="PIKACHU BUILD RUNNING ON "+selectedJob+"...";
try{const r=await fetch("/dashboard/jobs/"+encodeURIComponent(selectedJob)+"/pikachu",{method:"POST"});if(!r.ok)throw new Error(await r.text());notice.textContent="PIKACHU BUILD COMPLETE // FILES READY"}catch(e){notice.textContent="ERROR // "+e.message}finally{busy=false;create.disabled=pikachu.disabled=false;rerun.disabled=false;await tick()}}
create.onclick=()=>newJob(false);pikachu.onclick=()=>newJob(true);rerun.onclick=runSelected;tick();setInterval(tick,3000);
</script></body></html>"""
    )

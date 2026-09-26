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

            if request.get("prompt") == "Deployment MCP smoke test cube":
                continue

            artifacts = {category: _artifact_rows(root, category) for category in PUBLIC_ARTIFACT_CATEGORIES}
            renders = [
                item
                for item in artifacts["renders"]
                if Path(item["name"]).suffix.lower() in IMAGE_SUFFIXES
            ][:64]

            history: list[dict] = []
            history_path = root / "history.json"
            if history_path.exists():
                try:
                    parsed_history = json.loads(history_path.read_text(encoding="utf-8"))
                    if isinstance(parsed_history, list):
                        history = parsed_history[-100:]
                except (OSError, json.JSONDecodeError):
                    history = []

            latest_qa: dict | None = None
            latest_qa_file: str | None = None
            qa_files = sorted(
                (root / "exports").glob("*-qa.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for qa_path in qa_files:
                try:
                    parsed_qa = json.loads(qa_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(parsed_qa, dict):
                    latest_qa = parsed_qa
                    latest_qa_file = qa_path.name
                    break

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
                    "history": history,
                    "iterations": [
                        event for event in history if event.get("event") == "pikachu_iteration"
                    ],
                    "qa": latest_qa,
                    "qa_file": latest_qa_file,
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
        r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Modeling AI</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--text:#172033;--muted:#687386;--blue:#2563eb;--green:#15803d;--red:#b42318;--shadow:0 8px 24px rgba(15,23,42,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select,textarea{font:inherit}button{cursor:pointer}.wrap{max-width:1280px;margin:auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:24px}.brand h1{margin:0;font-size:24px}.brand p{margin:3px 0 0;color:var(--muted)}
.btn{border:1px solid var(--line);background:#fff;color:var(--text);border-radius:9px;padding:9px 13px;font-weight:650}.btn:hover{background:#f8fafc}.btn.primary{background:var(--blue);color:#fff;border-color:var(--blue)}.btn.danger{border-color:#fecaca;color:#b42318;background:#fff5f5}.btn.danger:hover{background:#fee2e2}
.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}.job-card{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);cursor:pointer;transition:.15s}.job-card:hover{transform:translateY(-2px)}
.thumb{aspect-ratio:1;background:#eef1f5;position:relative;overflow:hidden}.thumb img{width:100%;height:100%;object-fit:cover;display:block}.empty-thumb{width:100%;height:100%;display:grid;place-items:center;color:#98a2b3;font-weight:650}
.card-body{padding:13px}.prompt{font-weight:700;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:42px}.meta{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px;margin-top:8px}.status{font-weight:700}.ready{color:var(--green)}.failed{color:var(--red)}.running{color:#b54708}
.detail{display:none}.detail.show{display:block}.home.hidden{display:none}.detail-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px}.detail-head h2{margin:8px 0 4px;font-size:24px}.detail-head p{margin:0;color:var(--muted)}
.render-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.render{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}.render img{width:100%;aspect-ratio:1;object-fit:cover;display:block}.render-name{padding:9px 11px;color:var(--muted);font-size:12px}
.downloads{margin-top:22px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}.downloads h3{margin:0 0 10px}.file-list{display:flex;gap:8px;flex-wrap:wrap}.file{display:inline-flex;text-decoration:none;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#fafafa}.file b{margin-right:6px}
.details{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}.details summary{cursor:pointer;font-weight:700}.details pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;padding:12px;border-radius:8px;max-height:320px;overflow:auto}
.dialog-backdrop{display:none;position:fixed;inset:0;background:#11182788;z-index:50;padding:20px;align-items:center;justify-content:center}.dialog-backdrop.show{display:flex}.dialog{width:min(620px,100%);background:#fff;border-radius:14px;padding:20px;box-shadow:0 25px 80px #0003}.dialog h2{margin:0 0 14px}.field{margin-bottom:12px}.field label{display:block;font-weight:650;margin-bottom:5px}.field textarea,.field input,.field select{width:100%;border:1px solid #cfd4dc;border-radius:8px;padding:10px}.field textarea{min-height:110px;resize:vertical}.actions{display:flex;justify-content:flex-end;gap:8px}.notice{margin-top:10px;color:var(--muted)}.empty-state{padding:50px;text-align:center;color:var(--muted)}
@media(max-width:980px){.gallery{grid-template-columns:repeat(3,1fr)}.render-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.wrap{padding:14px}.gallery{grid-template-columns:repeat(2,1fr);gap:10px}.render-grid{grid-template-columns:1fr}.brand h1{font-size:20px}.job-card{border-radius:10px}}
</style>
</head>
<body>
<div class="wrap">
<header><div class="brand"><h1>3D Modeling AI</h1><p>Your generated models</p></div><button class="btn primary" id="newJobBtn">+ New Job</button></header>

<section class="home" id="homeView"><div class="gallery" id="jobGallery"></div></section>

<section class="detail" id="detailView">
  <div class="detail-head">
    <div><button class="btn" id="backBtn">← Gallery</button><h2 id="detailTitle">Job</h2><p id="detailMeta"></p></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn" id="improveBtn">Improve model</button><button class="btn danger" id="deleteBtn">Delete job</button></div>
  </div>
  <div class="render-grid" id="renderGrid"></div>
  <div class="downloads"><h3>Downloads</h3><div class="file-list" id="fileList"></div></div>
  <details class="details"><summary>More details</summary><pre id="detailJson"></pre></details>
</section>
</div>

<div class="dialog-backdrop" id="newJobDialog">
  <div class="dialog">
    <h2>New model</h2>
    <div class="field"><label for="jobPrompt">What should we make?</label><textarea id="jobPrompt" placeholder="Describe the object or character..."></textarea></div>
    <div class="field"><label for="jobUse">Use</label><select id="jobUse"><option value="rendering">Rendering</option><option value="3d_printing">3D printing</option><option value="game_asset">Game asset</option></select></div>
    <div class="field"><label for="jobWidth">Width in mm (optional)</label><input id="jobWidth" type="number" min="0.01" max="10000" step="0.1"></div>
    <div class="actions"><button class="btn" id="cancelNew">Cancel</button><button class="btn primary" id="createModel">Create model</button></div>
    <div class="notice" id="newNotice"></div>
  </div>
</div>

<script>
const byId=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const els={
  home:byId("homeView"),detail:byId("detailView"),gallery:byId("jobGallery"),
  title:byId("detailTitle"),meta:byId("detailMeta"),renders:byId("renderGrid"),files:byId("fileList"),json:byId("detailJson"),
  newBtn:byId("newJobBtn"),dialog:byId("newJobDialog"),cancel:byId("cancelNew"),create:byId("createModel"),
  prompt:byId("jobPrompt"),use:byId("jobUse"),width:byId("jobWidth"),notice:byId("newNotice"),
  back:byId("backBtn"),improve:byId("improveBtn"),deleteBtn:byId("deleteBtn")
};
let data=null;
let currentJob=null;
let busy=false;

function fileUrl(job,category,name){return "/dashboard/artifacts/"+encodeURIComponent(job)+"/"+encodeURIComponent(category)+"/"+encodeURIComponent(name)}
function renderUrl(job,name,mtime){return "/dashboard/renders/"+encodeURIComponent(job)+"/"+encodeURIComponent(name)+"?v="+encodeURIComponent(mtime||"")}
function latestImage(job){return (job.renders||[])[0]||null}
function route(){
  const match=location.hash.match(/^#job\/([0-9a-f-]+)$/i);
  currentJob=match&&data?data.jobs.find(job=>job.job_id===match[1])||null:null;
  render();
}
function goHome(){history.pushState(null,"",location.pathname);currentJob=null;render()}
function openJob(id){location.hash="job/"+id}
function render(){
  if(!data)return;
  els.home.classList.toggle("hidden",!!currentJob);
  els.detail.classList.toggle("show",!!currentJob);
  if(!currentJob){renderGallery();return}
  renderDetail(currentJob);
}
function renderGallery(){
  if(!data.jobs.length){els.gallery.innerHTML='<div class="empty-state" style="grid-column:1/-1">No models yet. Create the first one.</div>';return}
  els.gallery.innerHTML=data.jobs.map(job=>{
    const image=latestImage(job);
    const thumb=image?'<img loading="lazy" src="'+renderUrl(job.job_id,image.name,image.mtime)+'">':'<div class="empty-thumb">No render yet</div>';
    return '<article class="job-card" data-job="'+esc(job.job_id)+'"><div class="thumb">'+thumb+'</div><div class="card-body"><div class="prompt">'+esc(job.prompt)+'</div><div class="meta"><span class="status '+esc(job.state)+'">'+esc(job.state)+'</span><span>'+esc(job.stage)+'</span></div></div></article>';
  }).join("");
  document.querySelectorAll("[data-job]").forEach(card=>card.addEventListener("click",()=>openJob(card.dataset.job)));
}
function renderDetail(job){
  els.title.textContent=job.prompt;
  els.meta.textContent=(job.state||"")+" · "+(job.stage||"")+" · "+job.job_id;
  els.renders.innerHTML=(job.renders||[]).length
    ? job.renders.map(image=>'<div class="render"><a target="_blank" href="'+renderUrl(job.job_id,image.name,image.mtime)+'"><img loading="lazy" src="'+renderUrl(job.job_id,image.name,image.mtime)+'"></a><div class="render-name">'+esc(image.name)+'</div></div>').join("")
    : '<div class="empty-state" style="grid-column:1/-1">No renders yet. This page refreshes automatically while the job runs.</div>';
  const scene=(job.artifacts?.scene||[]).filter(file=>/\.(blend)$/i.test(file.name));
  const exports=(job.artifacts?.exports||[]).filter(file=>/\.(glb|obj|stl|3mf|json)$/i.test(file.name));
  const files=[...scene.map(file=>["scene",file]),...exports.map(file=>["exports",file])];
  els.files.innerHTML=files.length
    ? files.map(([category,file])=>'<a class="file" download href="'+fileUrl(job.job_id,category,file.name)+'"><b>↓</b>'+esc(file.name)+'</a>').join("")
    : '<span style="color:var(--muted)">No downloadable model files yet.</span>';
  els.json.textContent=JSON.stringify({status:{state:job.state,stage:job.stage},qa:job.qa,history:job.history},null,2);
  els.improve.disabled=busy||!scene.length;
  els.deleteBtn.disabled=busy||job.state==="running";
}
async function refresh(){
  try{
    const response=await fetch("/dashboard/api",{cache:"no-store"});
    if(!response.ok)throw new Error("Dashboard API "+response.status);
    data=await response.json();
    if(currentJob)currentJob=data.jobs.find(job=>job.job_id===currentJob.job_id)||currentJob;
    route();
  }catch(error){console.error(error)}
}
async function createModel(){
  if(busy)return;
  const prompt=els.prompt.value.trim();
  if(!prompt){els.notice.textContent="Please describe what to make.";return}
  busy=true;els.create.disabled=true;els.notice.textContent="Creating model...";
  try{
    const body={prompt,intended_use:els.use.value};
    if(els.width.value)body.target_width_mm=Number(els.width.value);
    let response=await fetch("/dashboard/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!response.ok)throw new Error(await response.text());
    const created=await response.json();
    els.dialog.classList.remove("show");
    await refresh();
    openJob(created.job_id);
    response=await fetch("/dashboard/jobs/"+encodeURIComponent(created.job_id)+"/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_research:true})});
    if(!response.ok)throw new Error(await response.text());
    await refresh();
  }catch(error){els.notice.textContent="Error: "+error.message;els.dialog.classList.add("show")}
  finally{busy=false;els.create.disabled=false}
}
async function improve(){
  if(!currentJob||busy)return;
  busy=true;els.improve.disabled=true;els.improve.textContent="Improving...";
  try{
    const response=await fetch("/dashboard/jobs/"+encodeURIComponent(currentJob.job_id)+"/improve",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({iterations:1})});
    if(!response.ok)throw new Error(await response.text());
    await refresh();
  }catch(error){alert("Improve failed: "+error.message)}
  finally{busy=false;els.improve.textContent="Improve model";render()}
}
async function deleteCurrentJob(){
  if(!currentJob||busy)return;
  const prompt=currentJob.prompt||"this job";
  if(!confirm('Delete "'+prompt+'"?\n\nThis permanently removes all renders, model files, exports, references, logs and history for this job.'))return;
  busy=true;els.deleteBtn.disabled=true;els.deleteBtn.textContent="Deleting...";
  try{
    const response=await fetch("/dashboard/jobs/"+encodeURIComponent(currentJob.job_id),{method:"DELETE"});
    if(!response.ok)throw new Error(await response.text());
    goHome();
    await refresh();
  }catch(error){alert("Delete failed: "+error.message)}
  finally{busy=false;els.deleteBtn.textContent="Delete job";render()}
}
els.newBtn.addEventListener("click",()=>{els.notice.textContent="";els.dialog.classList.add("show");els.prompt.focus()});
els.cancel.addEventListener("click",()=>els.dialog.classList.remove("show"));
els.dialog.addEventListener("click",event=>{if(event.target===els.dialog)els.dialog.classList.remove("show")});
els.create.addEventListener("click",createModel);
els.back.addEventListener("click",goHome);
els.improve.addEventListener("click",improve);
els.deleteBtn.addEventListener("click",deleteCurrentJob);
window.addEventListener("hashchange",route);
refresh();
setInterval(refresh,3000);
</script>
</body></html>"""
    )

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
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Modeling AI</title>
<style>
:root{
  --bg:#f4f6f8;--surface:#ffffff;--surface-2:#f8fafc;--border:#dfe3e8;
  --text:#182230;--muted:#667085;--primary:#2563eb;--primary-dark:#1d4ed8;
  --success:#15803d;--success-bg:#dcfce7;--danger:#b42318;--danger-bg:#fee4e2;
  --warning:#b54708;--warning-bg:#fef0c7;--shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.1)
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select,textarea{font:inherit}
a{color:var(--primary)}
.app{max-width:1440px;margin:0 auto;min-height:100vh}
.header{height:68px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 24px;position:sticky;top:0;z-index:20}
.brand{display:flex;align-items:center;gap:12px}.logo{width:36px;height:36px;border-radius:9px;background:var(--primary);color:white;display:grid;place-items:center;font-weight:800;font-size:16px}.brand h1{font-size:18px;margin:0}.brand p{margin:1px 0 0;color:var(--muted);font-size:12px}
.header-right{display:flex;align-items:center;gap:10px}.live{display:flex;align-items:center;gap:7px;color:var(--success);font-weight:600;font-size:12px}.live-dot{width:8px;height:8px;border-radius:50%;background:#22c55e}.selected-chip{max-width:360px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid var(--border);border-radius:8px;padding:7px 10px;background:var(--surface-2);color:var(--muted);font-size:12px}
.nav{background:var(--surface);border-bottom:1px solid var(--border);display:flex;gap:4px;padding:0 24px;overflow:auto;position:sticky;top:68px;z-index:19}
.tab-btn{appearance:none;border:0;background:transparent;color:var(--muted);padding:14px 14px 12px;border-bottom:2px solid transparent;font-weight:600;cursor:pointer;white-space:nowrap}.tab-btn:hover{color:var(--text)}.tab-btn.active{color:var(--primary);border-bottom-color:var(--primary)}
.main{padding:24px}.tab{display:none}.tab.active{display:block}
.page-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:18px}.page-head h2{font-size:22px;margin:0 0 4px}.page-head p{margin:0;color:var(--muted)}
.grid{display:grid;gap:16px}.grid-4{grid-template-columns:repeat(4,minmax(0,1fr))}.grid-2{grid-template-columns:repeat(2,minmax(0,1fr))}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);padding:18px}.card h3{font-size:14px;margin:0 0 12px}
.stat-label{color:var(--muted);font-size:12px}.stat-value{font-size:28px;font-weight:700;margin-top:4px}.stat-note{color:var(--muted);font-size:11px;margin-top:2px}
.badge{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:700}.badge.running,.badge.ready{background:var(--success-bg);color:var(--success)}.badge.failed{background:var(--danger-bg);color:var(--danger)}.badge.created,.badge.unknown{background:#eef2f6;color:#475467}
.active-title{font-size:18px;font-weight:700;margin:6px 0}.meta{color:var(--muted);font-size:12px;word-break:break-all}
.form-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px}.field label{display:block;font-weight:600;margin-bottom:6px}.field small{display:block;color:var(--muted);margin-top:5px}.field textarea,.field input,.field select{width:100%;border:1px solid #cfd4dc;border-radius:8px;background:white;color:var(--text);padding:10px 11px;outline:none}.field textarea{min-height:120px;resize:vertical}.field textarea:focus,.field input:focus,.field select:focus{border-color:#84adff;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.btn{appearance:none;border:1px solid #cfd4dc;background:white;color:var(--text);border-radius:8px;padding:9px 13px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;justify-content:center;gap:6px}.btn:hover{background:#f8fafc}.btn.primary{background:var(--primary);border-color:var(--primary);color:white}.btn.primary:hover{background:var(--primary-dark)}.btn.success{background:#16a34a;border-color:#16a34a;color:white}.btn:disabled{opacity:.5;cursor:not-allowed}
.notice{margin-top:14px;border-radius:8px;padding:10px 12px;background:#f8fafc;color:var(--muted);display:none}.notice.show{display:block}.notice.error{background:var(--danger-bg);color:var(--danger)}.notice.success{background:var(--success-bg);color:var(--success)}.notice.busy{background:var(--warning-bg);color:var(--warning)}
.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.shot{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;box-shadow:var(--shadow)}.shot .imgwrap{aspect-ratio:1;background:#e8edf2;display:block}.shot img{width:100%;height:100%;object-fit:cover;display:block}.shot-info{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px}.shot-name{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.download{font-size:12px;text-decoration:none;font-weight:600}
.empty{padding:36px 18px;text-align:center;color:var(--muted);border:1px dashed #cbd5e1;border-radius:10px;background:#fafbfc}
.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse}.table th,.table td{text-align:left;padding:11px 12px;border-bottom:1px solid #eaecf0;vertical-align:middle}.table th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;background:#f9fafb}.table tr.clickable{cursor:pointer}.table tr.clickable:hover{background:#f8fafc}.table tr.selected{background:#eff6ff}.prompt-cell{max-width:520px}.prompt-cell strong{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.small{font-size:12px;color:var(--muted)}
.files-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.file-group{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}.file-group-head{display:flex;justify-content:space-between;padding:12px 14px;background:#f9fafb;border-bottom:1px solid var(--border);font-weight:700}.file-row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 14px;border-bottom:1px solid #eef0f3}.file-row:last-child{border-bottom:0}.file-name{min-width:0}.file-name span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.file-name small{color:var(--muted)}
.job-picker{display:flex;align-items:center;gap:10px}.job-picker select{max-width:460px;border:1px solid #cfd4dc;border-radius:8px;background:white;padding:9px 10px}
.recent-list{display:grid;gap:10px}.recent-item{display:flex;justify-content:space-between;gap:12px;padding:12px;border:1px solid var(--border);border-radius:9px;cursor:pointer}.recent-item:hover{background:#f8fafc}.recent-main{min-width:0}.recent-main strong{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.footer{padding:20px 24px 30px;color:var(--muted);font-size:12px}
@media(max-width:980px){.grid-4{grid-template-columns:repeat(2,1fr)}.grid-2,.form-grid,.files-grid{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.header{padding:0 14px}.brand p,.selected-chip{display:none}.nav{padding:0 10px}.main{padding:14px}.grid-4{grid-template-columns:1fr 1fr}.gallery{grid-template-columns:1fr}.page-head{flex-direction:column}.job-picker{width:100%;align-items:stretch;flex-direction:column}.job-picker select{max-width:none;width:100%}}
</style>
</head>
<body>
<div class="app">
<header class="header">
  <div class="brand"><div class="logo">3D</div><div><h1>3D Modeling AI</h1><p>Blender generation dashboard</p></div></div>
  <div class="header-right"><div class="live"><span class="live-dot"></span>Live</div><div class="selected-chip" id="selectedChip">No job selected</div></div>
</header>
<nav class="nav" id="nav">
  <button class="tab-btn active" data-tab="overview">Overview</button>
  <button class="tab-btn" data-tab="new">New Job</button>
  <button class="tab-btn" data-tab="gallery">Gallery</button>
  <button class="tab-btn" data-tab="files">Files</button>
  <button class="tab-btn" data-tab="jobs">Jobs</button>
</nav>
<main class="main">

<section class="tab active" id="tab-overview">
  <div class="page-head"><div><h2>Overview</h2><p>Current service status and recent activity.</p></div><button class="btn primary" data-go="new">+ New Job</button></div>
  <div class="grid grid-4">
    <div class="card"><div class="stat-label">Total jobs</div><div class="stat-value" id="total">0</div><div class="stat-note">Jobs stored on this server</div></div>
    <div class="card"><div class="stat-label">Running</div><div class="stat-value" id="running">0</div><div class="stat-note">Currently processing</div></div>
    <div class="card"><div class="stat-label">Failed</div><div class="stat-value" id="failed">0</div><div class="stat-note">Jobs that need attention</div></div>
    <div class="card"><div class="stat-label">Last refresh</div><div class="stat-value" style="font-size:18px" id="stamp">—</div><div class="stat-note">Auto-refreshes every 3 seconds</div></div>
  </div>
  <div class="grid grid-2" style="margin-top:16px">
    <div class="card"><h3>Active job</h3><div id="active"></div></div>
    <div class="card"><h3>Recent jobs</h3><div class="recent-list" id="recent"></div></div>
  </div>
</section>

<section class="tab" id="tab-new">
  <div class="page-head"><div><h2>New Job</h2><p>Create a new modeling job or run the Pikachu test preset.</p></div></div>
  <div class="card">
    <div class="form-grid">
      <div class="field"><label for="prompt">Prompt</label><textarea id="prompt" placeholder="Describe the 3D object you want to create...">Create a stylized Pikachu test model</textarea><small>Describe shape, proportions, style and intended result.</small></div>
      <div class="field"><label for="intended">Intended use</label><select id="intended"><option value="rendering">Rendering</option><option value="3d_printing">3D printing</option><option value="game_asset">Game asset</option></select></div>
      <div class="field"><label for="width">Target width (mm)</label><input id="width" type="number" min="0.01" max="10000" step="0.1" placeholder="Optional"></div>
    </div>
    <div class="actions"><button class="btn primary" id="create">Create Job</button><button class="btn success" id="pikachu">Create + Run Pikachu Test</button></div>
    <div class="notice" id="notice"></div>
  </div>
</section>

<section class="tab" id="tab-gallery">
  <div class="page-head">
    <div><h2>Gallery</h2><p>Rendered views from the selected job.</p></div>
    <div class="job-picker"><select id="galleryJob"></select><button class="btn" id="rerun" disabled>Run Pikachu Test</button></div>
  </div>
  <div class="gallery" id="renders"></div>
</section>

<section class="tab" id="tab-files">
  <div class="page-head">
    <div><h2>Files</h2><p>Download generated scenes, renders and exports.</p></div>
    <div class="job-picker"><select id="filesJob"></select></div>
  </div>
  <div class="files-grid" id="artifacts"></div>
</section>

<section class="tab" id="tab-jobs">
  <div class="page-head"><div><h2>Jobs</h2><p>Select a job to inspect its renders and generated files.</p></div><button class="btn primary" data-go="new">+ New Job</button></div>
  <div class="card" style="padding:0"><div class="table-wrap"><table class="table"><thead><tr><th>Status</th><th>Prompt</th><th>Use</th><th>Stage</th><th>Updated</th><th></th></tr></thead><tbody id="jobs"></tbody></table></div></div>
</section>

</main>
<div class="footer">Open prototype · job creation, test execution and downloads are currently public.</div>
</div>
<script>
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=n=>n<1024?n+" B":n<1048576?(n/1024).toFixed(1)+" KB":(n/1048576).toFixed(1)+" MB";
const dateFmt=v=>{if(!v)return "—";try{return new Date(v).toLocaleString()}catch(e){return v}};
let selectedJob=localStorage.getItem("selected3dJob")||null,lastData=null,busy=false,currentTab=localStorage.getItem("selected3dTab")||"overview";

function setTab(name){
  currentTab=name;localStorage.setItem("selected3dTab",name);
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x.id==="tab-"+name));
  document.querySelectorAll(".tab-btn").forEach(x=>x.classList.toggle("active",x.dataset.tab===name));
}
document.querySelectorAll(".tab-btn").forEach(x=>x.onclick=()=>setTab(x.dataset.tab));
document.querySelectorAll("[data-go]").forEach(x=>x.onclick=()=>setTab(x.dataset.go));
setTab(currentTab);

function jobFromData(d){return d.jobs.find(j=>j.job_id===selectedJob)||d.active||d.jobs[0]||null}
function fileUrl(job,cat,name){return "/dashboard/artifacts/"+encodeURIComponent(job)+"/"+encodeURIComponent(cat)+"/"+encodeURIComponent(name)}
function selectJob(id,goTab){
  selectedJob=id;localStorage.setItem("selected3dJob",id);
  if(lastData)renderUI(lastData);
  if(goTab)setTab(goTab);
}
function badge(j){return '<span class="badge '+esc(j.state)+'">'+esc(j.state)+'</span>'}
function showNotice(message,type){
  notice.textContent=message;notice.className="notice show "+(type||"");
}
function populateSelectors(d,current){
  const options=d.jobs.length?d.jobs.map(j=>'<option value="'+esc(j.job_id)+'">'+esc(j.prompt).slice(0,70)+' · '+esc(j.state)+'</option>').join(""):'<option value="">No jobs yet</option>';
  [galleryJob,filesJob].forEach(sel=>{const old=sel.value;sel.innerHTML=options;if(current)sel.value=current.job_id;else if(old)sel.value=old});
}

function renderUI(d){
  lastData=d;
  total.textContent=d.counts.total;running.textContent=d.counts.running;failed.textContent=d.counts.failed;
  stamp.textContent=new Date(d.generated_at).toLocaleTimeString();
  const current=jobFromData(d);
  if(current&&!selectedJob){selectedJob=current.job_id;localStorage.setItem("selected3dJob",selectedJob)}
  selectedChip.textContent=current?"Selected: "+current.prompt:"No job selected";
  populateSelectors(d,current);
  rerun.disabled=!current||busy;

  active.innerHTML=d.active
    ? badge(d.active)+'<div class="active-title">'+esc(d.active.prompt)+'</div><div class="meta">'+esc(d.active.stage)+' · '+esc(d.active.job_id)+'</div><div class="actions"><button class="btn" onclick="selectJob(\''+esc(d.active.job_id)+'\',\'gallery\')">View Gallery</button></div>'
    : '<div class="empty">No jobs are currently running.</div>';

  recent.innerHTML=d.jobs.length
    ? d.jobs.slice(0,5).map(j=>'<div class="recent-item" data-select="'+esc(j.job_id)+'"><div class="recent-main"><strong>'+esc(j.prompt)+'</strong><span class="small">'+esc(j.stage)+' · '+dateFmt(j.updated_at)+'</span></div>'+badge(j)+'</div>').join("")
    : '<div class="empty">No jobs yet.</div>';

  renders.innerHTML=current&&current.renders.length
    ? current.renders.map(x=>'<div class="shot"><a class="imgwrap" target="_blank" href="/dashboard/renders/'+encodeURIComponent(current.job_id)+'/'+encodeURIComponent(x.name)+'"><img loading="lazy" src="/dashboard/renders/'+encodeURIComponent(current.job_id)+'/'+encodeURIComponent(x.name)+'?v='+encodeURIComponent(x.mtime)+'"></a><div class="shot-info"><span class="shot-name">'+esc(x.name)+'</span><a class="download" download href="'+fileUrl(current.job_id,"renders",x.name)+'">Download</a></div></div>').join("")
    : '<div class="empty" style="grid-column:1/-1">No renders are available for the selected job yet.</div>';

  if(current){
    const groups=[["renders","Renders"],["scene","Scene files"],["exports","Exports"]];
    artifacts.innerHTML=groups.map(([cat,label])=>{const files=current.artifacts[cat]||[];return '<div class="file-group"><div class="file-group-head"><span>'+label+'</span><span class="small">'+files.length+' file'+(files.length===1?'':'s')+'</span></div>'+(files.length?files.map(f=>'<div class="file-row"><div class="file-name"><span>'+esc(f.name)+'</span><small>'+fmt(f.bytes)+'</small></div><a class="btn" download href="'+fileUrl(current.job_id,cat,f.name)+'">Download</a></div>').join(""):'<div class="file-row"><span class="small">No files yet</span></div>')+'</div>'}).join("");
  }else{
    artifacts.innerHTML='<div class="empty" style="grid-column:1/-1">Create or select a job to see generated files.</div>';
  }

  jobs.innerHTML=d.jobs.length
    ? d.jobs.map(j=>'<tr class="clickable '+(j.job_id===selectedJob?'selected':'')+'" data-select="'+esc(j.job_id)+'"><td>'+badge(j)+'</td><td class="prompt-cell"><strong>'+esc(j.prompt)+'</strong><span class="small">'+esc(j.job_id)+'</span></td><td>'+esc(j.intended_use)+'</td><td>'+esc(j.stage)+'</td><td>'+dateFmt(j.updated_at)+'</td><td><button class="btn" data-gallery="'+esc(j.job_id)+'">Gallery</button></td></tr>').join("")
    : '<tr><td colspan="6"><div class="empty">No jobs yet.</div></td></tr>';

  document.querySelectorAll("[data-select]").forEach(el=>el.onclick=e=>{if(e.target.closest("[data-gallery]"))return;selectJob(el.dataset.select)});
  document.querySelectorAll("[data-gallery]").forEach(el=>el.onclick=e=>{e.stopPropagation();selectJob(el.dataset.gallery,"gallery")});
}

async function tick(){
  try{
    const r=await fetch("/dashboard/api",{cache:"no-store"});
    if(!r.ok)throw new Error("Dashboard API "+r.status);
    renderUI(await r.json());
  }catch(e){stamp.textContent="Offline"}
}

async function newJob(runTest){
  if(busy)return;
  const p=prompt.value.trim();
  if(!p){showNotice("Prompt is required.","error");return}
  busy=true;create.disabled=pikachu.disabled=rerun.disabled=true;showNotice("Creating job...","busy");
  try{
    const body={prompt:p,intended_use:intended.value};if(width.value)body.target_width_mm=Number(width.value);
    let r=await fetch("/dashboard/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok)throw new Error(await r.text());
    const d=await r.json();selectedJob=d.job_id;localStorage.setItem("selected3dJob",selectedJob);showNotice("Job created successfully.","success");await tick();
    if(runTest){
      showNotice("Running Pikachu test. This can take a little while...","busy");setTab("gallery");
      r=await fetch("/dashboard/jobs/"+encodeURIComponent(d.job_id)+"/pikachu",{method:"POST"});
      if(!r.ok)throw new Error(await r.text());
      showNotice("Pikachu test completed. Generated files are ready.","success");await tick();
    }
  }catch(e){showNotice("Error: "+e.message,"error")}
  finally{busy=false;create.disabled=pikachu.disabled=false;rerun.disabled=!selectedJob;await tick()}
}

async function runSelected(){
  if(!selectedJob||busy)return;
  busy=true;create.disabled=pikachu.disabled=rerun.disabled=true;showNotice("Running Pikachu test on selected job...","busy");setTab("gallery");
  try{
    const r=await fetch("/dashboard/jobs/"+encodeURIComponent(selectedJob)+"/pikachu",{method:"POST"});
    if(!r.ok)throw new Error(await r.text());
    showNotice("Pikachu test completed.","success");
  }catch(e){showNotice("Error: "+e.message,"error")}
  finally{busy=false;create.disabled=pikachu.disabled=false;rerun.disabled=false;await tick()}
}

galleryJob.onchange=()=>selectJob(galleryJob.value);
filesJob.onchange=()=>selectJob(filesJob.value);
create.onclick=()=>newJob(false);
pikachu.onclick=()=>newJob(true);
rerun.onclick=runSelected;
tick();setInterval(tick,3000);
</script>
</body></html>"""
    )

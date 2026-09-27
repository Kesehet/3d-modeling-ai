from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from .config import JOBS_ROOT
from .feature_tasks import feature_plan_summary
from .history import append_history

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PUBLIC_ARTIFACT_CATEGORIES = {"references", "renders", "scene", "exports"}

# A single AI/Blender stage should never sit untouched this long. Long-running
# requests refresh status as they move between stages; anything older is an
# abandoned persisted state rather than a trustworthy indication of live work.
RUNNING_JOB_STALE_AFTER = timedelta(hours=2)


def _status_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _persist_status(root: Path, status: dict) -> None:
    tmp = root / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(root / "status.json")


def reconcile_running_status(
    root: Path,
    status: dict,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> dict:
    """Convert an abandoned persisted running state into a terminal failure."""
    if status.get("state") != "running":
        return status

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    updated_at = _status_timestamp(status.get("updated_at"))
    age_seconds = (
        max(0.0, (current_time - updated_at).total_seconds())
        if updated_at is not None
        else None
    )

    if (
        not force
        and age_seconds is not None
        and age_seconds <= RUNNING_JOB_STALE_AFTER.total_seconds()
    ):
        return status

    previous_stage = str(status.get("stage") or "unknown")
    if force:
        reason = (
            "The API process restarted while this job was marked running, so the old "
            "operation is no longer active."
        )
    elif updated_at is None:
        reason = "The job was marked running without a valid activity timestamp."
    else:
        reason = (
            f"No job status update was recorded for {int(age_seconds or 0)} seconds; "
            "the operation is considered abandoned."
        )

    recovered = dict(status)
    recovered.update(
        {
            "state": "failed",
            # Keep the stage that was actually interrupted. Improve/retry logic can
            # still see where the previous attempt stopped.
            "stage": previous_stage,
            "error": reason,
            "interrupted": True,
            "interrupted_at": current_time.isoformat(),
            "interrupted_stage": previous_stage,
            "updated_at": current_time.isoformat(),
        }
    )
    _persist_status(root, recovered)
    append_history(
        root,
        "stale_running_recovered",
        stage=previous_stage,
        reason=reason,
        age_seconds=age_seconds,
        forced=force,
    )
    return recovered


def reconcile_all_running_jobs(*, force: bool = False) -> int:
    """Reconcile persisted running states, returning how many were repaired."""
    if not JOBS_ROOT.exists():
        return 0

    repaired = 0
    now = datetime.now(UTC)
    for root in JOBS_ROOT.iterdir():
        if not root.is_dir():
            continue
        status_path = root / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status, dict) or status.get("state") != "running":
            continue
        reconciled = reconcile_running_status(root, status, now=now, force=force)
        if reconciled.get("state") != "running":
            repaired += 1
    return repaired



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


def _reference_rows(root: Path) -> list[dict]:
    rows = [
        item
        for item in _artifact_rows(root, "references")
        if Path(item["name"]).suffix.lower() in IMAGE_SUFFIXES
    ]
    index_path = root / "references.json"
    metadata_by_name: dict[str, dict] = {}
    if index_path.exists():
        try:
            parsed = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                metadata_by_name = {
                    str(item.get("stored_name")): item
                    for item in parsed
                    if isinstance(item, dict) and item.get("stored_name")
                }
        except (OSError, json.JSONDecodeError):
            metadata_by_name = {}

    visible_rows: list[dict] = []
    for row in rows:
        metadata = metadata_by_name.get(row["name"])
        if not isinstance(metadata, dict):
            # Candidate downloads are never shown until research has explicitly
            # admitted them into references.json.
            continue
        provider = str(metadata.get("provider") or "").lower()
        automatic = provider in {"wikipedia", "wikimedia_commons", "wikimedia"}
        if automatic:
            try:
                match_score = float(metadata.get("match_score") or 0.0)
            except (TypeError, ValueError):
                match_score = 0.0
            if not (
                metadata.get("verified") is True
                and metadata.get("exact_identity_match") is True
                and metadata.get("useful_for_geometry") is True
                and match_score >= 0.70
            ):
                continue
        row.update(
            {
                "original_name": metadata.get("original_name"),
                "title": metadata.get("title"),
                "provider": metadata.get("provider"),
                "source_url": metadata.get("source_url"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "uploaded_at": metadata.get("uploaded_at") or metadata.get("researched_at"),
                "verified": metadata.get("verified"),
                "match_score": metadata.get("match_score"),
                "verification_reason": metadata.get("verification_reason"),
            }
        )
        visible_rows.append(row)
    return visible_rows


def _latest_vision_images(root: Path) -> list[str]:
    path = root / "vision-latest.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    images = payload.get("images") if isinstance(payload, dict) else None
    return [str(item) for item in images] if isinstance(images, list) else []


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
                if isinstance(status, dict):
                    status = reconcile_running_status(root, status)
            except (OSError, json.JSONDecodeError):
                continue

            if request.get("prompt") == "Deployment MCP smoke test cube":
                continue

            artifacts = {category: _artifact_rows(root, category) for category in PUBLIC_ARTIFACT_CATEGORIES}
            references = _reference_rows(root)
            latest_vision_images = _latest_vision_images(root)
            all_renders = [
                item
                for item in artifacts["renders"]
                if Path(item["name"]).suffix.lower() in IMAGE_SUFFIXES
            ]
            active_generic = status.get("generic_model") if isinstance(status.get("generic_model"), dict) else {}
            active_version = active_generic.get("version")
            if isinstance(active_version, int):
                prefix = f"model-v{active_version}-"
                accepted_renders = [item for item in all_renders if item["name"].startswith(prefix)]
                renders = accepted_renders or all_renders[:64]
            else:
                renders = all_renders[:64]

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
            if isinstance(active_version, int):
                active_qa = root / "exports" / f"model-v{active_version}-qa.json"
                if active_qa.is_file():
                    qa_files = [active_qa] + [path for path in qa_files if path != active_qa]
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
                    "modeling_strategy": status.get("modeling_strategy"),
                    "quality_gate": status.get("quality_gate") if isinstance(status.get("quality_gate"), dict) else None,
                    "feature_plan": feature_plan_summary(root),
                    "updated_at": status.get("updated_at"),
                    "renders": renders,
                    "references": references,
                    "latest_vision_images": latest_vision_images,
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
.quality-banner{display:none;margin:0 0 18px;padding:12px 14px;border-radius:10px;border:1px solid #fed7aa;background:#fff7ed;color:#9a3412}.quality-banner.show{display:block}.quality-banner strong{display:block;margin-bottom:3px}
.feature-panel{display:none;margin:0 0 18px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}.feature-panel.show{display:block}.feature-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:12px}.feature-head h3{margin:0 0 3px}.feature-progress{color:var(--muted);font-size:12px}.feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.feature-card{border:1px solid var(--line);border-radius:9px;padding:10px;background:#fafafa}.feature-card.running{border-color:#fdba74;background:#fff7ed}.feature-card.accepted{border-color:#bbf7d0;background:#f0fdf4}.feature-card.retry{border-color:#fde68a;background:#fffbeb}.feature-card.blocked,.feature-card.failed{border-color:#fecaca;background:#fff5f5}.feature-name{font-weight:750}.feature-meta{font-size:11px;color:var(--muted);margin-top:3px}.feature-state{font-size:10px;text-transform:uppercase;letter-spacing:.04em;font-weight:800}.feature-criteria{margin-top:6px;font-size:11px;color:#475467}
.render-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.render{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}.render img{width:100%;aspect-ratio:1;object-fit:cover;display:block}.render-name{padding:9px 11px;color:var(--muted);font-size:12px}
.reference-panel{margin-top:22px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}.reference-panel h3{margin:0 0 4px}.reference-help{color:var(--muted);font-size:12px;margin:0 0 12px}.reference-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.reference-card{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafafa}.reference-card img{display:block;width:100%;aspect-ratio:1;object-fit:cover}.reference-info{padding:8px 9px;font-size:11px;color:var(--muted)}.reference-info strong{display:block;color:var(--text);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.reference-used{display:inline-block;margin-top:5px;padding:2px 6px;border-radius:999px;background:#dcfce7;color:#166534;font-weight:700;font-size:10px}
.downloads{margin-top:22px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}.downloads h3{margin:0 0 10px}.file-list{display:flex;gap:8px;flex-wrap:wrap}.file{display:inline-flex;text-decoration:none;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#fafafa}.file b{margin-right:6px}
.details{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}.details summary{cursor:pointer;font-weight:700}.details pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;padding:12px;border-radius:8px;max-height:320px;overflow:auto}
.dialog-backdrop{display:none;position:fixed;inset:0;background:#11182788;z-index:50;padding:20px;align-items:center;justify-content:center}.dialog-backdrop.show{display:flex}.dialog{width:min(620px,100%);background:#fff;border-radius:14px;padding:20px;box-shadow:0 25px 80px #0003}.dialog h2{margin:0 0 14px}.field{margin-bottom:12px}.field label{display:block;font-weight:650;margin-bottom:5px}.field textarea,.field input,.field select{width:100%;border:1px solid #cfd4dc;border-radius:8px;padding:10px}.field textarea{min-height:110px;resize:vertical}.actions{display:flex;justify-content:flex-end;gap:8px}.notice{margin-top:10px;color:var(--muted)}.empty-state{padding:50px;text-align:center;color:var(--muted)}
@media(max-width:980px){.gallery{grid-template-columns:repeat(3,1fr)}.render-grid{grid-template-columns:repeat(2,1fr)}.reference-grid{grid-template-columns:repeat(3,1fr)}.feature-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.wrap{padding:14px}.gallery{grid-template-columns:repeat(2,1fr);gap:10px}.render-grid{grid-template-columns:1fr}.reference-grid{grid-template-columns:repeat(2,1fr)}.feature-grid{grid-template-columns:1fr}.brand h1{font-size:20px}.job-card{border-radius:10px}}
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
  <div class="quality-banner" id="qualityBanner"></div>
  <section class="feature-panel" id="featurePanel"><div class="feature-head"><div><h3>AI feature sub-jobs</h3><div class="feature-progress" id="featureProgress"></div></div></div><div class="feature-grid" id="featureGrid"></div></section>
  <div class="render-grid" id="renderGrid"></div>
  <section class="reference-panel"><h3>Reference images used</h3><p class="reference-help">These are the saved reference images attached to this job. Images included in the latest vision pass are marked below.</p><div class="reference-grid" id="referenceGrid"></div></section>
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
  title:byId("detailTitle"),meta:byId("detailMeta"),quality:byId("qualityBanner"),featurePanel:byId("featurePanel"),featureProgress:byId("featureProgress"),featureGrid:byId("featureGrid"),renders:byId("renderGrid"),references:byId("referenceGrid"),files:byId("fileList"),json:byId("detailJson"),
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
  const quality=job.quality_gate||null;
  const needsMesh=job.stage==="generic_needs_strategy_switch";
  const improvingMesh=job.stage==="adaptive_mesh_needs_refinement";
  const retryingQuality=job.stage==="generic_quality_unverified";
  if(quality?.recognizable===false){
    if(needsMesh){
      els.quality.innerHTML='<strong>Primitive model is not recognizable enough.</strong>'+esc(quality.summary||"The primitive blockout does not sufficiently match the requested subject.")+' The next improvement will switch to the adaptive mesh builder.';
    }else if(improvingMesh){
      els.quality.innerHTML='<strong>Mesh fallback is closer, but still needs work.</strong>'+esc(quality.summary||"The adaptive mesh has not passed recognizability yet.")+' Improve will generate another reference-driven mesh pass.';
    }else{
      const strategy=quality.recommended_strategy?(" Recommended next strategy: "+quality.recommended_strategy+"."):"";
      els.quality.innerHTML='<strong>Quality gate: current model is not recognizable enough.</strong>'+esc(quality.summary||"The generated geometry does not sufficiently match the requested subject.")+esc(strategy);
    }
    els.quality.classList.add("show");
  }else if(job.stage==="generic_quality_unverified"){
    els.quality.innerHTML='<strong>Visual QA needs another pass.</strong>Improve will ask the AI modeling director to judge the current renders again and choose whether to revise the procedural model or rebuild it as a mesh.';
    els.quality.classList.add("show");
  }
  const plan=job.feature_plan||null;
  if(plan&&Array.isArray(plan.features)&&plan.features.length){
    const counts=plan.counts||{};
    const accepted=counts.accepted||0;
    const terminal=accepted+(counts.blocked||0)+(counts.failed||0);
    els.featureProgress.textContent=accepted+" accepted · "+terminal+"/"+plan.features.length+" resolved"+(plan.next_feature_id?" · next: "+plan.next_feature_id:"");
    els.featureGrid.innerHTML=plan.features.map(feature=>{
      const deps=(feature.depends_on||[]).length?" · after "+feature.depends_on.join(", "):"";
      const criteria=(feature.acceptance_criteria||[]).slice(0,2).join(" · ");
      return '<div class="feature-card '+esc(feature.status)+'"><div style="display:flex;justify-content:space-between;gap:8px"><div class="feature-name">'+esc(feature.name)+'</div><div class="feature-state">'+esc(feature.status)+'</div></div><div class="feature-meta">P'+esc(feature.priority)+' · '+esc(feature.strategy)+deps+' · attempts '+esc(feature.attempts||0)+'</div>'+(criteria?'<div class="feature-criteria">'+esc(criteria)+'</div>':'')+'</div>';
    }).join("");
    els.featurePanel.classList.add("show");
  }else{
    els.featurePanel.classList.remove("show");
    els.featureGrid.innerHTML="";
    els.featureProgress.textContent="";
  }
  els.renders.innerHTML=(job.renders||[]).length
    ? job.renders.map(image=>'<div class="render"><a target="_blank" href="'+renderUrl(job.job_id,image.name,image.mtime)+'"><img loading="lazy" src="'+renderUrl(job.job_id,image.name,image.mtime)+'"></a><div class="render-name">'+esc(image.name)+'</div></div>').join("")
    : '<div class="empty-state" style="grid-column:1/-1">No renders yet. This page refreshes automatically while the job runs.</div>';
  const latestVision=new Set(job.latest_vision_images||[]);
  els.references.innerHTML=(job.references||[]).length
    ? job.references.map(ref=>{
        const label=ref.original_name||ref.title||ref.name;
        const meta=[ref.provider,ref.width&&ref.height?(ref.width+"×"+ref.height):null].filter(Boolean).join(" · ");
        const used=latestVision.has("references/"+ref.name);
        return '<div class="reference-card"><a target="_blank" href="'+fileUrl(job.job_id,"references",ref.name)+'"><img loading="lazy" src="'+fileUrl(job.job_id,"references",ref.name)+'"></a><div class="reference-info"><strong title="'+esc(label)+'">'+esc(label)+'</strong>'+(meta?'<div>'+esc(meta)+'</div>':'')+(used?'<span class="reference-used">Used in latest vision pass</span>':'')+'</div></div>';
      }).join("")
    : '<div class="empty-state" style="grid-column:1/-1">No reference images are saved for this job.</div>';
  const scene=(job.artifacts?.scene||[]).filter(file=>/\.(blend)$/i.test(file.name));
  const exports=(job.artifacts?.exports||[]).filter(file=>/\.(glb|obj|stl|3mf|json)$/i.test(file.name));
  const files=[...scene.map(file=>["scene",file]),...exports.map(file=>["exports",file])];
  els.files.innerHTML=files.length
    ? files.map(([category,file])=>'<a class="file" download href="'+fileUrl(job.job_id,category,file.name)+'"><b>↓</b>'+esc(file.name)+'</a>').join("")
    : '<span style="color:var(--muted)">No downloadable model files yet.</span>';
  els.json.textContent=JSON.stringify({status:{state:job.state,stage:job.stage,modeling_strategy:job.modeling_strategy,quality_gate:job.quality_gate},feature_plan:job.feature_plan,references:job.references,latest_vision_images:job.latest_vision_images,qa:job.qa,history:job.history},null,2);
  els.improve.disabled=busy||!scene.length;
  if(!busy){
    els.improve.textContent=retryingQuality?"Ask AI director again":(needsMesh?"AI rebuild as mesh":(improvingMesh?"AI improve mesh":"AI improve model"));
  }
  els.improve.title=retryingQuality?"Let the multimodal modeling director inspect the references and current renders and choose the next action.":(needsMesh?"Let the AI rebuild the current result using the mesh strategy.":"");
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

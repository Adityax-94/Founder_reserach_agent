"""
Flask web server for the Founder Research Agent frontend.
Endpoints:
  GET  /                  → serve index.html
  POST /api/research      → start a research job
  GET  /api/status/<id>   → poll job status
  GET  /api/reports       → list past reports
  GET  /api/report/<file> → fetch a specific report
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

@app.route("/health")
def health():
    return "ok", 200

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store: {job_id: {status, progress, result, error}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_research(job_id: str, target_name: str, context: str, iterations: int):
    """Run the agent in a background thread."""
    try:
        _update_job(job_id, status="running", progress="Planning research strategy...")

        from agent.graph import compile_graph
        graph = compile_graph()

        # Monkey-patch progress updates by wrapping nodes
        initial_state = {
            "target_name":    target_name,
            "target_context": context,
            "max_iterations": iterations,
        }

        _update_job(job_id, progress="Searching the web...")
        final_state = graph.invoke(initial_state)

        _update_job(
            job_id,
            status="done",
            progress="Complete!",
            result={
                "report_markdown": final_state.get("report_markdown", ""),
                "profile":         final_state.get("profile", {}),
                "facts_count":     len(final_state.get("extracted_facts", [])),
                "pages_count":     len(final_state.get("scraped_pages", [])),
                "report_path":     final_state.get("report_path", ""),
            }
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()  
        _update_job(job_id, status="error", error=str(exc))


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/research", methods=["POST"])
def start_research():
    data = request.get_json(force=True)
    target_name = data.get("name", "").strip()
    context     = data.get("context", "").strip()
    iterations  = int(data.get("iterations", 2))

    if not target_name:
        return jsonify({"error": "name is required"}), 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "id":         job_id,
            "name":       target_name,
            "status":     "queued",
            "progress":   "Queued...",
            "created_at": datetime.utcnow().isoformat(),
            "result":     None,
            "error":      None,
        }

    thread = threading.Thread(
        target=_run_research,
        args=(job_id, target_name, context, iterations),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/reports")
def list_reports():
    reports = []
    for f in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            profile = data.get("profile", {})
            reports.append({
                "filename": f.name,
                "name":     profile.get("name", f.stem),
                "summary":  profile.get("summary", "")[:120],
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        except Exception:
            pass
    return jsonify(reports)


@app.route("/api/report/<filename>")
def get_report(filename: str):
    # Serve markdown version if exists
    md_file = OUTPUT_DIR / filename.replace(".json", ".md")
    if md_file.exists():
        return md_file.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain"}
    json_file = OUTPUT_DIR / filename
    if json_file.exists():
        return send_from_directory(str(OUTPUT_DIR), filename)
    return jsonify({"error": "report not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RENDER") is None
    print(f"\n🚀  Founder Research Agent UI")
    print(f"   Open http://localhost:{port} in your browser\n")
    app.run(debug=debug, host="0.0.0.0", port=port, threaded=True)

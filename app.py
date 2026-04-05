from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from uuid import uuid4
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from people_counter_core.config import OUTPUT_DIR
from people_counter_core.pipeline import run_pipeline
from people_counter_core.utils import allowed_file

app = Flask(__name__)
app.secret_key = "people-counter-local-demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "people_counter_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TASKS = {}


def _update_task(task_id: str, progress: int, message: str):
    if task_id in TASKS:
        TASKS[task_id]["progress"] = max(0, min(100, int(progress)))
        TASKS[task_id]["message"] = message


def _open_folder(path: Path) -> None:
    path = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _run_batch(task_id: str, targets: list[tuple[Path, str]]):
    results = []
    total = len(targets)
    try:
        TASKS[task_id]["status"] = "running"
        TASKS[task_id]["progress"] = 1
        TASKS[task_id]["message"] = f"Starting batch of {total} file(s)..."
        for i, (target, original_name) in enumerate(targets, start=1):
            def cb(p, m, idx=i):
                seg_start = int(((idx - 1) / total) * 100)
                seg_end = int((idx / total) * 100)
                scaled = seg_start + int((seg_end - seg_start) * (p / 100))
                _update_task(task_id, scaled, f"[{idx}/{total}] {original_name} - {m}")

            result = run_pipeline(video_path=target, show_live_window=False, progress_callback=cb, output_name=original_name)
            result["display_name"] = original_name
            results.append(result)
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass
        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["message"] = "Batch finished."
        TASKS[task_id]["results"] = results
    except Exception as exc:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["message"] = str(exc)
        for target, _ in targets:
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_videos():
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"ok": False, "error": "No files selected."}), 400
    targets = []
    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            return jsonify({"ok": False, "error": f"Unsupported file type: {file.filename}"}), 400
        filename = secure_filename(file.filename)
        target = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
        file.save(target)
        targets.append((target, filename))
    if not targets:
        return jsonify({"ok": False, "error": "No valid files uploaded."}), 400
    task_id = uuid4().hex
    TASKS[task_id] = {"status": "queued", "progress": 0, "message": "Queued...", "results": [], "count": len(targets)}
    threading.Thread(target=_run_batch, args=(task_id, targets), daemon=True).start()
    return jsonify({"ok": True, "task_id": task_id})


@app.route("/status/<task_id>", methods=["GET"])
def task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({"ok": False, "error": "Task not found."}), 404
    return jsonify({"ok": True, "status": task["status"], "progress": task["progress"], "message": task["message"], "results": task.get("results", []), "count": task.get("count", 0)})


@app.route("/outputs/<path:filename>")
def outputs_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/open-output-folder", methods=["POST"])
def open_output_folder():
    try:
        _open_folder(OUTPUT_DIR)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False)

"""Flask app for blind, image-based annotation of a stratified frame sample.

Differences from the original review app (``legacy/post_processing/flask_script.py``):

* annotators see the **frame image** (decoded from the video), not just the
  Florence-2 string, and never see either system's label;
* every answer is stored with annotator id, timestamp and time spent, one row
  per (item, annotator), so inter-annotator agreement can be computed;
* items are shown in a per-annotator random order (seeded by the annotator id)
  to avoid order effects;
* ``unsure`` / ``unviewable`` are explicit answers rather than forced choices;
* binds to ``127.0.0.1`` and never runs in debug mode.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import threading
from pathlib import Path

import pandas as pd

from ..annotation import NON_LABELS
from ..taxonomy import LABELS

ANSWER_FIELDS = ["sample_id", "annotator", "label", "comment", "seconds", "answered_at"]
PRETTY = {
    "weapons": "Weapons",
    "aircraft": "Military aircraft",
    "maritime": "Military maritime",
    "ground_vehicles": "Military ground vehicles",
    "nonmilitary": "Non-military",
    "unsure": "Unsure",
    "unviewable": "Frame missing / unviewable",
}


def _index_videos(video_dir: Path | None) -> dict[str, Path]:
    if video_dir is None:
        return {}
    return {p.name: p for p in video_dir.rglob("*") if p.is_file()}


def create_app(sheet: str | Path, answers: str | Path, video_dir: str | Path | None = None, show_text: bool = False):
    from flask import Flask, abort, redirect, render_template, request, send_file, url_for

    from ..video import read_frame

    sheet_df = pd.read_csv(sheet, dtype={"sample_id": str, "video": str})
    answers = Path(answers)
    videos = _index_videos(Path(video_dir) if video_dir else None)
    lock = threading.Lock()
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["SECRET_KEY"] = hashlib.sha256(str(answers.resolve()).encode()).hexdigest()

    def answered(annotator: str) -> set[str]:
        if not answers.exists():
            return set()
        with answers.open(newline="") as fh:
            return {r["sample_id"] for r in csv.DictReader(fh) if r["annotator"] == annotator}

    def order_for(annotator: str) -> list[str]:
        seed = int(hashlib.sha256(annotator.encode()).hexdigest()[:8], 16)
        return sheet_df["sample_id"].sample(frac=1.0, random_state=seed).tolist()

    @app.get("/")
    def index():
        return render_template("index.html", n_items=len(sheet_df))

    @app.post("/start")
    def start():
        annotator = request.form.get("annotator", "").strip()
        if not annotator or not annotator.replace("-", "").replace("_", "").isalnum():
            return render_template("index.html", n_items=len(sheet_df), error="Use letters, digits, - or _ only.")
        return redirect(url_for("annotate", annotator=annotator))

    @app.get("/annotate/<annotator>")
    def annotate(annotator: str):
        done = answered(annotator)
        remaining = [sid for sid in order_for(annotator) if sid not in done]
        if not remaining:
            return render_template("done.html", annotator=annotator, n_items=len(sheet_df))
        row = sheet_df.set_index("sample_id").loc[remaining[0]]
        return render_template(
            "annotate.html",
            annotator=annotator,
            sample_id=remaining[0],
            text=row.get("florence_text") if show_text else None,
            has_image=row["video"] in videos,
            labels=[(label, PRETTY[label]) for label in LABELS],
            extras=[(label, PRETTY[label]) for label in NON_LABELS],
            progress=len(done),
            n_items=len(sheet_df),
            started=dt.datetime.now(dt.timezone.utc).timestamp(),
        )

    @app.post("/annotate/<annotator>")
    def submit(annotator: str):
        sid = request.form["sample_id"]
        label = request.form.get("label", "")
        if sid not in set(sheet_df["sample_id"]) or label not in (*LABELS, *NON_LABELS):
            abort(400)
        seconds = dt.datetime.now(dt.timezone.utc).timestamp() - float(request.form.get("started", "0") or 0)
        with lock:
            new = not answers.exists()
            answers.parent.mkdir(parents=True, exist_ok=True)
            with answers.open("a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ANSWER_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": sid,
                        "annotator": annotator,
                        "label": label,
                        "comment": request.form.get("comment", "")[:500],
                        "seconds": round(seconds, 1),
                        "answered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    }
                )
        return redirect(url_for("annotate", annotator=annotator))

    @app.get("/frame/<sample_id>.jpg")
    def frame(sample_id: str):
        rows = sheet_df[sheet_df["sample_id"] == sample_id]
        if rows.empty or rows.iloc[0]["video"] not in videos:
            abort(404)
        rgb = read_frame(videos[rows.iloc[0]["video"]], int(rows.iloc[0]["frame_number"]))
        if rgb is None:
            abort(404)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    return app


def serve(sheet, answers, video_dir=None, show_text=False, host="127.0.0.1", port=5000) -> None:
    create_app(sheet, answers, video_dir, show_text).run(host=host, port=port, debug=False)

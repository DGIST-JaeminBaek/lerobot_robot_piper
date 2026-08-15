#!/usr/bin/env python3
"""SSH로 접속해 브라우저로 QC 자동 계산 결과를 사람이 최종 판정하는 웹 서버.

`qc_recompute_all.py` + `qc_extract_review_frames.py`가 만든 315개 녹화의
시작 직후 20프레임 / 종료 직전 20프레임(top+wrist)을 한 화면에 보여주고
OK/SUSPECT/BAD 판정을 받는다. `synthetic/calibration/select_board_points_web.py`와
같은 방식(순수 http.server, 127.0.0.1 고정, 토큰 인증)을 쓰지만, 그 도구는
"1회 저장 후 서버 종료"였던 반면 이건 315개를 계속 순회해야 하므로 서버를
`serve_forever()`로 계속 띄워둔다.

실행:
    python scripts/tools/qc_review_web.py
    # 출력된 URL을 VS Code PORTS 패널(또는 ssh -L)로 포워딩해 브라우저로 접속
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QC_JSON = REPO_ROOT / "configs/qc_review_all_315.json"
DEFAULT_FRAMES_DIR = REPO_ROOT / "outputs/analysis/qc_review_315/frames"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/qc_review_315/verdicts.csv"

# batch_replay_review.py와 동일한 3단계 체계 — 재사용 목적으로 값도 맞췄다.
VERDICTS = {"1": "OK", "2": "SUSPECT", "3": "BAD"}
CSV_FIELDS = ["source_dataset", "session", "shape", "qc_level", "verdict", "note", "reviewed_at"]

MAX_REQUEST_BYTES = 8 * 1024


# --------------------------------------------------------------------------
# 판정 저장 (batch_replay_review.py::load_reviews/save_reviews 패턴,
# 키만 recording명 대신 source_dataset 전체 경로로 바꿔 세션 간 동명 충돌을 피함)
# --------------------------------------------------------------------------


def load_verdicts(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return {row["source_dataset"]: row for row in csv.DictReader(f) if row.get("source_dataset")}


def save_verdicts(csv_path: Path, verdicts: dict[str, dict[str, str]]) -> None:
    """매 판정마다 전체를 다시 쓴다 — 315행이라 비용이 무의미하고, 중간에
    강제 종료돼도 결과가 남는 편이 훨씬 중요하다 (batch_replay_review.py와 동일 이유)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in verdicts.values():
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    tmp.replace(csv_path)


# --------------------------------------------------------------------------
# 상태
# --------------------------------------------------------------------------


@dataclass
class ReviewState:
    episodes: list[dict]
    frames_dir: Path
    verdict_path: Path
    verdicts: dict[str, dict[str, str]]
    token: str
    order: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def episode_dir(self, episode: dict) -> Path:
        name = Path(episode["source_dataset"]).name
        return self.frames_dir / f"{episode['session']}__{name}"

    def frame_urls(self, episode: dict) -> dict[str, dict[str, list[str]]]:
        out: dict[str, dict[str, list[str]]] = {}
        ep_dir = self.episode_dir(episode)
        for camera in episode["cameras"]:
            out[camera] = {}
            for tag in ("start", "end"):
                files = sorted(ep_dir.glob(f"{camera}_{tag}_*.jpg"))
                out[camera][tag] = [f"/frames/{f.relative_to(self.frames_dir).as_posix()}" for f in files]
        return out

    def rebuild_order(self) -> None:
        """미판정 먼저, 그다음 이미 판정된 것 — resume 시 미판정부터 보이도록."""
        unreviewed = [i for i, e in enumerate(self.episodes) if e["source_dataset"] not in self.verdicts]
        reviewed = [i for i, e in enumerate(self.episodes) if e["source_dataset"] in self.verdicts]
        self.order = unreviewed + reviewed

    def next_unreviewed(self, after_source_dataset: str) -> int:
        try:
            pos = next(i for i, idx in enumerate(self.order) if self.episodes[idx]["source_dataset"] == after_source_dataset)
        except StopIteration:
            pos = -1
        for i in range(pos + 1, len(self.order)):
            idx = self.order[i]
            if self.episodes[idx]["source_dataset"] not in self.verdicts:
                return i
        for i, idx in enumerate(self.order):
            if self.episodes[idx]["source_dataset"] not in self.verdicts:
                return i
        return min(pos + 1, len(self.order) - 1) if pos >= 0 else 0


def episode_payload(state: ReviewState, order_pos: int) -> dict:
    order_pos = max(0, min(order_pos, len(state.order) - 1))
    idx = state.order[order_pos]
    episode = state.episodes[idx]
    existing = state.verdicts.get(episode["source_dataset"])
    return {
        "order_pos": order_pos,
        "total": len(state.order),
        "reviewed": len(state.verdicts),
        "source_dataset": episode["source_dataset"],
        "session": episode["session"],
        "shape": episode["shape"],
        "qc_level": episode["qc_level"],
        "qc_notes": episode["qc_notes"],
        "total_frames": episode["total_frames"],
        "auto_start_frame": episode["auto_start_frame"],
        "auto_end_frame": episode["auto_end_frame"],
        "erased_ratio": episode["erased_ratio"],
        "cameras": episode["cameras"],
        "frames": state.frame_urls(episode),
        "gripper_end": episode.get("gripper_end", {}),
        "existing_verdict": existing["verdict"] if existing else None,
        "existing_note": existing["note"] if existing else "",
    }


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def render_html(token: str) -> bytes:
    template = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>QC 리뷰 (315)</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; background: #1a1a1e; color: #ddd; }
  header { padding: 10px 16px; background: #24242a; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header b { color: #fff; }
  .pill { padding: 2px 8px; border-radius: 10px; font-size: 12px; }
  .green { background: #1baf7a33; color: #4fe0ac; }
  .yellow { background: #e0b64033; color: #f0cd6a; }
  .red { background: #e0505033; color: #f08080; }
  main { padding: 12px 16px; }
  .row-label { font-size: 12px; color: #999; margin: 10px 0 4px; }
  .strip { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px; }
  .strip img { height: 130px; border-radius: 4px; border: 1px solid #333; flex: 0 0 auto; }
  .end-stack { display: flex; flex-direction: column; gap: 3px; }
  .end-row { display: flex; align-items: center; gap: 10px; background: #202024; border-radius: 6px; padding: 4px 10px; }
  .end-row.crossing { outline: 1px solid #e0b640; }
  .end-row.past-cut { opacity: 0.75; border-left: 3px solid #666; }
  .end-meta { width: 170px; font-size: 12px; color: #bbb; font-family: monospace; flex: 0 0 auto; white-space: pre; }
  .end-row img { height: 96px; border-radius: 4px; border: 1px solid #333; }
  footer { position: sticky; bottom: 0; background: #24242a; padding: 10px 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button { background: #333; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px 14px; cursor: pointer; font-size: 14px; }
  button:hover { background: #444; }
  button.ok { border-color: #1baf7a; } button.ok.active { background: #1baf7a; color: #08251a; }
  button.suspect { border-color: #e0b640; } button.suspect.active { background: #e0b640; color: #2a2205; }
  button.bad { border-color: #e05050; } button.bad.active { background: #e05050; color: #2a0a0a; }
  textarea { flex: 1; min-width: 200px; background: #1a1a1e; color: #ddd; border: 1px solid #444; border-radius: 6px; padding: 6px; }
  .nav { display: flex; gap: 6px; }
  .progress { color: #999; font-size: 13px; }
  .notes { font-size: 12px; color: #f0a0a0; }
</style>
</head>
<body>
<header>
  <b id="src">-</b>
  <span id="level" class="pill">-</span>
  <span>session=<span id="session">-</span></span>
  <span>shape=<span id="shape">-</span></span>
  <span>frames=<span id="range">-</span></span>
  <span>erased=<span id="erased">-</span></span>
  <span class="progress" id="progress">-</span>
  <span class="notes" id="notes"></span>
</header>
<main id="strips"></main>
<footer>
  <div class="nav">
    <button id="prev">&larr; 이전</button>
    <button id="next">다음 &rarr;</button>
  </div>
  <button class="ok" data-v="OK" onclick="setVerdict('OK')">1: OK</button>
  <button class="suspect" data-v="SUSPECT" onclick="setVerdict('SUSPECT')">2: SUSPECT</button>
  <button class="bad" data-v="BAD" onclick="setVerdict('BAD')">3: BAD</button>
  <textarea id="note" rows="1" placeholder="비고 (선택)"></textarea>
</footer>
<script>
const TOKEN = "__TOKEN__";
let pos = 0;
let current = null;

async function loadPos(p) {
  const res = await fetch(`/api/episode/${p}?token=${TOKEN}`);
  if (!res.ok) return;
  current = await res.json();
  pos = current.order_pos;
  render();
}

function row(label, urls) {
  if (!urls || urls.length === 0) return "";
  const imgs = urls.map(u => `<img src="${u}?token=${TOKEN}" loading="lazy">`).join("");
  return `<div class="row-label">${label}</div><div class="strip">${imgs}</div>`;
}

function endTable(ep) {
  const g = ep.gripper_end || {};
  const idxs = g.frame_index || [];
  if (idxs.length === 0) return "";
  const threshold = g.plateau !== undefined ? g.plateau + 1.0 : null;
  let rows = "";
  let dividerShown = false;
  for (let i = 0; i < idxs.length; i++) {
    const topUrl = ep.frames.top?.end?.[i];
    const wristUrl = ep.frames.wrist?.end?.[i];
    const opened = threshold !== null && g.action[i] > threshold;
    const pastCut = idxs[i] >= ep.auto_end_frame;
    if (pastCut && !dividerShown) {
      rows += `<div class="row-label" style="margin-top:6px">↓ QC 컷 이후 (학습 데이터엔 안 들어감, 참고용)</div>`;
      dividerShown = true;
    }
    rows += `<div class="end-row${opened ? " crossing" : ""}${pastCut ? " past-cut" : ""}">` +
      `<div class="end-meta">frame ${idxs[i]}\naction ${g.action[i]}\nstate  ${g.state[i]}</div>` +
      (topUrl ? `<img src="${topUrl}?token=${TOKEN}" loading="lazy">` : "") +
      (wristUrl ? `<img src="${wristUrl}?token=${TOKEN}" loading="lazy">` : "") +
      `</div>`;
  }
  const label = threshold !== null
    ? `종료 직전 (그리퍼 plateau=${g.plateau}, 임계값 ${threshold.toFixed(1)} 넘으면 노란 테두리)`
    : "종료 직전";
  return `<div class="row-label">${label}</div><div class="end-stack">${rows}</div>`;
}

function render() {
  document.getElementById("src").textContent = current.source_dataset;
  const lvl = document.getElementById("level");
  lvl.textContent = current.qc_level;
  lvl.className = "pill " + current.qc_level;
  document.getElementById("session").textContent = current.session;
  document.getElementById("shape").textContent = current.shape;
  document.getElementById("range").textContent =
    `${current.auto_start_frame}-${current.auto_end_frame} / ${current.total_frames}`;
  document.getElementById("erased").textContent =
    current.erased_ratio === null ? "?" : (current.erased_ratio * 100).toFixed(0) + "%";
  document.getElementById("progress").textContent =
    `${pos + 1}/${current.total} (판정완료 ${current.reviewed})`;
  document.getElementById("notes").textContent = (current.qc_notes || []).join(" / ");
  document.getElementById("note").value = current.existing_note || "";

  let html = "";
  for (const cam of current.cameras) {
    html += row(`${cam} · 시작 직후`, current.frames[cam]?.start);
  }
  html += endTable(current);
  document.getElementById("strips").innerHTML = html;

  document.querySelectorAll("footer button[data-v]").forEach(b => {
    b.classList.toggle("active", b.dataset.v === current.existing_verdict);
  });
}

async function setVerdict(verdict) {
  const note = document.getElementById("note").value;
  const res = await fetch(`/verdict?token=${TOKEN}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_dataset: current.source_dataset, verdict, note }),
  });
  const data = await res.json();
  if (data.ok) loadPos(data.next_pos);
}

document.getElementById("prev").onclick = () => loadPos(Math.max(0, pos - 1));
document.getElementById("next").onclick = () => loadPos(Math.min(current.total - 1, pos + 1));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") setVerdict("OK");
  else if (e.key === "2") setVerdict("SUSPECT");
  else if (e.key === "3") setVerdict("BAD");
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
});

loadPos(0);
</script>
</body>
</html>
"""
    return template.replace("__TOKEN__", token).encode("utf-8")


# --------------------------------------------------------------------------
# HTTP 서버
# --------------------------------------------------------------------------


def make_handler(state: ReviewState, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QCReview/1"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[HTTP] {self.address_string()} {format_string % args}")

        def _token_ok(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            provided = query.get("token", [""])[0]
            return secrets.compare_digest(provided, state.token)

        def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not self._token_ok():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid token"})
                return
            if parsed.path == "/":
                self._send_bytes(HTTPStatus.OK, html, "text/html; charset=utf-8")
            elif parsed.path.startswith("/api/episode/"):
                try:
                    pos = int(parsed.path.removeprefix("/api/episode/"))
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad index"})
                    return
                with state.lock:
                    payload = episode_payload(state, pos)
                self._send_json(HTTPStatus.OK, payload)
            elif parsed.path.startswith("/frames/"):
                rel = parsed.path.removeprefix("/frames/")
                target = (state.frames_dir / rel).resolve()
                if not str(target).startswith(str(state.frames_dir.resolve()) + "/") or not target.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send_bytes(HTTPStatus.OK, target.read_bytes(), "image/jpeg")
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/verdict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._token_ok():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid token"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request size"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                source_dataset = body["source_dataset"]
                verdict = body["verdict"]
                note = str(body.get("note", ""))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if verdict not in VERDICTS.values():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"verdict must be one of {list(VERDICTS.values())}"})
                return
            episode = next((e for e in state.episodes if e["source_dataset"] == source_dataset), None)
            if episode is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown source_dataset"})
                return

            with state.lock:
                state.verdicts[source_dataset] = {
                    "source_dataset": source_dataset,
                    "session": episode["session"],
                    "shape": episode["shape"],
                    "qc_level": episode["qc_level"],
                    "verdict": verdict,
                    "note": note,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
                save_verdicts(state.verdict_path, state.verdicts)
                next_pos = state.next_unreviewed(source_dataset)

            print(f"[OK] {source_dataset} -> {verdict}")
            self._send_json(HTTPStatus.OK, {"ok": True, "next_pos": next_pos})

    return Handler


def create_server(state: ReviewState, port: int) -> ThreadingHTTPServer:
    html = render_html(state.token)
    handler = make_handler(state, html)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qc-json", type=Path, default=DEFAULT_QC_JSON)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8766, help="0이면 자유 포트 (default: 8766)")
    parser.add_argument(
        "--only-verdicts",
        type=str,
        default="",
        help="콤마로 구분한 기존 판정만 골라서 재검토 목록에 올림 (예: SUSPECT,BAD). 비우면 전체(기존 동작).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.qc_json.read_text())
    episodes = [e for e in payload["episodes"] if not e["broken"]]
    if not episodes:
        print("[ERROR] 판정할 에피소드가 없음", file=sys.stderr)
        return 2

    existing_verdicts = load_verdicts(args.output)
    only_verdicts = {v.strip().upper() for v in args.only_verdicts.split(",") if v.strip()}
    if only_verdicts:
        episodes = [
            e for e in episodes
            if existing_verdicts.get(e["source_dataset"], {}).get("verdict") in only_verdicts
        ]
        if not episodes:
            print(f"[ERROR] {only_verdicts}에 해당하는 판정이 없음", file=sys.stderr)
            return 2

    token = secrets.token_urlsafe(24)
    state = ReviewState(
        episodes=episodes,
        frames_dir=args.frames_dir,
        verdict_path=args.output,
        verdicts=existing_verdicts,
        token=token,
    )
    state.rebuild_order()

    server = create_server(state, args.port)
    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/?token={token}"
    print("[READY] QC 리뷰 서버")
    print(f"[URL] {url}")
    print(f"[INFO] 대상 {len(episodes)}개, 기판정 {len(state.verdicts)}개")
    print(f"[INFO] 판정 저장: {args.output}")
    print("[INFO] VS Code PORTS 패널에서 이 포트를 포워딩한 뒤 위 URL을 로컬 브라우저에서 여세요.")
    print("[INFO] Ctrl+C로 종료 — 그때까지의 판정은 이미 CSV에 저장돼 있습니다.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] 서버 종료")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

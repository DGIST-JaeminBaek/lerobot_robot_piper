#!/usr/bin/env python3
"""Select board corners in a browser forwarded over SSH.

The HTTP server listens on 127.0.0.1 only. It serves one frame and accepts one
four-point selection, writes the same point-selection JSON as the OpenCV tool,
then shuts down. It never connects to ROS, CAN, or the robot.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    FORMAT_VERSION,
    POINT_NAMES,
    POINT_SELECTION_TYPE,
    BoardSpec,
    CalibrationError,
    load_source_frame,
    make_correspondences,
    validate_image_quad,
    write_json,
)


MAX_REQUEST_BYTES = 64 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve one TOP image on localhost so board corners can be selected "
            "from a Windows browser through SSH port forwarding."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image")
    source.add_argument("--video", type=Path, help="Input video")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based video frame index (default: 0)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unit",
        choices=["normalized", "mm"],
        default="normalized",
    )
    parser.add_argument("--board-width", type=float, default=None)
    parser.add_argument("--board-height", type=float, default=None)
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Localhost TCP port; use 0 to choose a free port (default: 8765)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_board(args: argparse.Namespace) -> BoardSpec:
    if args.unit == "normalized":
        if args.board_width is not None or args.board_height is not None:
            raise CalibrationError(
                "--board-width/--board-height are only used with --unit mm"
            )
        board = BoardSpec(width=1.0, height=1.0, unit="normalized")
    else:
        if args.board_width is None or args.board_height is None:
            raise CalibrationError(
                "--unit mm requires both --board-width and --board-height"
            )
        board = BoardSpec(
            width=float(args.board_width),
            height=float(args.board_height),
            unit="mm",
        )
    board.validate()
    return board


def build_selection_payload(
    *,
    source: dict[str, Any],
    board: BoardSpec,
    image_points: Any,
) -> dict[str, Any]:
    points = validate_image_quad(
        image_points,
        width=int(source["width"]),
        height=int(source["height"]),
    )
    return {
        "format_version": FORMAT_VERSION,
        "type": POINT_SELECTION_TYPE,
        "status": "unverified",
        "source": source,
        "board": {
            "unit": board.unit,
            "width": board.width,
            "height": board.height,
            "origin": "top_left",
            "x_direction": "top_left_to_top_right",
            "y_direction": "top_left_to_bottom_left",
        },
        "point_order": list(POINT_NAMES),
        "correspondences": make_correspondences(points, board),
    }


def encode_png(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise CalibrationError("could not encode source frame as PNG")
    return encoded.tobytes()


def render_html(
    *,
    width: int,
    height: int,
    token: str,
    board: BoardSpec,
) -> bytes:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic board calibration</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    main { max-width: 1500px; margin: 0 auto; padding: 16px; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    p { margin: 6px 0; color: #bbb; }
    #stage { color: #ffe66d; font-weight: 700; }
    #workspace { display: grid; grid-template-columns: minmax(0, 1fr) 260px;
                 gap: 12px; align-items: start; margin-top: 12px; }
    #canvas-wrap { border: 1px solid #555; line-height: 0; overflow: auto;
                   max-height: 78vh; background: #080808; }
    #canvas { display: block; width: 100%; height: auto; cursor: crosshair; }
    #magnifier-wrap { position: sticky; top: 12px; }
    #magnifier { width: 240px; height: 240px; border: 1px solid #777;
                 image-rendering: pixelated; cursor: none; }
    #cursor { margin-top: 6px; color: #9fd3ff; font-family: monospace; }
    #point-editor { margin-top: 12px; display: grid; gap: 6px; }
    .point-row { display: grid; grid-template-columns: 24px 1fr 72px 72px;
                 gap: 5px; align-items: center; padding: 5px;
                 border: 1px solid #444; border-radius: 4px; cursor: pointer; }
    .point-row.active { border-color: #ffe66d; background: #292613; }
    .point-row input { width: 62px; background: #171717; color: #fff;
                       border: 1px solid #666; padding: 5px; }
    .point-index { font-weight: 700; text-align: center; }
    .point-name { font-size: 13px; }
    .controls { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    button { border: 1px solid #777; background: #292929; color: #fff;
             padding: 9px 14px; border-radius: 5px; cursor: pointer; }
    button:disabled { opacity: .4; cursor: default; }
    #save { background: #176b3a; }
    #status { margin-top: 10px; white-space: pre-wrap; color: #9fd3ff; }
    code { background: #222; padding: 2px 5px; border-radius: 3px; }
    @media (max-width: 900px) {
      #workspace { grid-template-columns: 1fr; }
      #magnifier-wrap { position: static; }
    }
  </style>
</head>
<body>
<main>
  <h1>Board corner selection</h1>
  <p>Click in order:
    <code>top_left → top_right → bottom_right → bottom_left</code>
  </p>
  <p>Board: __BOARD_WIDTH__ × __BOARD_HEIGHT__ __BOARD_UNIT__ ·
     source: __IMAGE_WIDTH__ × __IMAGE_HEIGHT__ px</p>
  <p>Editing: <span id="stage">1 · top_left</span></p>
  <p>Mouse: coarse placement · <code>1~4</code>: select point ·
     arrows: 1 px · Shift+arrows: 10 px · Enter: next point</p>
  <div id="workspace">
    <div id="canvas-wrap"><canvas id="canvas"
         width="__IMAGE_WIDTH__" height="__IMAGE_HEIGHT__"></canvas></div>
    <aside id="magnifier-wrap">
      <canvas id="magnifier" width="240" height="240"></canvas>
      <div id="cursor">cursor: -</div>
      <div id="point-editor">
        <div class="point-row active" data-index="0">
          <span class="point-index">1</span><span class="point-name">top_left</span>
          <input class="point-x" aria-label="top_left x" placeholder="x">
          <input class="point-y" aria-label="top_left y" placeholder="y">
        </div>
        <div class="point-row" data-index="1">
          <span class="point-index">2</span><span class="point-name">top_right</span>
          <input class="point-x" aria-label="top_right x" placeholder="x">
          <input class="point-y" aria-label="top_right y" placeholder="y">
        </div>
        <div class="point-row" data-index="2">
          <span class="point-index">3</span><span class="point-name">bottom_right</span>
          <input class="point-x" aria-label="bottom_right x" placeholder="x">
          <input class="point-y" aria-label="bottom_right y" placeholder="y">
        </div>
        <div class="point-row" data-index="3">
          <span class="point-index">4</span><span class="point-name">bottom_left</span>
          <input class="point-x" aria-label="bottom_left x" placeholder="x">
          <input class="point-y" aria-label="bottom_left y" placeholder="y">
        </div>
      </div>
    </aside>
  </div>
  <div class="controls">
    <button id="previous">Previous point</button>
    <button id="next">Next point</button>
    <button id="clear-active" disabled>Clear active</button>
    <button id="reset" disabled>Reset</button>
    <button class="zoom" data-zoom="fit">Fit</button>
    <button class="zoom" data-zoom="100">100%</button>
    <button class="zoom" data-zoom="200">200%</button>
    <button id="save" disabled>Save board_points.json</button>
  </div>
  <div id="status">Waiting for four points.</div>
</main>
<script>
"use strict";
const names = ["top_left", "top_right", "bottom_right", "bottom_left"];
const colors = ["#ffe66d", "#67e480", "#67d8e4", "#ff9f43"];
const token = "__TOKEN__";
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const magnifier = document.getElementById("magnifier");
const magnifierContext = magnifier.getContext("2d");
const cursorBox = document.getElementById("cursor");
const stage = document.getElementById("stage");
const statusBox = document.getElementById("status");
const previousButton = document.getElementById("previous");
const nextButton = document.getElementById("next");
const clearActiveButton = document.getElementById("clear-active");
const resetButton = document.getElementById("reset");
const saveButton = document.getElementById("save");
const pointRows = Array.from(document.querySelectorAll(".point-row"));
const points = [null, null, null, null];
const image = new Image();
let ready = false;
let cursorPoint = null;
let activeIndex = 0;

image.onload = () => {
  ready = true;
  selectActive(0);
  setZoom("fit");
};
image.onerror = () => {
  statusBox.textContent = "Could not load the source frame.";
};
image.src = `/frame.png?token=${encodeURIComponent(token)}`;

function pointCount() {
  return points.filter(point => point !== null).length;
}

function clampPoint(point) {
  return {
    x: Math.max(0, Math.min(canvas.width - 1, point.x)),
    y: Math.max(0, Math.min(canvas.height - 1, point.y))
  };
}

function syncPointEditor() {
  pointRows.forEach((row, index) => {
    row.classList.toggle("active", index === activeIndex);
    const xInput = row.querySelector(".point-x");
    const yInput = row.querySelector(".point-y");
    const point = points[index];
    xInput.value = point === null ? "" : point.x.toFixed(2);
    yInput.value = point === null ? "" : point.y.toFixed(2);
  });
}

function redraw() {
  if (!ready) return;
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  context.lineWidth = 3;
  context.strokeStyle = "#ffffff";
  for (let index = 0; index < 4; index++) {
    const nextIndex = (index + 1) % 4;
    if (points[index] !== null && points[nextIndex] !== null) {
      context.beginPath();
      context.moveTo(points[index].x, points[index].y);
      context.lineTo(points[nextIndex].x, points[nextIndex].y);
      context.stroke();
    }
  }
  points.forEach((point, index) => {
    if (point === null) return;
    context.fillStyle = colors[index];
    context.beginPath();
    context.arc(point.x, point.y, 8, 0, Math.PI * 2);
    context.fill();
    if (index === activeIndex) {
      context.strokeStyle = "#ff00ff";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(point.x, point.y, 14, 0, Math.PI * 2);
      context.stroke();
    }
    context.font = "bold 20px system-ui";
    context.fillText(names[index], point.x + 12, Math.max(point.y - 12, 24));
  });
  stage.textContent = `${activeIndex + 1} · ${names[activeIndex]}`;
  clearActiveButton.disabled = points[activeIndex] === null;
  resetButton.disabled = pointCount() === 0;
  saveButton.disabled = pointCount() !== 4;
  syncPointEditor();
  if (pointCount() < 4) {
    statusBox.textContent = `${pointCount()}/4 set. Editing ${names[activeIndex]}.`;
  } else {
    statusBox.textContent = "4/4 set. Select any point with keys 1~4, refine it, then Save.";
  }
  redrawMagnifier(
    points[activeIndex] !== null ? points[activeIndex] : cursorPoint
  );
}

function selectActive(index) {
  activeIndex = (index + 4) % 4;
  redraw();
}

function setZoom(value) {
  if (value === "fit") {
    canvas.style.width = "100%";
  } else {
    const scale = Number(value) / 100;
    canvas.style.width = `${Math.round(canvas.width * scale)}px`;
  }
  canvas.style.height = "auto";
}

function eventToImagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  const rawX = (event.clientX - rect.left) * canvas.width / rect.width;
  const rawY = (event.clientY - rect.top) * canvas.height / rect.height;
  return {
    x: Math.max(0, Math.min(canvas.width - 1, rawX)),
    y: Math.max(0, Math.min(canvas.height - 1, rawY))
  };
}

function redrawMagnifier(point) {
  if (!ready) return;
  if (point === null) {
    magnifierContext.clearRect(0, 0, magnifier.width, magnifier.height);
    cursorBox.textContent = "active point: not set";
    return;
  }
  const sourceSize = 40;
  const sourceX = Math.max(
    0, Math.min(canvas.width - sourceSize, point.x - sourceSize / 2)
  );
  const sourceY = Math.max(
    0, Math.min(canvas.height - sourceSize, point.y - sourceSize / 2)
  );
  magnifierContext.imageSmoothingEnabled = false;
  magnifierContext.clearRect(0, 0, magnifier.width, magnifier.height);
  magnifierContext.drawImage(
    image,
    sourceX, sourceY, sourceSize, sourceSize,
    0, 0, magnifier.width, magnifier.height
  );
  const centerX = (point.x - sourceX) * magnifier.width / sourceSize;
  const centerY = (point.y - sourceY) * magnifier.height / sourceSize;
  magnifierContext.strokeStyle = "#ff00ff";
  magnifierContext.lineWidth = 1;
  magnifierContext.beginPath();
  magnifierContext.moveTo(centerX, 0);
  magnifierContext.lineTo(centerX, magnifier.height);
  magnifierContext.moveTo(0, centerY);
  magnifierContext.lineTo(magnifier.width, centerY);
  magnifierContext.stroke();
  cursorBox.textContent = `cursor: (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
}

canvas.addEventListener("mousemove", event => {
  cursorPoint = eventToImagePoint(event);
  if (points[activeIndex] === null) redrawMagnifier(cursorPoint);
});

canvas.addEventListener("mouseleave", () => {
  cursorPoint = null;
  redrawMagnifier(points[activeIndex]);
  if (points[activeIndex] === null) cursorBox.textContent = "cursor: -";
});

canvas.addEventListener("click", event => {
  if (!ready) return;
  points[activeIndex] = eventToImagePoint(event);
  redraw();
});

window.addEventListener("keydown", event => {
  if (event.target instanceof HTMLInputElement) return;
  if (/^[1-4]$/.test(event.key)) {
    selectActive(Number(event.key) - 1);
    event.preventDefault();
    return;
  }
  if (event.key === "Enter") {
    selectActive(activeIndex + 1);
    event.preventDefault();
    return;
  }
  if (!event.key.startsWith("Arrow") || points[activeIndex] === null) return;
  const point = points[activeIndex];
  const amount = event.shiftKey ? 10 : 1;
  if (event.key === "ArrowLeft") point.x -= amount;
  if (event.key === "ArrowRight") point.x += amount;
  if (event.key === "ArrowUp") point.y -= amount;
  if (event.key === "ArrowDown") point.y += amount;
  points[activeIndex] = clampPoint(point);
  event.preventDefault();
  redraw();
});

pointRows.forEach((row, index) => {
  row.addEventListener("click", () => selectActive(index));
  const inputs = [
    row.querySelector(".point-x"),
    row.querySelector(".point-y")
  ];
  inputs.forEach(input => {
    input.addEventListener("click", event => {
      selectActive(index);
      event.stopPropagation();
    });
    input.addEventListener("change", () => {
      const x = Number(inputs[0].value);
      const y = Number(inputs[1].value);
      if (inputs[0].value === "" && inputs[1].value === "") {
        points[index] = null;
        selectActive(index);
        return;
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        statusBox.textContent = `${names[index]} requires finite numeric x and y.`;
        return;
      }
      points[index] = clampPoint({x, y});
      selectActive(index);
    });
  });
});

previousButton.addEventListener("click", () => selectActive(activeIndex - 1));
nextButton.addEventListener("click", () => selectActive(activeIndex + 1));

clearActiveButton.addEventListener("click", () => {
  points[activeIndex] = null;
  redraw();
});

resetButton.addEventListener("click", () => {
  for (let index = 0; index < 4; index++) points[index] = null;
  activeIndex = 0;
  redraw();
});

document.querySelectorAll(".zoom").forEach(button => {
  button.addEventListener("click", () => setZoom(button.dataset.zoom));
});

saveButton.addEventListener("click", async () => {
  saveButton.disabled = true;
  statusBox.textContent = "Saving...";
  try {
    const response = await fetch(`/save?token=${encodeURIComponent(token)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({points: points.map(point => [point.x, point.y])})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    statusBox.textContent =
      `Saved: ${result.output}\nThe server will now stop.`;
  } catch (error) {
    statusBox.textContent = `Save failed: ${error.message}`;
    saveButton.disabled = false;
  }
});
</script>
</body>
</html>
"""
    values = {
        "__IMAGE_WIDTH__": str(width),
        "__IMAGE_HEIGHT__": str(height),
        "__BOARD_WIDTH__": f"{board.width:g}",
        "__BOARD_HEIGHT__": f"{board.height:g}",
        "__BOARD_UNIT__": board.unit,
        "__TOKEN__": token,
    }
    for marker, value in values.items():
        template = template.replace(marker, value)
    return template.encode("utf-8")


@dataclass
class SelectionServerState:
    frame_png: bytes
    html: bytes
    source: dict[str, Any]
    board: BoardSpec
    output: Path
    overwrite: bool
    token: str
    saved: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def make_handler(state: SelectionServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SyntheticCalibration/1"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[HTTP] {self.address_string()} {format_string % args}")

        def _token_is_valid(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            provided = query.get("token", [""])[0]
            return secrets.compare_digest(provided, state.token)

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            status: HTTPStatus,
            payload: dict[str, Any],
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not self._token_is_valid():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid token"})
                return
            if parsed.path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    state.html,
                    "text/html; charset=utf-8",
                )
            elif parsed.path == "/frame.png":
                self._send_bytes(HTTPStatus.OK, state.frame_png, "image/png")
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/save":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._token_is_valid():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid token"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid request size"},
                )
                return
            try:
                request = json.loads(self.rfile.read(content_length))
                if not isinstance(request, dict):
                    raise CalibrationError("request body must be an object")
                points = request.get("points")
                payload = build_selection_payload(
                    source=state.source,
                    board=state.board,
                    image_points=points,
                )
                with state.lock:
                    if state.saved:
                        raise CalibrationError("selection has already been saved")
                    write_json(
                        state.output,
                        payload,
                        overwrite=state.overwrite,
                    )
                    state.saved = True
            except (json.JSONDecodeError, CalibrationError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )
                return

            self._send_json(
                HTTPStatus.OK,
                {
                    "saved": True,
                    "output": str(state.output),
                    "status": "unverified",
                },
            )
            print(f"[OK] point selection: {state.output}")
            threading.Timer(0.25, self.server.shutdown).start()

    return Handler


def create_server(
    state: SelectionServerState,
    port: int,
) -> ThreadingHTTPServer:
    if port < 0 or port > 65535:
        raise CalibrationError("--port must be between 0 and 65535")
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    except OSError as exc:
        raise CalibrationError(
            f"could not listen on 127.0.0.1:{port}: {exc}"
        ) from exc


def main() -> int:
    args = build_parser().parse_args()
    try:
        board = resolve_board(args)
        frame, source = load_source_frame(
            image_path=args.image,
            video_path=args.video,
            frame_index=args.frame,
        )
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise CalibrationError(
                f"output already exists: {output}; pass --overwrite to replace it"
            )
        token = secrets.token_urlsafe(24)
        state = SelectionServerState(
            frame_png=encode_png(frame),
            html=render_html(
                width=int(source["width"]),
                height=int(source["height"]),
                token=token,
                board=board,
            ),
            source=source,
            board=board,
            output=output,
            overwrite=args.overwrite,
            token=token,
        )
        server = create_server(state, args.port)
        port = int(server.server_address[1])
        url = f"http://127.0.0.1:{port}/?token={token}"
        print("[READY] SSH browser board selector")
        print(f"[URL] {url}")
        print(
            f"[INFO] source={source['path']} "
            f"frame={source['frame_index']} "
            f"size={source['width']}x{source['height']}"
        )
        print(f"[INFO] output={output}")
        print(
            "[INFO] Forward this port in VS Code PORTS, open the URL in the "
            "Windows browser, click four points, and press Save."
        )
        print("[INFO] Ctrl+C cancels without writing.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[CANCEL] server stopped; no new selection was written")
        finally:
            server.server_close()
        return 0 if state.saved else 1
    except CalibrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

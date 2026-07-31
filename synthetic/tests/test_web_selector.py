#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from synthetic.calibration.common import BoardSpec
from synthetic.calibration.select_board_points_web import (
    SelectionServerState,
    build_selection_payload,
    create_server,
    encode_png,
    render_html,
)


class WebSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "kind": "image",
            "path": "/tmp/test-frame.png",
            "frame_index": None,
            "width": 640,
            "height": 480,
        }
        self.board = BoardSpec(width=1.0, height=1.0, unit="normalized")
        self.points = [
            [50.0, 40.0],
            [590.0, 45.0],
            [580.0, 440.0],
            [60.0, 430.0],
        ]

    def test_payload_is_unverified_and_ordered(self) -> None:
        payload = build_selection_payload(
            source=self.source,
            board=self.board,
            image_points=self.points,
        )
        self.assertEqual(payload["status"], "unverified")
        self.assertEqual(
            payload["point_order"],
            ["top_left", "top_right", "bottom_right", "bottom_left"],
        )
        self.assertEqual(payload["correspondences"][2]["board_xy"], [1.0, 1.0])

    def test_html_uses_original_image_coordinates(self) -> None:
        html = render_html(
            width=640,
            height=480,
            token="test-token",
            board=self.board,
        ).decode("utf-8")
        self.assertIn('width="640"', html)
        self.assertIn('height="480"', html)
        self.assertIn("test-token", html)
        self.assertIn("canvas.width / rect.width", html)
        self.assertIn("Math.min(canvas.width - 1, rawX)", html)
        self.assertIn("Math.min(canvas.height - 1, rawY)", html)
        self.assertIn('data-zoom="200"', html)
        self.assertIn('class="point-row active"', html)
        self.assertIn('event.key === "Enter"', html)
        self.assertIn("event.shiftKey ? 10 : 1", html)
        self.assertIn("points[activeIndex] = eventToImagePoint(event)", html)

    def test_http_save_writes_point_selection_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "board_points.json"
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 40), (590, 440), (255, 255, 255), 2)
            token = "test-token"
            state = SelectionServerState(
                frame_png=encode_png(frame),
                html=render_html(
                    width=640,
                    height=480,
                    token=token,
                    board=self.board,
                ),
                source=self.source,
                board=self.board,
                output=output,
                overwrite=False,
                token=token,
            )
            server = create_server(state, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            frame_response = urlopen(
                f"http://127.0.0.1:{port}/frame.png?token={token}",
                timeout=2,
            )
            self.assertEqual(frame_response.status, 200)
            self.assertEqual(frame_response.headers["Content-Type"], "image/png")

            body = json.dumps({"points": self.points}).encode("utf-8")
            request = Request(
                f"http://127.0.0.1:{port}/save?token={token}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urlopen(request, timeout=2)
            result = json.loads(response.read())
            self.assertTrue(result["saved"])

            thread.join(timeout=2)
            server.server_close()
            self.assertFalse(thread.is_alive())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "unverified")
            self.assertEqual(len(payload["correspondences"]), 4)

    def test_invalid_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = SelectionServerState(
                frame_png=encode_png(np.zeros((20, 20, 3), dtype=np.uint8)),
                html=b"test",
                source={
                    **self.source,
                    "width": 20,
                    "height": 20,
                },
                board=self.board,
                output=Path(temporary_directory) / "unused.json",
                overwrite=False,
                token="correct",
            )
            server = create_server(state, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                with self.assertRaises(HTTPError) as context:
                    urlopen(
                        f"http://127.0.0.1:{port}/?token=wrong",
                        timeout=2,
                    )
                self.assertEqual(context.exception.code, 403)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

"""Play LeRobot videos with frame-synchronized robot joint data."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.depth_utils import dequantize_depth

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except Exception:  # pragma: no cover - optional fallback for simple local datasets
    LeRobotDatasetMetadata = None


DEFAULT_JOINT_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
    "gripper.pos",
]


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _resolve_root(dataset_repo_id: str | None, dataset_root: str | None) -> Path:
    if dataset_root:
        return Path(dataset_root).expanduser().resolve()
    if dataset_repo_id and Path(dataset_repo_id).exists():
        return Path(dataset_repo_id).expanduser().resolve()
    raise ValueError("--dataset-root is required unless --dataset-repo-id is a local path")


def _metadata(dataset_repo_id: str | None, root: Path) -> Any | None:
    if LeRobotDatasetMetadata is None or not dataset_repo_id:
        return None
    try:
        return LeRobotDatasetMetadata(dataset_repo_id, root=root)
    except Exception:
        return None


def _format_path(
    template: str,
    *,
    episode: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    # This fallback matches standard LeRobot chunk/file naming for common local datasets.
    chunk_index = episode // chunks_size
    file_index = episode // chunks_size
    return Path(
        template.format(
            episode_index=episode,
            chunk_index=chunk_index,
            file_index=file_index,
            video_key=video_key,
        )
    )


def _data_path(root: Path, info: dict[str, Any], meta: Any | None, episode: int) -> Path:
    if meta is not None and hasattr(meta, "get_data_file_path"):
        return Path(meta.root) / meta.get_data_file_path(episode)
    return root / _format_path(info["data_path"], episode=episode, chunks_size=int(info.get("chunks_size", 1000)))


def _video_path(
    root: Path,
    info: dict[str, Any],
    meta: Any | None,
    episode: int,
    video_key: str,
) -> Path:
    if meta is not None and hasattr(meta, "get_video_file_path"):
        try:
            return Path(meta.root) / meta.get_video_file_path(episode, video_key)
        except TypeError:
            try:
                return Path(meta.root) / meta.get_video_file_path(video_key, episode)
            except TypeError:
                pass
    return root / _format_path(
        info["video_path"],
        episode=episode,
        chunks_size=int(info.get("chunks_size", 1000)),
        video_key=video_key,
    )


def _is_depth_key(info: dict[str, Any], key: str) -> bool:
    return bool(info.get("features", {}).get(key, {}).get("info", {}).get("is_depth_map"))


def _video_keys(info: dict[str, Any], requested: list[str] | None, view: str = "both") -> list[str]:
    if requested:
        return requested
    features = info.get("features", {})
    keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if view == "both":
        return keys
    filtered = [k for k in keys if _is_depth_key(info, k) == (view == "depth")]
    if not filtered:
        print(f"[WARN] --view {view} but no matching stream found — showing all cameras instead")
        return keys
    return filtered


DEPTH_MIN_MM = 100.0
DEPTH_MAX_MM = 3000.0


def _depth_params(info: dict[str, Any], key: str) -> dict[str, Any]:
    feature_info = info.get("features", {}).get(key, {}).get("info", {})
    return {
        "depth_min": feature_info.get("video.depth_min", 0.0),
        "depth_max": feature_info.get("video.depth_max", 10.0),
        "shift": feature_info.get("video.shift", 3.5),
        "use_log": feature_info.get("video.use_log", True),
        "output_tensor": False,
    }


def _colorize_depth_frame(quantized: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """양자화된 12-bit depth code(uint16, HxW)를 실측 mm로 복원한 뒤 사람이 보기
    좋은 컬러맵 BGR로 변환 — docs/depth/tools/depth_video_viewer.py와 동일한 방식.
    저장된 MP4/데이터 자체는 건드리지 않고 미리보기에서만 씀."""
    depth_mm = dequantize_depth(quantized, **params).squeeze()
    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM)
    normalized = ((clipped - DEPTH_MIN_MM) * (255.0 / (DEPTH_MAX_MM - DEPTH_MIN_MM))).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[depth_mm <= DEPTH_MIN_MM] = 0
    return colored


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    return list(names) if names else DEFAULT_JOINT_NAMES


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def _draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    width = max(1, int(w * height / max(1, h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _panel(
    row: pd.Series,
    frame_pos: int,
    total: int,
    joint_names: list[str],
    data_key: str,
    action_key: str | None,
    width: int,
    height: int,
    paused: bool,
) -> np.ndarray:
    panel = np.full((height, width, 3), (28, 30, 34), dtype=np.uint8)
    timestamp = float(row["timestamp"]) if "timestamp" in row else 0.0
    frame_index = int(row["frame_index"]) if "frame_index" in row else frame_pos

    y = 28
    _draw_text(panel, "Piper replay player", (14, y), scale=0.62, color=(255, 255, 255), thickness=2)
    y += 30
    _draw_text(panel, f"frame {frame_pos + 1}/{total}  dataset_frame={frame_index}", (14, y))
    y += 22
    _draw_text(panel, f"t={timestamp:.3f}s  {'PAUSED' if paused else 'PLAYING'}", (14, y), color=(190, 220, 255))
    y += 28

    state = _as_float_list(row[data_key]) if data_key in row else []
    action = _as_float_list(row[action_key]) if action_key and action_key in row else []

    x_name, x_state, x_action = 14, 180, 290
    _draw_text(panel, "Joint", (x_name, y), color=(160, 200, 170))
    _draw_text(panel, "State", (x_state, y), color=(160, 200, 170))
    _draw_text(panel, "Action", (x_action, y), color=(160, 200, 170))
    y += 22

    for i, name in enumerate(joint_names):
        _draw_text(panel, f"{name[:16]}", (x_name, y))
        
        if i < len(state):
            _draw_text(panel, f"{state[i]: >8.3f}", (x_state, y))
        else:
            _draw_text(panel, "-", (x_state, y))
            
        if i < len(action):
            _draw_text(panel, f"{action[i]: >8.3f}", (x_action, y))
        else:
            _draw_text(panel, "-", (x_action, y))
            
        y += 21

    y = height - 110
    _draw_text(panel, "- CONTROLS -", (14, y), scale=0.45, color=(220, 220, 220))
    y += 18
    _draw_text(panel, "space: Pause/Resume | q, esc: Quit", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    _draw_text(panel, ", . or left/right: Seek Frame | a, d: Seek 10 Frames", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    _draw_text(panel, "up/down arrow: Prev/Next Episode", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    _draw_text(panel, "+/-: adjust speed by 0.25x", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    
    _draw_text(panel, f"data: {data_key}" + (f"  action: {action_key}" if action_key else ""), (14, y), scale=0.43, color=(185, 185, 185))
    return panel


def _load_video_frames(
    path: Path, *, is_depth: bool = False, depth_params: dict[str, Any] | None = None
) -> list[np.ndarray] | None:
    """path의 영상을 전부 디코딩해서 프레임 리스트로 반환, 실패하면 None.

    cv2.VideoCapture(내장 ffmpeg의 AV1 디코더)는 이 프로젝트의 mp4(AV1/libdav1d
    인코딩)를 seek할 때 'Missing Sequence Header'로 깨져서 프레임을 못 읽어옴
    (seek 없이 순차로 읽어도 마찬가지로 실패함) — 대신 PyAV(av)로 처음부터
    순차 디코딩해서 메모리에 전부 올려놓음. episode 길이가 짧아서 메모리도
    부담 없음(scripts/tools/piper_replay_player_rviz.py와 동일한 방식).

    is_depth=True면 gray12le 12-bit code를 그대로 디코딩한 뒤 _colorize_depth_frame()으로
    mm 단위 컬러맵 BGR로 변환함(원본 RGB 디코딩과 동일한 배열 형태로 맞춰서 이후
    파이프라인이 카메라 종류를 신경 쓸 필요 없게 함)."""
    try:
        container = av.open(str(path))
        if is_depth:
            frames = [
                _colorize_depth_frame(f.to_ndarray(format="gray12le"), depth_params or {})
                for f in container.decode(video=0)
            ]
        else:
            frames = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
        container.close()
    except Exception as e:
        print(f"[WARN] Could not open video: {path} ({e})")
        return None
    if not frames:
        print(f"[WARN] No frames decoded from video: {path}")
        return None
    return frames


def play(initial_args: argparse.Namespace) -> None:
    args = initial_args
    root = _resolve_root(args.dataset_repo_id, args.dataset_root)

    while True:
        info = _load_info(root)
        meta = _metadata(args.dataset_repo_id, root)

        try:
            total_episodes = int(info.get("episodes", -1))
            if total_episodes == -1 and LeRobotDatasetMetadata is not None:
                meta_for_count = LeRobotDatasetMetadata(args.dataset_repo_id, root=root)
                total_episodes = len(meta_for_count)
        except Exception:
            total_episodes = args.episode + 1

        data_path = _data_path(root, info, meta, args.episode)
        try:
            df = pd.read_parquet(data_path)
            if "episode_index" in df.columns:
                df = df[df["episode_index"] == args.episode].reset_index(drop=True)
            if df.empty:
                print(f"[WARN] No rows found for episode {args.episode} in {data_path}. Skipping.")
                if args.episode > 0:
                    args.episode -= 1
                    continue
                else:
                    break
        except FileNotFoundError:
            print(f"[WARN] Data file not found for episode {args.episode}: {data_path}. Skipping.")
            if args.episode > 0:
                args.episode -= 1
                continue
            else:
                break

        video_keys = _video_keys(info, args.video_key, args.view)
        if not video_keys:
            raise ValueError("No video features found. Pass --video-key if the metadata is unusual.")

        video_frames: dict[str, list[np.ndarray] | None] = {}
        for key in video_keys:
            path = _video_path(root, info, meta, args.episode, key)
            is_depth = _is_depth_key(info, key)
            video_frames[key] = _load_video_frames(
                path, is_depth=is_depth, depth_params=_depth_params(info, key) if is_depth else None
            )
            # 비디오가 없어도(None) 플레이어는 계속 동작함 — placeholder로 표시

        fps = float(args.fps or info.get("fps") or 30)
        delay = max(1, int(1000 / (fps * args.speed)))
        data_key = args.data_key
        action_key = args.action_key if args.action_key in df.columns else None
        joint_names = _feature_names(info, data_key)

        valid_lengths = [len(f) for f in video_frames.values() if f is not None]
        total_frames = min(len(df), *valid_lengths) if valid_lengths else len(df)

        frame_pos = 0
        paused = args.start_paused
        last_tick = time.monotonic()

        print("-" * 30)
        print(f"Dataset root: {root.name}")
        print(f"Playing Episode: {args.episode + 1} / {total_episodes}")
        print(f"Frames: {total_frames}, FPS: {fps:g}, Speed: {args.speed:g}x")
        print("Video keys:", ", ".join(video_keys))
        print("Controls: (space)pause | (left/right)seek | (a/d)seek 10 | (up/down)episode | (q)quit")

        episode_running = True

        overlay_text = ""
        overlay_end_time = 0

        while 0 <= frame_pos < total_frames and episode_running:
            frames = []
            for key in video_keys:
                decoded = video_frames.get(key)
                if decoded is None:
                    frame = np.full((240, 424, 3), (20, 20, 20), dtype=np.uint8)
                    _draw_text(frame, f"video not found: {key}", (18, 38), color=(80, 80, 255))
                elif frame_pos >= len(decoded):
                    frame = np.full((240, 424, 3), (20, 20, 20), dtype=np.uint8)
                    _draw_text(frame, f"missing frame: {key}", (18, 38), color=(80, 170, 255))
                else:
                    frame = decoded[frame_pos].copy()  # 캐시된 원본을 mutate하지 않도록 복사
                _draw_text(frame, key, (12, 24), color=(40, 230, 255), thickness=2)
                frames.append(_resize_to_height(frame, args.video_height))

            video_strip = cv2.vconcat(frames)
            # 패널에 에피소드 정보 추가
            panel = _panel(
                df.iloc[frame_pos],
                frame_pos,
                total_frames,
                joint_names,
                data_key,
                action_key,
                args.panel_width,
                video_strip.shape[0],
                paused,
            )
            # 에피소드 정보를 패널에 덮어쓰기
            _draw_text(panel, f"Episode {args.episode + 1}/{total_episodes}", (panel.shape[1] - 150, 28))


            canvas = cv2.hconcat([video_strip, panel])

            if time.monotonic() < overlay_end_time:
                h, w = canvas.shape[:2]
                (text_w, text_h), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
                
                rect_x1 = (w - text_w) // 2 - 20
                rect_y1 = (h - text_h) // 2 - 20
                rect_x2 = rect_x1 + text_w + 40
                rect_y2 = rect_y1 + text_h + 40

                rect_x1, rect_y1 = max(0, rect_x1), max(0, rect_y1)
                rect_x2, rect_y2 = min(w, rect_x2), min(h, rect_y2)

                try:
                    sub_img = canvas[rect_y1:rect_y2, rect_x1:rect_x2]
                    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                    res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 0)
                    canvas[rect_y1:rect_y2, rect_x1:rect_x2] = res
                except Exception:
                    pass

                cv2.putText(canvas, overlay_text, ((w - text_w) // 2, (h + text_h) // 2), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(args.window_name, canvas)

            key = cv2.waitKey(delay if not paused else 30) & 0xFF

            # --- 키 입력 처리 ---
            if key in (ord("q"), 27):
                episode_running = False
                break
            # 배속 조절
            elif key in (ord('+'), ord('=')):  # + 또는 = 키
                if args.speed < 4.0:  # 최대 4배속
                    args.speed += 0.25
                    delay = max(1, int(1000 / (fps * args.speed)))
                    overlay_text = f"Speed: {args.speed:.2f}x"
                    overlay_end_time = time.monotonic() + 1.0
            elif key == ord('-'):  # - 키
                if args.speed > 0.25:  # 최소 0.25배속
                    args.speed -= 0.25
                    delay = max(1, int(1000 / (fps * args.speed)))
                    overlay_text = f"Speed: {args.speed:.2f}x"
                    overlay_end_time = time.monotonic() + 1.0
            # 에피소드 전환
            elif key in (82, ord('[')):  # 위쪽 방향키 또는 [
                if args.episode < total_episodes - 1:
                    args.episode += 1
                    episode_running = False
                else:
                    overlay_text = "This is the last episode"
                    overlay_end_time = time.monotonic() + 1.0
            elif key in (84, ord(']')):  # 아래쪽 방향키 또는 ]
                if args.episode > 0:
                    args.episode -= 1
                    episode_running = False
                else:
                    overlay_text = "This is the first episode"
                    overlay_end_time = time.monotonic() + 1.0
            # 프레임 탐색 및 일시정지
            elif key == ord(" "):
                paused = not paused
            elif key in (81, ord(",")):
                frame_pos = max(0, frame_pos - 1)
                paused = True
            elif key in (83, ord(".")):
                frame_pos = min(total_frames - 1, frame_pos + 1)
                paused = True
            elif key == ord("a"):
                frame_pos = max(0, frame_pos - 10)
                paused = True
            elif key == ord("d"):
                frame_pos = min(total_frames - 1, frame_pos + 10)
                paused = True
            
            if not paused and episode_running:
                now = time.monotonic()
                if now - last_tick >= 1.0 / max(1e-6, fps * args.speed):
                    frame_pos += 1
                    last_tick = now
        
        if key in (ord("q"), 27):
            break
    
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively play LeRobot dataset videos synchronized with joint record data.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
INTERACTIVE CONTROLS:
  space: Pause / Resume
  , / . or left/right arrow: Seek 1 frame
  a / d: Seek 10 frames
  [ / ] or up/down arrow: Previous / Next episode
  q / esc: Quit
  +/-: Adjust speed by 0.25x
""".strip(),
    )
    # -------------------------------------------------------------
    # 아래의 parser.add_argument(...) 라인들은 전혀 바꿀 필요 없습니다.
    # -------------------------------------------------------------
    parser.add_argument("--dataset-repo-id", default=None, help="LeRobot repo id or local dataset path")
    parser.add_argument("--dataset-root", default=None, help="Local LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to play")
    parser.add_argument("--video-key", action="append", help="Video feature key to display. Repeat for multiple cameras")
    parser.add_argument(
        "--view", choices=["both", "rgb", "depth"], default="both",
        help="Filter auto-discovered video streams to RGB only, depth only, or both (default)",
    )
    parser.add_argument("--data-key", default="observation.state", help="Joint state column to display")
    parser.add_argument("--action-key", default="action", help="Action column to display beside state")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--start-frame", type=int, default=0, help="Initial frame index")
    parser.add_argument("--start-paused", action="store_true", help="Start paused for frame-by-frame inspection")
    parser.add_argument("--video-height", type=int, default=480, help="Display height for each video")
    parser.add_argument("--panel-width", type=int, default=430, help="Width of the joint-data panel")
    parser.add_argument("--window-name", default="Piper replay player", help="OpenCV window title")
    return parser.parse_args()

def main() -> None:
    play(parse_args())


if __name__ == "__main__":
    main()

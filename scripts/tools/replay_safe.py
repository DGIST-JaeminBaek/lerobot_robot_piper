#!/usr/bin/env python
"""시작 자세로 먼저 이동한 뒤 재생하는 리플레이.

lerobot-replay는 시작 자세 정렬 없이 곧장 재생 루프로 들어간다
(lerobot/scripts/lerobot_replay.py). 팔이 에피소드 첫 자세와 떨어져 있으면
max_relative_target(5.0) 클램프에 걸려 매 스텝 "현재 위치 + 5"만 나가고,
그 사이 타임라인은 계속 흘러서 팔이 궤적을 따라가지 못한 채 등속으로 기어간다.
= 8/4 리플레이 영상에서 본 그 증상.

여기서는 재생 전에 첫 자세로 천천히 정렬하고, 도달을 확인한 뒤 재생한다.

usage:
  python replay_safe.py --root <DATASET_ROOT> --dry-run      # 움직이지 않고 자세 차이만 출력
  python replay_safe.py --root <DATASET_ROOT>                # 실제 재생
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_piper.config_piper import PiperFollowerConfig
from lerobot_robot_piper.piper_follower import PiperFollower

logger = logging.getLogger(__name__)


def episode_actions(root: str, episode: int) -> tuple[np.ndarray, list[str]]:
    ds = LeRobotDataset(repo_id="local/replay", root=root, episodes=[episode])
    frames = ds.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    names = ds.features["action"]["names"]
    return np.array(frames.select_columns("action")["action"], dtype=np.float32), names


def first_unclipped(actions: np.ndarray) -> int:
    """정규화 한계(±100)에 붙은 선두 프레임 수.

    파킹 자세는 캘리브 범위(piper_follower.py의 하드코딩 range) 밖에 있어서
    joint2=-100 / joint3=+100 으로 잘려 기록된다. 이건 실제 자세가 아니라 범위 끝점이라,
    그대로 명령하면 팔이 파킹 위치가 아닌 엉뚱한 자세로 간다.
    재생은 여기를 지나 실측값이 시작되는 지점부터 하는 게 맞다."""
    clipped = (np.abs(actions) >= 99.99).any(axis=1)
    return int(np.argmax(~clipped)) if (~clipped).any() else len(clipped)


def preflight(actions: np.ndarray, start: np.ndarray, args) -> None:
    """팔을 움직이지 않고 재생 성공 여부를 미리 판정한다.

    실제 현재 자세를 시작점으로 넣고 sim_replay의 팔 모델(클램프 + 실측 속도 한계)로
    재생을 돌려본다. 명령은 한 줄도 나가지 않는다."""
    from sim_replay import run  # 같은 디렉터리

    speed = float(np.percentile(np.abs(np.diff(actions, axis=0)).max(axis=1), 99))
    checks = []

    clipped = int((np.abs(actions[0]) >= 99.99).sum())
    checks.append((clipped == 0, f"시작 목표가 정규화 한계에 잘리지 않음 (잘린 축 {clipped}개)"))

    gap = float(np.abs(actions[0] - start).max())
    checks.append((gap <= args.preflight_gap,
                   f"시작 자세 차이 {gap:.1f} ≤ {args.preflight_gap} (정렬 부담)"))

    r = run(actions, start, speed, align=True)
    checks.append((r["err_max"] <= args.abort_err,
                   f"예상 최대 추종 오차 {r['err_max']:.2f} ≤ {args.abort_err}"))
    checks.append((r["bad_frac"] < 0.01,
                   f"예상 오차>5 프레임 비율 {r['bad_frac']:.1%} < 1%"))

    # offset을 끄지 않고 돌렸을 때 무슨 일이 생기는지도 같이 보여준다
    r_off = run(actions, start, speed, align=False, use_offset=True)

    print()
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"\n  참고 — offset을 끄지 않고 재생하면: 평균오차 {r_off['err_mean']:.1f}, "
          f"오차>5 프레임 {r_off['bad_frac']:.0%}")
    verdict = all(ok for ok, _ in checks)
    print(f"\n  판정: {'재생해도 됩니다' if verdict else '재생하지 마세요 — 위 FAIL 항목을 먼저 해결'}\n")
    if not verdict:
        raise SystemExit(1)


def present(robot: PiperFollower, names: list[str]) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[n] for n in names], dtype=np.float32)


def align(robot: PiperFollower, target: np.ndarray, names: list[str], args) -> bool:
    """첫 자세로 천천히 정렬. 도달하면 True."""
    deadline = time.perf_counter() + args.align_timeout
    while time.perf_counter() < deadline:
        cur = present(robot, names)
        gap = target - cur
        if np.abs(gap).max() <= args.align_tol:
            return True
        # 클램프(5.0)보다 작은 보폭으로 나눠 보낸다 — 정렬은 급할 이유가 없다
        step = np.clip(gap, -args.align_step, args.align_step)
        robot.send_action({n: float(v) for n, v in zip(names, cur + step)})
        precise_sleep(1 / args.fps)
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--port", default="can_follower1")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--align-step", type=float, default=1.0, help="정렬 시 스텝당 최대 이동 (정규화 단위)")
    p.add_argument("--align-tol", type=float, default=1.0, help="정렬 완료 판정 오차")
    p.add_argument("--align-timeout", type=float, default=30.0)
    p.add_argument("--abort-err", type=float, default=8.0, help="추종 오차가 이 값을 넘으면 이상 (정규화 단위)")
    p.add_argument("--abort-frames", type=int, default=10, help="이만큼 연속 초과하면 재생 중단")
    p.add_argument("--replay-clipped", action="store_true",
                   help="파킹 자세(정규화 한계에 잘린 값)까지 그대로 재생한다. 팔이 엉뚱한 자세로 갈 수 있음")
    p.add_argument("--preflight", action="store_true",
                   help="팔을 움직이지 않고 재생 성공 여부를 미리 판정한다 (실패 시 exit 1)")
    p.add_argument("--preflight-gap", type=float, default=60.0, help="허용할 시작 자세 차이")
    p.add_argument("--dry-run", action="store_true", help="연결·관측만 하고 명령은 보내지 않음")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    actions, names = episode_actions(args.root, args.episode)
    logger.info("episode %d: %d frames (%.1fs)", args.episode, len(actions), len(actions) / args.fps)

    if not args.replay_clipped:
        n = first_unclipped(actions)
        if n:
            logger.info("선두 %d프레임(%.1fs)이 정규화 한계에 잘린 파킹 자세 — 건너뛰고 재생", n, n / args.fps)
            actions = actions[n:]
        if len(actions) == 0:
            raise SystemExit("전 구간이 클리핑 상태 — 재생할 실측 궤적이 없다")

    # 녹화된 action은 이미 follower 절대 좌표계 값 — offset을 또 얹으면 이중 보정이 된다
    robot = PiperFollower(PiperFollowerConfig(id="replay", port=args.port, use_action_offset=False))
    robot.connect()
    try:
        gap = actions[0] - present(robot, names)
        logger.info("시작 자세 차이: %s (max %.1f)", np.round(gap, 1), np.abs(gap).max())

        if args.dry_run or args.preflight:
            if args.preflight:
                preflight(actions, present(robot, names), args)
            logger.info("[명령 없이 종료]")
            return

        if not align(robot, actions[0], names, args):
            raise SystemExit("시작 자세 정렬 실패 (align-timeout). 팔 상태를 확인할 것")
        logger.info("정렬 완료, 재생 시작")

        # 팔이 궤적을 못 따라가는 상황을 추종 오차 하나로 잡는다.
        # 안전 컷오프(effort 초과로 마지막 명령 유지), 충돌, 도달 불가 자세 — 전부 여기로 드러난다.
        # send_action의 반환값으로 컷오프를 판정하려 하면 안 된다: 정상 추종 중에도
        # max_relative_target 클램프 때문에 반환값이 명령값과 다르다.
        over = 0
        for i, a in enumerate(actions):
            t0 = time.perf_counter()
            robot.send_action({n: float(v) for n, v in zip(names, a)})
            err = float(np.abs(a - present(robot, names)).max())

            if err > args.abort_err:
                over += 1
                if over >= args.abort_frames:
                    raise SystemExit(
                        f"{i / args.fps:.1f}s 지점에서 추종 오차 {err:.1f}이 "
                        f"{over}프레임 연속 {args.abort_err} 초과 — 재생 중단. "
                        "안전 컷오프(effort 초과)나 충돌 가능성. 팔 주변을 확인할 것"
                    )
            else:
                over = 0

            precise_sleep(max(1 / args.fps - (time.perf_counter() - t0), 0.0))
            if i % (args.fps * 5) == 0:
                logger.info("  %5.1fs  추종 오차 %.1f", i / args.fps, err)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

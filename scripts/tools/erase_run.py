#!/usr/bin/env python3
"""erase_run.py — 지우기 과제 closed-loop 추론 러너 (종료 게이트 + 선택적 HIL 개입).

배경은 docs/erase_run_design.md 참고. 요약:
  - 잉크 metric은 팔이 도형을 가리는 동안 못 쓴다(에피소드의 51%가 가려짐).
    그래서 판정은 chunk 경계가 아니라 **시도(attempt) 경계**에서 한다 —
    정책 1회 실행 → 팔 park → 사진 → erase_check 판정 → 실패면 재시도.
  - 그 덕에 async client의 chunk 큐(50스텝=1.67s) 문제를 아예 우회한다.
    대신 동기 루프를 직접 돈다(piper_infer_preview.py의 predict_action 경로 그대로).

사용:
    # 1) 리더암 읽기 점검 (로봇 안 움직임 — 안전)
    python erase_run.py --probe-leader --leader-port can_leader1

    # 2) 종료 게이트만 (HIL 없음)
    python erase_run.py --policy_path <ckpt> --dataset_root <학습에 쓴 데이터셋> \
        --target triangle --task "erase the triangle" --confirm

    # 3) HIL 개입 켜기 — space로 리더암에 제어권을 넘기고, space로 되돌린다
    python erase_run.py ... --hil --leader-port can_leader1 --confirm

주의:
  - --confirm 없이는 로봇에 명령을 보내지 않는다(dry 점검만).
  - park 복귀는 정책을 믿지 않고 러너가 강제한다. 기본은 follower.parking().
    parking()이 카메라 시야를 가리는 자세면 --park-pose로 관절값을 직접 준다.
  - 학습에 쓴 데이터셋의 features가 필요하다(관측 재포장용). --dataset_root 필수.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# erase_check(cv2 의존)와 lerobot은 main()에서 늦게 import한다 —
# 아래 중재/루프 로직은 순수 파이썬이라 하드웨어·cv2 없이 테스트된다.

# HIL 개입 로직(클러치/토글)은 hil_clutch.py에 있다 — InferenceRunner와 같은
# 구현을 공유해야 해서 공용 모듈로 뺐다. 아래 이름들은 기존 호출자·테스트가
# erase_run에서 그대로 import하고 있어 re-export로 유지한다.
from hil_clutch import (  # noqa: E402,F401
    ACTION_NAMES,
    GRIPPER,
    JOINTS,
    Clutch,
    ClutchMixer,
    KeyToggle,
    deviation,
)


# ══════════════════════════════════════════════════════════════
# 한 번의 시도 — 정책을 max_steps 동안 돌리고 park로 복귀
# robot/policy_fn/leader를 주입받아 하드웨어 없이도 테스트 가능
# ══════════════════════════════════════════════════════════════
def run_attempt(robot, policy_fn, park_fn, fps=30, max_steps=940,
                toggle=None, leader=None, clutch=None):
    """정책 1회 실행. 반환: 스텝 로그 리스트.

    max_steps는 종료 조건이 아니라 fallback이다 — 실제 종료 판정은 시도가 끝난 뒤
    park 상태에서 erase_check가 한다. (데이터셋 중앙값 ~720프레임의 약 1.3배)

    toggle.active가 True인 동안 리더암이 로봇을 몬다(클러치 델타 방식).
    toggle.abort가 True면 시도를 즉시 끝낸다.
    """
    dt = 1.0 / fps
    hil = toggle is not None and leader is not None and clutch is not None
    mixer = ClutchMixer(toggle, leader, clutch) if hil else None
    log = []
    for step in range(max_steps):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        action = policy_fn(obs)
        intervened = False
        dev = None

        if mixer is not None:
            action, intervened, dev = mixer.step(action, obs)

        robot.send_action(action)
        entry = {"step": step, "intervention": intervened,
                 "action": {k: float(v) for k, v in action.items()}}
        if dev is not None:
            entry["engage_deviation"] = round(dev, 2)
        log.append(entry)

        if toggle is not None and toggle.abort:
            break

        sleep = dt - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)

    if mixer is not None:
        mixer.release()
    park_fn()  # 정책이 park로 갈 거라고 믿지 않는다 — 러너가 강제한다
    return log


def grab_park_frame(robot, park_fn, cam_key="top", settle_s=1.0):
    """판정용 프레임. 팔이 park에 있고 정지한 뒤에 찍어야 도형이 다 보인다."""
    park_fn()
    time.sleep(settle_s)
    obs = robot.get_observation()
    if cam_key not in obs:
        raise KeyError(f"관측에 '{cam_key}' 카메라가 없다. 있는 키: {sorted(obs)}")
    return obs[cam_key]


# ══════════════════════════════════════════════════════════════
# 하드웨어 연결 (여기부터는 실제 로봇 필요)
# ══════════════════════════════════════════════════════════════
def build_robot(follower_port, top_cam, wrist_cam, cam_type="realsense"):
    from lerobot_robot_piper import PiperFollower, PiperFollowerConfig

    cfg = PiperFollowerConfig(
        id="erase_run", port=follower_port,
        camera_type=cam_type, top_cam=top_cam, wrist_cam=wrist_cam,
    )
    robot = PiperFollower(cfg)
    robot.connect()
    return robot


def build_leader(leader_port):
    from lerobot_robot_piper import PiperLeader, PiperLeaderConfig

    leader = PiperLeader(PiperLeaderConfig(id="erase_run", port=leader_port))
    leader.connect()
    return leader


# build_policy_fn은 제거했다. 자체 추론 경로를 들고 있었는데 두 가지가 틀렸다:
#   1) 카메라 크롭·리사이즈를 안 거치고 raw 관측을 그대로 정책에 먹였다. 학습은
#      512x512 크롭본으로 했고 실제 카메라는 1280x720이라 화각이 다르다.
#      (piper_infer_runner는 preprocess_live_camera_observation을 지난다)
#   2) predict_action은 1스텝만 반환하고 chunk를 노출하지 않아 temporal ensemble이
#      원리적으로 불가능했다 — 스무딩 없이 도는 셈이라 흔들림이 그대로 남는다.
# 이제 추론·스무딩·안전은 전부 InferenceRunner가 맡고, 이 파일은 시도 경계에서
# 판정하고 재시도를 결정하는 바깥 루프만 담당한다.


def grab_judge_frame(serial: str, width=1280, height=720, warmup_s=3.0, fps=30):
    """판정용 프레임을 top 카메라에서 직접 뜬다 (BGR HWC).

    정책 관측과 분리해서 읽는 이유: InferenceRunner가 시도마다 로봇·카메라를
    잡았다 놓으므로, 시도 사이에는 러너의 카메라 핸들이 없다. 판정기는 크롭 전
    원본 1280x720을 봐야 하고(board ROI가 그 좌표계다) 정책 입력과 요구가 달라서,
    따로 읽는 편이 결합도 낮다 — 설계 문서 §4.4의 "판정기 전용 센서" 방향과 같다.
    """
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        # 자동 노출이 자리잡을 때까지 버린다. 프레임 수가 아니라 시간 기준이어야
        # 한다 — recording.env의 REALSENSE_WARMUP_S=3.0과 같은 근거이고, 1.5초로
        # 줄였다가 덜 밝은 프레임이 나와 도형 검출이 통째로 실패한 적이 있다.
        deadline = time.time() + warmup_s
        frames = pipe.wait_for_frames()
        while time.time() < deadline:
            frames = pipe.wait_for_frames()
        image = np.asanyarray(frames.get_color_frame().get_data()).copy()
    finally:
        pipe.stop()

    # 판정 전체가 이 프레임 하나에 걸려 있으므로, 쓸 수 없는 프레임이면 여기서
    # 분명하게 죽는다. 아래로 흘려보내면 "도형을 못 찾음"으로 나와 원인이 가려진다.
    if float(image.max()) < 40:
        raise RuntimeError(
            f"판정용 프레임이 거의 검다 (max={image.max()}). "
            f"카메라 워밍업({warmup_s:g}s)이 짧거나 시리얼 {serial}이 틀렸을 수 있다."
        )
    return image


# 상태 블록을 쓰는 동안에도 반드시 보여야 하는 로그. 나머지는 삼킨다.
IMPORTANT_LOG_TAGS = ("[RECORD]", "[HIL]", "[ERROR]", "[STOP]", "[WARN]", "[SAFETY]")


def _filtered_log(message: str) -> None:
    if any(tag in message for tag in IMPORTANT_LOG_TAGS):
        print(f"    {message}", flush=True)


def run_attempt_with_runner(settings, on_log=None, on_step=None,
                            on_runner=None) -> dict:
    """InferenceRunner로 시도 1회. 반환: 요약 dict.

    러너는 스레드라 events 큐로 진행상황이 나온다. 여기서는 로그만 흘려보내고
    FINISHED를 기다린다. park는 러너가 disconnect(park=True)로 강제한다.
    """
    import queue as _queue

    from piper_infer_runner import Event, InferenceRunner

    run = InferenceRunner(settings)
    steps: list[dict] = []
    run.start()
    if on_runner:
        on_runner(run)

    def _pump_events() -> bool:
        """이벤트 큐를 계속 비운다. FINISHED를 만나면 True."""
        while True:
            try:
                kind, payload = run.events.get(timeout=1.0)
            except _queue.Empty:
                if not run.is_alive():
                    return True
                continue
            if kind == Event.LOG:
                if on_log:
                    on_log(payload)
                else:
                    print(f"    {payload}")
            elif kind == Event.STEP:
                # 스텝별 진단. 요약만 남기면 "왜 실패했는지"를 사후에 못 가린다 —
                # 클램프 포화가 원인인지 결과인지 같은 질문이 여기서 갈린다.
                steps.append(
                    {
                        "step": payload["step"],
                        "action": payload["action"],
                        "measured": payload["measured"],
                        "votes": payload["votes"],
                        "rate_clamp": payload["rate_clamp"],
                        "intervention": payload.get("intervention", False),
                        "infer_ms": payload["infer_ms"],
                    }
                )
                if on_step:
                    on_step(payload)
            elif kind == Event.FINISHED:
                return True

    try:
        _pump_events()
    except KeyboardInterrupt:
        # erase_eval_ui.py의 [컷오프]/[중단]은 프로세스 그룹에 SIGINT를 보낸다
        # ("러너의 park 정리를 태운다"는 주석이 있는데, 이 핸들러가 없으면 그
        # 기대가 거짓이 된다). 예전엔 여기서 그대로 죽어서 InferenceRunner가
        # disconnect(park=True) 도중(최대 ~15초, parking+램프+그리퍼 사이클)이면
        # 팔이 그 상태로 프로세스와 함께 끊기고, 녹화 중이던 영상도 마감이 안 돼
        # 다음 판정이 "top 영상 없음"으로 죽었다. 실물에서 실제로 겪었다
        # (2026-08-18, stop-on-release가 916/940에서 걸린 직후 60초 컷오프와
        # 거의 동시에 발생). stop_event만 세우고 정리가 끝날 때까지 계속
        # 이벤트를 뽑아준다 — piper_infer_runner.py 자체 CLI가 쓰는 것과 같은
        # 패턴(_drain_until_finished)이다.
        print("[INTERRUPT] 중단 요청 — 정리 중…")
        run.stop_event.set()
        _pump_events()
    run.join(timeout=30)
    return {
        "status": run.status,
        "steps": len(run.trajectory),
        "interventions": run.intervention_steps,
        "engage_deviations": [round(d, 2) for d in run.engage_deviations],
        "engage_deviation_details": run.engage_deviation_details,
        "measured_fps": round(run.measured_fps(), 2),
        "aborted": run.hil_aborted,
        "_steps": steps,
        # ★ 스무딩 이전의 정책 원출력(각 chunk의 첫 스텝). 실물 검증에서
        #   "그리퍼가 시연의 강한 폐합 명령(80~100)을 한 번도 안 낸다"가 나왔는데,
        #   그게 정책 원출력이 낮아서인지 EMA가 피크를 깎아서인지 가를 수 없었다.
        #   최종 action만 저장하면 그 질문에 영원히 답을 못 한다.
        "_raw": [a.tolist() for a in run.raw_trajectory],
    }


def save_step_traces(attempts: list[dict], out_path: Path) -> Path | None:
    """스텝별 궤적을 npz로 따로 뺀다 (JSON에 넣으면 수십 MB가 된다)."""
    arrays = {}
    for entry in attempts:
        steps = entry.pop("_steps", None)
        if not steps:
            entry.pop("_raw", None)  # JSON에 수천 줄이 새어나가지 않게
            continue
        i = entry["attempt"]
        arrays[f"attempt{i}_action"] = np.array([s["action"] for s in steps], dtype=np.float32)
        arrays[f"attempt{i}_measured"] = np.array([s["measured"] for s in steps], dtype=np.float32)
        # rate_clamp는 구현에 따라 스칼라이거나 관절별 배열이라 평균으로 눕힌다.
        arrays[f"attempt{i}_rate_clamp"] = np.array(
            [float(np.mean(np.asarray(s["rate_clamp"], dtype=np.float64)))
             if s["rate_clamp"] is not None else 0.0
             for s in steps],
            dtype=np.float32,
        )
        arrays[f"attempt{i}_intervention"] = np.array(
            [bool(s["intervention"]) for s in steps], dtype=bool
        )
        # votes = 그 스텝의 목표를 만드는 데 몇 개의 예측이 겹쳤나.
        # aggregate_fn 3종을 실물에서 비교하려면 이 값이 있어야 한다 —
        # 특히 HIL 반환 직후 votes가 1로 주저앉는 구간(개입 중 pipeline.reset으로
        # 앙상블 버퍼를 비우므로)이 방식마다 얼마나 오래 가는지가 관심사다.
        arrays[f"attempt{i}_votes"] = np.array(
            [int(s["votes"]) for s in steps], dtype=np.int16
        )
        raw = entry.pop("_raw", None)
        if raw:
            # 스텝 수와 길이가 다르다 — chunk가 도착한 횟수만큼만 있다.
            arrays[f"attempt{i}_raw"] = np.array(raw, dtype=np.float32)
    if not arrays:
        return None
    path = out_path.with_suffix(".steps.npz")
    np.savez_compressed(path, **arrays)
    return path


def check_dataset_matches_checkpoint(policy_path, dataset_root) -> None:
    """--dataset-root가 그 체크포인트를 학습시킨 데이터셋인지 확인한다.

    왜 필요한가 — 체크포인트는 학습 때의 정규화 통계를 함께 저장한다
    (policy_preprocessor_step_*_normalizer_processor.safetensors). 그런데
    train_config.json의 dataset.root는 **학습한 PC의 절대 경로**라 여기서는
    존재하지 않는 경우가 많고(예: /home/ugrp308/...), 그러면 사람이 손으로
    아무 데이터셋이나 물리게 된다.

    통계가 다른 데이터셋을 물리면 조용히 틀린 값이 나온다. 실측 예:
      smolvla_pickup_prompt_132_v2 의 state mean
        올바른 짝(...0727_0812_0813am_132): [-1.9 -10.1  54.5  5.4 10.3 -23.0 22.8]
        엉뚱한 짝(...0802_0804_0805_135) : [-1.5   0.2  29.9  0.4 42.0 -17.5 26.9]
      joint3가 54.5 vs 29.9, joint5가 10.3 vs 42.0이다. 이 상태로 돌면 정책이
      전혀 다른 좌표계에서 동작한다 — 그런데 실행은 멀쩡히 되므로 '모델이
      나쁘다'로 오인하기 딱 좋다.

    그래서 여기서 막는다. 경고가 아니라 중단이다.
    """
    from pathlib import Path as _P

    import numpy as _np

    ckpt = _P(policy_path)
    hits = sorted(ckpt.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not hits:
        print("[WARN] 체크포인트에 정규화 통계가 없어 데이터셋 정합을 확인하지 못했다.")
        return
    try:
        from safetensors.torch import load_file

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        baked = load_file(str(hits[0]))
        key = next(k for k in baked if "observation.state" in k and k.endswith("mean"))
        ck_mean = baked[key].numpy().ravel()
        ds_mean = _np.asarray(
            LeRobotDatasetMetadata("local/_check", root=str(dataset_root))
            .stats["observation.state"]["mean"],
            dtype=_np.float32,
        )
    except Exception as exc:  # 확인 실패가 실행을 막을 이유는 아니다
        print(f"[WARN] 데이터셋 정합 확인 실패({type(exc).__name__}: {exc}) — 그대로 진행한다.")
        return

    if _np.allclose(ck_mean, ds_mean, atol=1e-3):
        print("[OK] 데이터셋이 체크포인트의 학습 데이터와 일치한다 (정규화 통계 기준).")
        return

    raise SystemExit(
        "[STOP] --dataset-root가 이 체크포인트를 학습시킨 데이터셋이 아니다.\n"
        f"  체크포인트 state mean: {_np.round(ck_mean, 2)}\n"
        f"  준 데이터셋 state mean: {_np.round(ds_mean, 2)}\n"
        "  정규화가 어긋난 채로도 실행은 되지만 정책이 전혀 다른 좌표계에서\n"
        "  동작하게 된다. 올바른 데이터셋을 지정할 것."
    )


def make_park_fn(robot, park_pose: dict | None):
    if park_pose:
        return lambda: robot.send_action(dict(park_pose))
    return robot.parking


# ══════════════════════════════════════════════════════════════
# 리더암 상태 점검 — 로봇에 명령을 보내지 않는다
# 임계값 튜닝용이 아니라(키보드 토글이라 임계값이 없다) 배선/정렬 확인용.
# ══════════════════════════════════════════════════════════════
def probe_leader(leader, robot, seconds=10.0, fps=30):
    print(f"[INFO] {seconds:.0f}초간 리더암을 읽는다. 값이 갱신되는지, 손으로 움직이면 따라 바뀌는지 확인.")
    devs = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        devs.append(deviation(leader.get_action(), robot.get_observation()))
        time.sleep(1.0 / fps)
    devs = np.array(devs)
    print(f"  리더-팔로워 편차  n={len(devs)}  mean={devs.mean():.2f}  max={devs.max():.2f}")
    print("  (클러치 델타 방식이라 이 값이 커도 인계는 안전하다. 0에 가까우면 CAN 배선/정렬 정상.)")
    return devs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy_path")
    p.add_argument("--dataset_root", help="학습에 쓴 LeRobotDataset 루트 (features 참조용)")
    # ★ 학습 데이터의 task 문장 그대로여야 한다. 이 리포의 erase 데이터셋은 전부
    #   도형 이름이 없는 단일 문장이다("erase the shape" / "pick up the eraser and
    #   erase the shape"). 도형 이름을 넣으면 분포 밖이라 정책이 본 적 없는 입력이
    #   된다. --target(판정 대상)은 우리가 쥔 라벨이라 task와 별개다.
    p.add_argument("--task", default="pick up the eraser and erase the shape")
    p.add_argument("--target", default="triangle", choices=["circle", "triangle", "rectangle"])
    # 기본값은 configs/recording.env와 같아야 한다. 예전 기본값이 can_*1이었는데
    # 이 PC의 인터페이스는 can_follower / can_leader라 --probe-leader가 바로 죽었다.
    # 실제 추론 경로는 러너가 env에서 읽으므로 영향이 없었고, 그래서 안 드러났다.
    p.add_argument("--follower-port", default=os.environ.get("FOLLOWER_PORT", "can_follower"))
    p.add_argument("--leader-port", default=os.environ.get("LEADER_PORT", "can_leader"))
    p.add_argument("--top-cam", default="327122074262", help="판정용 top 카메라 시리얼")
    p.add_argument("--wrist-cam", default="")
    p.add_argument("--cam-type", default="realsense")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=940, help="시도당 상한 (중앙값 720의 약 1.3배)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--stop-on-release", action="store_true",
                   help="지우개를 놓으면 그 시도를 즉시 끊고 파킹한다 (러너로 전달). "
                        "판정은 어차피 park에서 하므로 판정 품질에는 영향이 없다")
    p.add_argument("--mode", default="demo", help="러너 모드 프리셋 (demo | augment)")
    p.add_argument(
        "--top-crop",
        default="280,0,720",
        help="X,Y,SIZE. 학습 데이터를 만든 값과 같아야 한다 "
             "(docs/training/smolvla_finetuning.md 기준 280,0,720)",
    )
    p.add_argument("--wrist-crop", default="", help="X,Y,SIZE. top_only 체크포인트면 비워둔다")
    p.add_argument("--camera-output-size", type=int, default=512)
    # 제어 인터페이스 노브 2개를 러너로 흘려보낸다. 실측에서 팔이 목표를 약 5.8만큼
    # 뒤처져 따라가 클램프(기본 5)에 상시 걸렸고, 그 구간에서는 명령이 '실측+5'로
    # 덮여 스무딩이 무의미해진다. 원인 가르기에 필요한 값들이다.
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="send_action 클램프. None이면 recording.env의 MAX_RELATIVE_TARGET(5)")
    p.add_argument("--move-speed-rate", type=int, default=None,
                   help="컨트롤러 이동 속도 %% (기본 30). 팔이 목표를 못 쫓아가면 이걸 올린다")
    p.add_argument("--hil", action="store_true", help="리더암 개입 활성화")
    p.add_argument("--clutch-gain", type=float, default=1.0, help="리더 변화량 -> 팔로워 반영 비율 (1.0=등배)")
    p.add_argument("--probe-leader", action="store_true", help="리더암 노이즈 플로어만 측정하고 종료")
    p.add_argument("--out", default="erase_run_log.json")
    # 녹화 데이터셋은 records/local/ 아래 쌓인다. HIL 산출물은 그 옆에 따로 모은다 —
    # 판정 프레임까지 같이 남겨야 나중에 임계값을 바꿔 재판정하거나, 판정이 틀렸을 때
    # 원인을 볼 수 있다. 지금까지는 판정 프레임을 그냥 버려서 매번 다시 찍어야 했다.
    p.add_argument("--hil-dir", default="records/hil",
                   help="HIL/게이트 산출물(로그·궤적·판정 프레임)을 모을 폴더")
    p.add_argument("--record-manual", action="store_true",
                   help="패널의 '녹화 시작/종료' 버튼으로 구간을 골라 LeRobotDataset에 "
                        "담는다. --mode augment(전 구간 자동 녹화)와 달리 개입 구간만 "
                        "골라 담을 수 있다")
    p.add_argument("--no-record-on-intervention", action="store_true",
                   help="space(개입 토글)에 녹화를 묶지 않는다. 기본은 묶여 있어서 "
                        "개입 시작=녹화 시작, 반환=에피소드 저장이 된다")
    p.add_argument("--record-raw-frames", action="store_true",
                   help="크롭 전 원본(1280x720)을 담는다. 기본은 정책 입력과 같은 "
                        "512x512 크롭본 — raw는 30Hz를 못 지킨다(실측 19.5Hz)")
    p.add_argument("--record-root", default=None,
                   help="수동 녹화 데이터셋 루트 (기본: records/hil/<시각>/dataset)")
    p.add_argument("--record-repo-id", default=None,
                   help="수동 녹화 repo_id (기본: local/hil_<시각>)")
    p.add_argument("--no-save-frames", action="store_true",
                   help="판정 프레임을 저장하지 않는다")
    p.add_argument("--panel", dest="panel", action="store_true", default=None,
                   help="화면에 상태 창을 띄운다 (--hil이면 기본 켜짐). "
                        "터미널을 못 보는 상태로 HIL을 하면 인계 순간을 놓친다")
    p.add_argument("--no-panel", dest="panel", action="store_false",
                   help="상태 창을 끈다")
    p.add_argument("--no-status", action="store_true",
                   help="실시간 상태 블록을 끄고 러너 로그를 그대로 흘린다 (디버깅용)")
    p.add_argument("--aggregate-fn", default=None,
                   choices=["weighted_average", "latest_only", "temporal_ensemble"],
                   help="chunk 합치는 방식. 기본은 러너 프리셋 값 "
                        "(weighted_average). 게이트/HIL과의 상호작용은 "
                        "docs/erase_run_design.md §5.5 참고")
    p.add_argument("--confirm", action="store_true", help="실물 로봇에 명령 전송을 허용")
    args = p.parse_args()

    if args.probe_leader:
        robot = build_robot(args.follower_port, args.top_cam, args.wrist_cam, args.cam_type)
        leader = build_leader(args.leader_port)
        probe_leader(leader, robot)
        return 0

    for req in ("policy_path", "dataset_root"):
        if not getattr(args, req):
            p.error(f"--{req} 필요")
    if not args.confirm:
        print("[DRY] --confirm 없음 — 로봇에 명령을 보내지 않고 종료한다.")
        print("      실행 전 확인: park 자세가 카메라 시야를 안 가리는지, CAN이 비어있는지.")
        return 0

    import erase_check as EC

    from piper_infer_runner import (
        DEFAULT_ENV_FILE,
        REAL_ROBOT_CONFIRM,
        RunSettings,
        load_env_file,
        mode_preset,
        resolve_crops,
    )

    # build_robot_from_env는 recording.env 파일이 아니라 os.environ을 읽는다.
    # 평소에는 scripts/9__run_client.sh 같은 셸이 env를 source해서 넘겨주는데,
    # 이 CLI를 직접 실행하면 TOP_CAM이 비어서 카메라가 아예 안 붙고, 루프 첫
    # 스텝에서 "Live observation is missing camera 'top'"으로 죽는다.
    # 이미 셸에서 넘어온 값이 있으면 그쪽을 존중한다.
    env = load_env_file(DEFAULT_ENV_FILE)
    for key, value in env.items():
        os.environ.setdefault(key, value)

    # 추론·스무딩·안전은 러너가 맡는다. 여기서 정하는 건 "시도 하나가 어떤
    # 조건으로 도는가"뿐이다.
    base = dict(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        policy_path=args.policy_path,
        task=args.task,
        device=args.device,
        source="robot",
        apply_to_robot=True,
        real_robot_confirm=REAL_ROBOT_CONFIRM,
        fps=float(args.fps),
        max_steps=args.max_steps,
        stop_on_release=args.stop_on_release,
        rviz=False,
        park_on_exit=True,          # 시도 끝 park는 러너가 강제한다
        crops=resolve_crops({}, args.top_crop, args.wrist_crop or None),
        camera_output_size=args.camera_output_size,
        hil=args.hil,
        leader_port=args.leader_port,
        clutch_gain=args.clutch_gain,
        max_relative_target=args.max_relative_target,
        move_speed_rate=args.move_speed_rate,
        record_manual=args.record_manual,
        record_on_intervention=not args.no_record_on_intervention,
    )
    if args.hil:
        print("[HIL] space = 개입 on/off,  q = 시도 중단")

    import erase_status as ES

    # aggregate_fn은 RunSettings 최상위가 아니라 smoothing 안에 있다. 모드 프리셋이
    # 정해준 SmoothingConfig를 그대로 두고 이 필드만 갈아끼운다 — ema_alpha 0.2 같은
    # 실측 최적값을 덮어쓰면 안 된다(docs/policy/smoothing.md).
    if args.aggregate_fn:
        import dataclasses as _dc

        # mode_preset()으로 프리셋만 꺼낸다. RunSettings.from_mode(mode)를 쓰면
        # dataset_root/policy_path가 필수 인자라 그 자리에서 TypeError로 죽는다.
        preset = mode_preset(args.mode)
        base["smoothing"] = _dc.replace(preset.smoothing, aggregate_fn=args.aggregate_fn)
        print(f"[INFO] aggregate_fn = {args.aggregate_fn} "
              f"(나머지 스무딩은 '{args.mode}' 프리셋 그대로: "
              f"{preset.smoothing.summary()})")

    check_dataset_matches_checkpoint(args.policy_path, base["dataset_root"])

    checker = EC.EraseChecker()
    attempts = []

    # 이번 실행의 산출물 폴더. 시각으로 이름 지어 실행마다 하나씩 쌓인다.
    run_dir = Path(args.hil_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_path": args.policy_path,
        "dataset_root": str(args.dataset_root),
        "task": args.task,
        "target": args.target,
        "mode": args.mode,
        "hil": bool(args.hil),
        "max_attempts": args.max_attempts,
        "max_steps": args.max_steps,
        "stop_on_release": args.stop_on_release,
        "aggregate_fn": args.aggregate_fn,
        "top_crop": args.top_crop,
        "wrist_crop": args.wrist_crop,
        "max_relative_target": args.max_relative_target,
    }, ensure_ascii=False, indent=2))
    if args.record_manual:
        # 데이터셋도 이 실행 폴더 안에 둔다 — 로그·판정 프레임과 같이 있어야
        # 나중에 "이 개입이 어떤 판정에서 나온 것인지"를 이어붙일 수 있다.
        stamp = run_dir.name
        base["record_root"] = Path(
            args.record_root or (run_dir / "dataset")
        ).expanduser().resolve()
        base["record_repo_id"] = args.record_repo_id or f"local/hil_{stamp}"
        # ★ raw(1280x720) 대신 정책 입력과 같은 크롭본(512x512)을 담는다.
        # 이유 두 가지:
        #  - 비용. raw는 카메라 2대 × 1280×720×3 ≈ 5.5MB/프레임이라 30Hz를 못 지킨다.
        #    실측에서 제어 주기가 30 -> 19.53Hz로 떨어졌고, 데이터셋 meta에는 fps가
        #    30으로 적혀 학습 때 시간축이 어긋난다.
        #  - 일관성. 기존 학습 데이터셋이 전부 512x512 크롭본이다. 같은 형식이어야
        #    바로 이어붙여 재학습할 수 있다.
        base["record_raw_frames"] = args.record_raw_frames
        print(f"[INFO] 수동 녹화 켜짐 — 데이터셋: {base['record_root']}")
        if args.no_record_on_intervention:
            print("       패널의 '● 녹화 시작' / '■ 녹화 종료'로 구간을 담습니다.")
        else:
            print("       space로 개입하면 자동으로 녹화가 시작되고, 반환하면 "
                  "그 구간이 에피소드로 저장됩니다.")
            print("       (버튼으로 직접 제어하려면 --no-record-on-intervention)")
    print(f"[INFO] 산출물 폴더: {run_dir}")

    def save_frame(name, frame):
        if args.no_save_frames:
            return
        import cv2

        cv2.imwrite(str(run_dir / f"{name}.png"), frame)
    # 진행 상황 표시. 여기서 보여주는 퍼센트가 두 종류라는 게 중요하다 —
    # [진행]은 시간축, [측정]은 park에서 실제로 잰 잉크 값이다. 자세한 근거는
    # erase_status 모듈 docstring (시도 도중에는 잉크를 잴 방법이 없다).
    status = None if args.no_status else ES.LiveStatus(
        max_attempts=args.max_attempts, max_steps=args.max_steps, hil=args.hil
    )
    # 화면 패널. HIL에서는 기본으로 켠다 — 인계 순간 팔이 튀는지 눈으로 봐야 하는데
    # 터미널을 안 보고 리더암 앞에 서 있으면 터미널 표시가 사람에게 도달하지 않는다.
    use_panel = args.hil if args.panel is None else args.panel
    panel = None
    if use_panel:
        from erase_hil_panel import HilPanel

        panel = HilPanel(args.max_attempts, args.max_steps).start()

    def grab():
        if status:
            status.set_phase("판정용 프레임 촬영 (park)")
        if panel:
            panel.set_phase("판정용 프레임 촬영 (park)")
        return grab_judge_frame(args.top_cam)

    # 러너 참조를 붙잡아 둔다. 패널 버튼(녹화·개입)이 이걸 통해 러너를 제어하고,
    # 끝나고 녹화 결과를 요약할 때도 필요하다.
    runner_ref = {}

    def on_runner(run):
        runner_ref["run"] = run
        if panel:
            panel.attach_runner(run)

    def on_step(payload):
        if status:
            status.on_step(payload)
        if panel:
            panel.on_step(payload)

    try:
        # ★ 기준은 시도 1 이전에 딱 한 번만 잡는다. 지운 마카는 되돌릴 수 없어서
        #   재시도는 항상 이전 시도 위에 쌓이고, erased_frac은 누적값이어야 맞다.
        ref_frame = grab()
        save_frame("00_reference", ref_frame)
        ref = checker.set_reference(ref_frame)
        print(f"[INFO] 기준 프레임 — 검출된 도형: {[k for k, _ in ref['shapes']]}")
        if not any(k.startswith(args.target) for k, _ in ref["shapes"]):
            print(f"[WARN] target '{args.target}'을 기준 프레임에서 못 찾았다 — "
                  f"판정이 무의미하다. 보드/조명 확인 후 다시 실행할 것.")
            return 2

        for i in range(1, args.max_attempts + 1):
            print(f"[INFO] ── 시도 {i}/{args.max_attempts} ──")
            if status:
                status.start_attempt(i)
            if panel:
                panel.start_attempt(i)
            summary = run_attempt_with_runner(
                RunSettings.from_mode(args.mode, **base),
                # 러너 로그를 다 찍으면 상태 블록이 흐트러진다. 그렇다고 전부
                # 삼키면 안 된다 — 처음엔 전부 삼켰는데, 그 바람에 녹화가 안 된
                # 실행에서 '[RECORD] 녹화 시작'이 찍혔는지조차 알 수 없었다.
                # 사람이 놓치면 안 되는 것만 통과시킨다.
                on_log=_filtered_log if status else None,
                on_step=on_step,
                # 패널 버튼이 누를 개입 토글은 러너가 만든다. 러너가 뜬 직후에
                # 건네받아야 첫 개입부터 버튼이 먹는다.
                on_runner=on_runner,
            )
            if status:
                status.set_phase("판정 중")
            if panel:
                panel.set_phase("판정 중")
                panel.set_fps(summary["measured_fps"],
                              intervention_steps=summary["interventions"])
            after_frame = grab()
            save_frame(f"{i:02d}_after", after_frame)
            r = checker.check(after_frame, args.target)
            r["attempt"] = i
            r.update(summary)
            attempts.append(r)
            if status:
                status.set_measurement(r)
                status.finish()
            if panel:
                panel.set_measurement(r)
            print(f"  → success={r['success']} target_erased={r.get('target_erased')} "
                  f"남음={r.get('remaining_frac')} "
                  f"max_distractor={r.get('max_distractor_erased')} "
                  f"steps={summary['steps']} interventions={summary['interventions']} "
                  f"fps={summary['measured_fps']}")
            # 실패했으면 "어디가" 남았는지를 사람이 보고 다음 판단을 한다.
            # 이 정보를 정책에 넣을 통로는 아직 없다 (설계 문서 §4.6-③).
            if args.record_manual:
                run = runner_ref.get("run")
                n = int(getattr(run, "recorded_episodes", 0)) if run else 0
                if n:
                    fps = summary.get("measured_fps", 0.0)
                    note = ""
                    if fps and abs(fps - args.fps) / args.fps > 0.1:
                        # 데이터셋 meta에는 설정 fps가 적히므로, 실측이 크게 다르면
                        # 그대로 학습에 쓸 때 시간축이 어긋난다.
                        note = (f"  ★ 실측 {fps:.1f}Hz vs 설정 {args.fps}Hz — "
                                f"학습 전 확인 필요")
                    print(f"  녹화: 에피소드 {n}개 저장 -> {base['record_root']}{note}")
                else:
                    # 조용히 넘어가면 나중에야 빈 데이터셋을 발견하게 된다.
                    print("  녹화: ★ 저장된 에피소드가 없다 — 패널의 '● 녹화 시작'을 "
                          "누르고 '■ 녹화 종료'로 끊어야 담긴다")
            if r.get("residual"):
                print(f"  잔여: {ES.format_residual(r['residual'])}")
                for line in ES.format_grid(r["residual"]):
                    print(line)
            if summary["status"] not in ("finished", "hil_abort"):
                print(f"[STOP] 러너가 {summary['status']}로 끝났다 — 재시도하지 않는다.")
                break
            if r["success"]:
                break
    finally:
        if panel:
            panel.close()
        # 산출물은 run_dir에 모으고, --out 경로에도 그대로 남긴다(기존 호출자 호환).
        traces = save_step_traces(attempts, run_dir / "log.json")
        payload = json.dumps(attempts, ensure_ascii=False, indent=2)
        (run_dir / "log.json").write_text(payload)
        Path(args.out).write_text(payload)
        print(f"[INFO] 로그 저장: {run_dir / 'log.json'}"
              + (f" / {traces.name}" if traces else "")
              + f"  (사본: {args.out})")

    ok = bool(attempts) and attempts[-1]["success"]
    total = sum(a["steps"] for a in attempts)
    intervened = sum(a["interventions"] for a in attempts)
    print(f"[결과] success={ok}  attempts={len(attempts)}  total_steps={total}  "
          f"intervention_rate={100*intervened/total if total else 0:.1f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

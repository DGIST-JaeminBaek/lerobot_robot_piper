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
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# erase_check(cv2 의존)와 lerobot은 main()에서 늦게 import한다 —
# 아래 중재/루프 로직은 순수 파이썬이라 하드웨어·cv2 없이 테스트된다.

JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER = "gripper"
ACTION_NAMES = [f"{n}.pos" for n in JOINTS + [GRIPPER]]


# ══════════════════════════════════════════════════════════════
# HIL 개입 — 키보드 토글 + 델타(클러치) 인계
#
# 왜 키보드인가: 리더암 움직임으로 개입을 자동 감지하려면 임계값 3개를 실측으로
#   맞춰야 하고, 진입/이탈 각각에 실패 모드가 있었다(git 히스토리 참고).
#   명시적 토글은 그 전부를 없앤다. lerobot HIL-SERL도 space 토글을 쓴다.
#
# 왜 델타인가 (★ PiPER 고유 제약):
#   PiperLeader는 **명령을 받을 수 없다**. piper_leader.py connect() 주석:
#   "Master arm: only sends control frame messages" — EnableArm 대상이 아니고
#   서보 활성화를 시도하면 타임아웃난다(실제 하드웨어에서 확인됨).
#   따라서 SO101처럼 "정책 실행 중 리더가 팔로워를 따라가게 두는" 방식이 불가능하다.
#   개입 시점에 리더와 팔로워는 반드시 어긋나 있다.
#   → 리더의 **절대 자세**를 보내면 팔로워가 튄다.
#   → 리더의 **변화량**만 팔로워의 현재 자세에 더한다(클러치/인덱싱).
#     인계 순간 목표 = 팔로워 현재 자세이므로 점프가 원리적으로 0이다.
# ══════════════════════════════════════════════════════════════
class Clutch:
    """개입 중 리더 변화량을 팔로워에 더해주는 델타 인계기.

    engage() 시점의 리더/팔로워 자세를 기준점으로 잡고,
        target = follower_at_engage + (leader_now - leader_at_engage)
    를 보낸다. 인계 첫 스텝의 target이 팔로워 현재 자세와 같으므로 점프가 없다.
    """

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self._lead0 = None
        self._foll0 = None

    @property
    def engaged(self) -> bool:
        return self._lead0 is not None

    def engage(self, leader_action: dict, follower_obs: dict) -> None:
        self._lead0 = {k: leader_action[k] for k in ACTION_NAMES if k in leader_action}
        self._foll0 = {k: follower_obs[k] for k in ACTION_NAMES if k in follower_obs}

    def release(self) -> None:
        self._lead0 = self._foll0 = None

    def target(self, leader_action: dict) -> dict:
        if not self.engaged:
            raise RuntimeError("engage()를 먼저 호출할 것")
        return {
            k: self._foll0[k] + self.gain * (leader_action[k] - self._lead0[k])
            for k in self._foll0
            if k in leader_action
        }


def deviation(leader_action: dict, follower_obs: dict) -> float:
    """리더-팔로워 최대 관절 편차. 판정에는 안 쓰고 로깅/경고용으로만 남긴다."""
    devs = [
        abs(leader_action[f"{j}.pos"] - follower_obs[f"{j}.pos"])
        for j in JOINTS
        if f"{j}.pos" in leader_action and f"{j}.pos" in follower_obs
    ]
    return max(devs) if devs else 0.0


class KeyToggle:
    """space = 개입 on/off, q = 시도 중단. pynput 전역 리스너(teleop_ui.py와 같은 의존성).

    테스트에서는 이 클래스 대신 아무 객체나 넣으면 된다 — .active / .abort 두 속성만 본다.
    """

    def __init__(self):
        self.active = False
        self.abort = False
        self._listener = None

    def start(self):
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.space:
                self.active = not self.active
                print(f"[HIL] {'개입 ON — 리더암 조작' if self.active else '개입 OFF — 정책 반환'}")
            elif getattr(key, "char", None) == "q":
                self.abort = True
                print("[HIL] 시도 중단 요청")

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        return self

    def stop(self):
        if self._listener:
            self._listener.stop()


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
    log = []
    for step in range(max_steps):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        action = policy_fn(obs)
        intervened = False
        dev = None

        if hil:
            lead = leader.get_action()
            if toggle.active:
                if not clutch.engaged:
                    clutch.engage(lead, obs)          # 기준점 = 지금 자세 → 점프 0
                    dev = deviation(lead, obs)        # 참고용 기록
                action = clutch.target(lead)
                intervened = True
            elif clutch.engaged:
                clutch.release()

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

    if hil:
        clutch.release()
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


def build_policy_fn(policy_path, dataset_root, task, device):
    """관측 dict -> action dict 함수를 만든다.

    호출 경로는 piper_infer_preview.py와 동일(= lerobot_record.py의 record_loop).
    robot.get_observation()이 이미 build_dataset_frame이 기대하는 raw 형식
    ({"joint1.pos": float, ..., "top": HWC uint8})이라 변환이 필요 없다.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import OBS_STR, build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    root = Path(dataset_root)
    ds_meta = LeRobotDatasetMetadata(repo_id=root.name, root=root)

    from piper_infer_preview import load_policy  # 같은 디렉터리

    cfg, policy, pre, post = load_policy(policy_path, ds_meta, device)
    torch_device = get_safe_torch_device(policy.config.device)
    features = ds_meta.features

    def policy_fn(obs: dict) -> dict:
        frame = build_dataset_frame(features, obs, prefix=OBS_STR)
        values = predict_action(
            observation=frame, policy=policy, device=torch_device,
            preprocessor=pre, postprocessor=post,
            use_amp=getattr(policy.config, "use_amp", False),
            task=task, robot_type="piper_follower",
        )
        return make_robot_action(values, features)

    return policy_fn


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
    p.add_argument("--task", default="erase the triangle")
    p.add_argument("--target", default="triangle", choices=["circle", "triangle", "rectangle"])
    p.add_argument("--follower-port", default="can_follower1")
    p.add_argument("--leader-port", default="can_leader1")
    p.add_argument("--top-cam", default="")
    p.add_argument("--wrist-cam", default="")
    p.add_argument("--cam-type", default="realsense")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=940, help="시도당 상한 (중앙값 720의 약 1.3배)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--park-pose", default="", help='예: "joint1.pos=0,joint2.pos=0,..." (미지정 시 robot.parking())')
    p.add_argument("--hil", action="store_true", help="리더암 개입 활성화")
    p.add_argument("--clutch-gain", type=float, default=1.0, help="리더 변화량 -> 팔로워 반영 비율 (1.0=등배)")
    p.add_argument("--probe-leader", action="store_true", help="리더암 노이즈 플로어만 측정하고 종료")
    p.add_argument("--out", default="erase_run_log.json")
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

    park_pose = None
    if args.park_pose:
        park_pose = {k: float(v) for k, v in (kv.split("=") for kv in args.park_pose.split(","))}

    import erase_check as EC

    robot = build_robot(args.follower_port, args.top_cam, args.wrist_cam, args.cam_type)
    leader = build_leader(args.leader_port) if args.hil else None
    clutch = Clutch(args.clutch_gain) if args.hil else None
    toggle = KeyToggle().start() if args.hil else None
    if args.hil:
        print("[HIL] space = 개입 on/off,  q = 시도 중단")
    policy_fn = build_policy_fn(args.policy_path, args.dataset_root, args.task, args.device)
    park_fn = make_park_fn(robot, park_pose)

    checker = EC.EraseChecker()
    attempts = []

    def grab():
        return grab_park_frame(robot, park_fn)

    try:
        checker.set_reference(grab())
        for i in range(1, args.max_attempts + 1):
            print(f"[INFO] 시도 {i}/{args.max_attempts} 시작")
            steps = run_attempt(robot, policy_fn, park_fn, args.fps, args.max_steps,
                                toggle, leader, clutch)
            r = checker.check(grab(), args.target)
            r["attempt"] = i
            r["steps"] = len(steps)
            r["interventions"] = sum(s["intervention"] for s in steps)
            attempts.append({"result": r, "log": steps})
            print(f"  → success={r['success']} target_erased={r.get('target_erased')} "
                  f"max_distractor={r.get('max_distractor_erased')} interventions={r['interventions']}")
            if r["success"]:
                break
    finally:
        if toggle is not None:
            toggle.stop()
        park_fn()
        Path(args.out).write_text(json.dumps(attempts, ensure_ascii=False, indent=2))
        print(f"[INFO] 로그 저장: {args.out}")

    ok = bool(attempts) and attempts[-1]["result"]["success"]
    total = sum(a["result"]["steps"] for a in attempts)
    print(f"[결과] success={ok}  attempts={len(attempts)}  total_steps={total}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

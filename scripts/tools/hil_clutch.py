#!/usr/bin/env python3
"""hil_clutch.py — 리더암 HIL 개입의 순수 로직 (키보드 토글 + 델타 인계).

erase_run.py에 있던 것을 여기로 뺐다. 게이트(시도 단위)와 클러치(스텝 단위)는
층위가 달라서, 클러치는 제어 루프 안에 들어가야 한다 — piper_infer_runner.py의
InferenceRunner와 erase_run.py가 **같은 구현**을 쓰도록 공용 모듈로 둔다.

하드웨어·cv2·lerobot 의존이 없다. pynput은 KeyToggle.start()에서만 늦게 import한다.

배경은 docs/erase_run_design.md §5. 요약:

왜 키보드인가
  리더암 움직임으로 개입을 자동 감지하려면 임계값 3개를 실측으로 맞춰야 하고,
  진입/이탈 각각에 실패 모드가 있었다 — 이탈 판정에 편차를 쓰면 개입 중 편차가
  0으로 수렴해 즉시 반환되고, 진입 판정에 편차만 쓰면 손 뗀 뒤 무한 재개입한다.
  명시적 토글은 그 전부를 없앤다. lerobot HIL-SERL도 space 토글을 쓴다.

왜 델타인가 (★ PiPER 고유 제약)
  PiperLeader는 **명령을 받을 수 없다**. piper_leader.py connect() 주석:
  "Master arm: only sends control frame messages" — EnableArm 대상이 아니고
  서보 활성화를 시도하면 타임아웃난다(실제 하드웨어에서 확인됨).
  따라서 SO101처럼 "정책 실행 중 리더가 팔로워를 따라가게 두는" 방식이 불가능하고,
  개입 시점에 리더와 팔로워는 반드시 어긋나 있다.
    → 리더의 **절대 자세**를 보내면 팔로워가 튄다.
    → 리더의 **변화량**만 팔로워의 현재 자세에 더한다(클러치/인덱싱).
      인계 순간 목표 = 팔로워 현재 자세이므로 점프가 원리적으로 0이다.
"""

from __future__ import annotations

JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER = "gripper"
ACTION_NAMES = [f"{n}.pos" for n in JOINTS + [GRIPPER]]
GRIPPER_KEY = f"{GRIPPER}.pos"   # action dict의 실제 키. GRIPPER 자체가 아니다.


class Clutch:
    """개입 중 리더 변화량을 팔로워에 더해주는 델타 인계기.

    engage() 시점의 리더/팔로워 자세를 기준점으로 잡고,
        target = follower_at_engage + gain * (leader_now - leader_at_engage)
    를 보낸다. 인계 첫 스텝의 target이 팔로워 현재 자세와 같으므로 점프가 없다.
    """

    def __init__(self, gain: float = 1.0, absolute_keys: tuple[str, ...] = (GRIPPER_KEY,)):
        """absolute_keys에 든 축은 델타가 아니라 **리더 절대값**을 그대로 보낸다.

        기본값이 그리퍼인 이유 — 델타 규칙을 그리퍼에 걸면 끝까지 닫는 게
        원리적으로 불가능하다. 인계 시점 팔로워 30 / 리더 60이면 리더를 100까지
        닫아도 목표는 30+(100-60)=70이다. 실물에서 "리더 그리퍼를 끝까지 닫아도
        팔로워가 안 닫혀 물건을 잡을 수조차 없다"로 나타났다.

        델타 규칙의 존재 이유는 '리더와 팔로워가 크게 어긋난 상태에서 절대값을
        보내면 팔이 튄다'인데, 그리퍼는 1자유도에 가동범위가 막혀 있어 그 위험이
        관절과 다르다. 게다가 사람의 의도 자체가 절대값이다("닫아라").
        인계 순간 그리퍼가 급히 닫히는 건 팔로워의 max_relative_target 클램프가
        스텝당 변화량을 잘라 완만하게 만든다.
        """
        self.gain = gain
        self.absolute_keys = tuple(absolute_keys)
        self._lead0: dict | None = None
        self._foll0: dict | None = None

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
        out = {}
        for k in self._foll0:
            if k not in leader_action:
                continue
            if k in self.absolute_keys:
                out[k] = leader_action[k]          # 절대값 통과 (그리퍼)
            else:
                out[k] = self._foll0[k] + self.gain * (leader_action[k] - self._lead0[k])
        return out


def deviation_per_joint(leader_action: dict, follower_obs: dict) -> dict:
    """관절별 리더-팔로워 편차. 최댓값만 남기면 '자세를 잘못 맞춘 것'과
    '두 팔의 읽기값 대응이 어긋난 것'을 구분할 수 없다.

    전자는 매번 다른 관절에서 크게 나오고, 후자는 **항상 같은 관절에서 비슷한
    값**으로 나온다. 실물 첫 HIL에서 joint5가 리더 0.00 vs 팔로워 100.00으로
    나온 게 후자로 의심되는 상황이라, 재현되는지 보려면 관절별로 남겨야 한다.
    """
    return {
        j: round(float(leader_action[f"{j}.pos"] - follower_obs[f"{j}.pos"]), 2)
        for j in JOINTS
        if f"{j}.pos" in leader_action and f"{j}.pos" in follower_obs
    }


def deviation(leader_action: dict, follower_obs: dict) -> float:
    """리더-팔로워 최대 관절 편차. 판정에는 안 쓰고 로깅/경고용으로만 남긴다.

    클러치 방식이라 이 값이 커도 인계는 안전하다. 인계 시점 값을
    engage_deviation으로 남겨두면 나중에 자동 감지를 다시 시도할 때 실측 근거가 된다.
    """
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


class ClutchMixer:
    """제어 루프 안에서 쓰는 얇은 래퍼 — 매 스텝 '정책 action'을 받아 최종 action을 낸다.

    InferenceRunner처럼 numpy 벡터로 도는 루프와 erase_run처럼 dict로 도는 루프가
    같은 규칙을 쓰게 하려고 dict 경계만 담당한다. 상태 전이(engage/release)를
    한 곳에 모아두어 두 호출자가 순서를 각자 구현하다 어긋나는 걸 막는다.
    """

    def __init__(self, toggle, leader, clutch: Clutch):
        self.toggle = toggle
        self.leader = leader
        self.clutch = clutch
        self.engage_deviation: float | None = None
        self.intervened_steps = 0

    @property
    def active(self) -> bool:
        return bool(self.toggle is not None and self.toggle.active)

    def step(self, policy_action: dict, follower_obs: dict) -> tuple[dict, bool, float | None]:
        """반환: (최종 action, 개입 여부, 이번 스텝에 새로 인계했다면 그 편차)"""
        if self.toggle is None or self.leader is None:
            return policy_action, False, None

        if not self.toggle.active:
            if self.clutch.engaged:
                self.clutch.release()
            return policy_action, False, None

        lead = self.leader.get_action()
        new_engage_dev = None
        if not self.clutch.engaged:
            self.clutch.engage(lead, follower_obs)  # 기준점 = 지금 자세 → 점프 0
            new_engage_dev = deviation(lead, follower_obs)
            self.engage_deviation = new_engage_dev
        self.intervened_steps += 1
        return self.clutch.target(lead), True, new_engage_dev

    def release(self) -> None:
        self.clutch.release()

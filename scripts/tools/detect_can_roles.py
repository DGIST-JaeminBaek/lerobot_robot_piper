#!/usr/bin/env python3
"""UP 상태인 CAN 인터페이스별로 leader/follower 역할을 판별해서 출력.

재부팅하면 CAN 이름이 can0/can1로 돌아가는데(udev 규칙이 없음), 어느 쪽이 leader이고
어느 쪽이 follower인지는 USB 포트 순서로 추측하면 틀릴 수 있다 — 그래서 실제로
ctrl_mode를 읽어서 판별한다(teleop_ui.detect_can_role과 같은 방식:
0x06=Linkage teaching input mode면 leader).

sudo가 필요 없다. 인터페이스가 이미 UP이고 bitrate가 맞춰져 있어야 한다.

출력(한 줄에 하나): "can0 follower" / "can1 leader" / "can2 unknown"

사용 예:
    python3 scripts/tools/detect_can_roles.py can0 can1
"""

import sys
import time

from piper_sdk import C_PiperInterface_V2


def detect(iface: str, tries: int = 10, wait_s: float = 0.2) -> str:
    try:
        piper = C_PiperInterface_V2(iface, judge_flag=False, can_auto_init=False)
        piper.CreateCanBus(iface)
        piper.ConnectPort(piper_init=False, start_thread=True)
        try:
            for _ in range(tries):
                time.sleep(wait_s)
                ctrl_mode = piper.GetArmStatus().arm_status.ctrl_mode
                mode = ctrl_mode.value if hasattr(ctrl_mode, "value") else int(ctrl_mode)
                if mode != 0:
                    return "leader" if mode == 0x06 else "follower"
            return "unknown"
        finally:
            piper.DisconnectPort()
    except Exception as exc:  # 판별 실패는 치명적이지 않음 — 사용자가 직접 확인
        print(f"# {iface}: {exc}", file=sys.stderr)
        return "unknown"


def main() -> None:
    interfaces = sys.argv[1:] or ["can0", "can1"]
    for iface in interfaces:
        print(f"{iface} {detect(iface)}")


if __name__ == "__main__":
    main()

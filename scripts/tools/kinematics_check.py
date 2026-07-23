#!/usr/bin/env python3
"""
kinematics_check.py — piper_sdk의 FK 계산이 팔 펌웨어의 EEF 피드백과 실제로
일치하는지 실측으로 검증하는 도구.

배경: piper_sdk.GetFK(mode="feedback")는 GetArmJointMsgs()(joint 피드백)를
CalFK()(DH 파라미터 기반 순수 계산)에 넣은 결과일 뿐임(소스로 확인됨).
반면 GetArmEndPoseMsgs()는 팔 펌웨어가 CAN으로 직접 보고하는 EEF 값 —
시리얼 로봇팔은 EEF 위치 센서가 없어 펌웨어도 내부적으로 joint encoder ->
FK 계산을 할 수밖에 없지만, 그 내부 구현(DH 파라미터/보정값)이 piper_sdk의
CalFK와 정확히 같은지는 소스로 확인 불가 — 실측으로만 검증 가능함.

이 스크립트는 팔을 움직이지 않음(연결하고 읽기만 함, EnableFkCal()도 계산만
켤 뿐 명령 전송이 아님) — 아무 때나 안전하게 돌려도 됨.

사용법:
    python scripts/tools/kinematics_check.py --port can_follower1
    python scripts/tools/kinematics_check.py --port can_follower1 --samples 5 --interval 1.0

옵션:
    --port      follower CAN 인터페이스 (기본 can_follower1)
    --samples   비교 샘플 개수 (기본 1) — 여러 자세에서 반복 확인하려면 그 사이에
                leader를 손으로 움직이면서 늘리면 됨
    --interval  샘플 사이 대기 시간(초), 기본 2.0

판정 기준: 위치(x/y/z) 차이 1mm 이하, 각도(rx/ry/rz) 차이 1도 이하면 SDK 계산
(CalFK/GetFK)과 펌웨어 피드백이 같은 모델을 쓴다는 뜻 — EEF action/state를
CalFK()로 계산해도 신뢰 가능. 차이가 크면 펌웨어에 우리가 모르는 보정이 있다는
신호라, 그 경우 실측값(GetArmEndPoseMsgs)을 우선해야 함.
"""

import argparse
import time

from piper_sdk import C_PiperInterface_V2


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"


def ok(m):   print(f"{C.GREEN}[OK]{C.RESET} {m}")
def warn(m): print(f"{C.YELLOW}[WARN]{C.RESET} {m}")
def info(m): print(f"{C.CYAN}[INFO]{C.RESET} {m}")


LABELS = ["x(mm)", "y(mm)", "z(mm)", "rx(deg)", "ry(deg)", "rz(deg)"]
POS_TOL_MM = 1.0
ANGLE_TOL_DEG = 1.0


def compare_once(piper: C_PiperInterface_V2) -> tuple[list[float], list[float]]:
    sdk_fk = piper.GetFK(mode="feedback")[-1]  # link6 = EEF, [x,y,z,rx,ry,rz] (mm, deg)

    hw = piper.GetArmEndPoseMsgs().end_pose
    hw_pose = [
        hw.X_axis / 1000.0, hw.Y_axis / 1000.0, hw.Z_axis / 1000.0,
        hw.RX_axis / 1000.0, hw.RY_axis / 1000.0, hw.RZ_axis / 1000.0,
    ]
    return sdk_fk, hw_pose


def print_comparison(sdk_fk: list[float], hw_pose: list[float]) -> bool:
    print(f"\n{'축':>8}  {'SDK CalFK':>12}  {'펌웨어 피드백':>14}  {'차이':>10}")
    all_ok = True
    for i, (label, a, b) in enumerate(zip(LABELS, sdk_fk, hw_pose)):
        diff = abs(a - b)
        tol = ANGLE_TOL_DEG if "deg" in label else POS_TOL_MM
        status = f"{C.GREEN}OK{C.RESET}" if diff <= tol else f"{C.RED}FAIL{C.RESET}"
        if diff > tol:
            all_ok = False
        print(f"{label:>8}  {a:>12.3f}  {b:>14.3f}  {diff:>10.3f}  {status}")
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="piper_sdk FK 계산 vs 팔 펌웨어 EEF 피드백 실측 대조")
    p.add_argument("--port", default="can_follower1", help="follower CAN 인터페이스")
    p.add_argument("--samples", type=int, default=1, help="비교 샘플 개수")
    p.add_argument("--interval", type=float, default=2.0, help="샘플 사이 대기 시간(초)")
    args = p.parse_args()

    piper = C_PiperInterface_V2(args.port)
    piper.ConnectPort()
    time.sleep(0.5)

    fw_version = piper.GetPiperFirmwareVersion()
    info(f"펌웨어 버전: {fw_version}")

    piper.EnableFkCal()
    time.sleep(1.0)  # 내부 FK 계산 스레드가 몇 프레임 돌 시간 확보

    all_samples_ok = True
    for i in range(args.samples):
        if args.samples > 1:
            info(f"샘플 {i + 1}/{args.samples}")
        sdk_fk, hw_pose = compare_once(piper)
        sample_ok = print_comparison(sdk_fk, hw_pose)
        all_samples_ok = all_samples_ok and sample_ok
        if i < args.samples - 1:
            info(f"{args.interval}초 뒤 다음 샘플 — 필요하면 leader를 움직여 자세를 바꾸세요")
            time.sleep(args.interval)

    print()
    if all_samples_ok:
        ok("전체 샘플에서 SDK 계산과 펌웨어 피드백 일치 (위치 1mm/각도 1도 이내) — CalFK()/GetFK() 신뢰 가능")
    else:
        warn("일부 축에서 오차가 허용 범위를 벗어남 — 펌웨어에 추가 보정이 있을 수 있음, EEF state는 실측값(GetArmEndPoseMsgs) 우선 권장")

    piper.DisconnectPort()


if __name__ == "__main__":
    main()

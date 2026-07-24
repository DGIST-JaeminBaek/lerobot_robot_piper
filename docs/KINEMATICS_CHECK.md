# FK/IK 검증 — 완료 기록과 재확인 방법

## 검증 결과

`scripts/tools/kinematics_check.py`로 팔 펌웨어가 CAN으로 직접 보고하는 EEF 피드백(`GetArmEndPoseMsgs()`)과 SDK 계산(`CalFK`/`GetFK`)을 실물 로봇에서 비교했고, 검증을 완료했다.

따라서 현재 컨벤션에서는 `CalFK()`/`GetFK()` 계산값을 EEF state/action 계산에 사용해도 된다. 자세한 오차 수치는 별도 실험 로그가 있으면 이 문서에 추가한다.

## 배경 (소프트웨어만으로 먼저 확인했던 것)

RViz/EEF 관련 논의에서 확인된 것들 — 전부 **하드웨어 없이** 소스/문서 대조로 검증됨:

1. `piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics.CalFK()` — 순수 DH 파라미터 기반 FK 계산(하드웨어 불필요)
2. `dh_is_offset=0x01`(SDK 기본값)이 Agilex 공식 User Manual의 DH 파라미터 표와 정확히 일치함
3. 신형 펌웨어(S-V1.6-3 이상, J2/J3 좌표계가 구형 대비 2° 이동)가 이 `0x01` 컨벤션에 해당함 — 우리 팔 펌웨어는 `S-V1.8-2`로 확인된 적 있음(신형)
4. RViz가 쓰는 URDF(`agx_arm_urdf`의 `piper_description.urdf`)로 `ikpy` FK를 돌린 결과가 `piper_sdk`의 `CalFK`와 zero-config 기준 0.1mm 이하 오차로 일치함
5. `GetFK(mode="feedback"/"control")`은 각각 `GetArmJointMsgs`/`GetArmJointCtrl` 값을 받아 `CalFK()`를 호출하는 것뿐임(소스로 확인)

이후 팔 펌웨어가 CAN으로 직접 보고하는 EEF 피드백(`GetArmEndPoseMsgs()`)도 실측으로 확인했다.

## 재확인 방법

**GUI(`teleop_ui.py`/`piper-teleop`)는 안 켜도 됨** — 이 스크립트는 `piper_sdk`로 CAN 포트에 직접 연결하는 독립 스크립트라, GUI/`lerobot-record`/`lerobot-teleoperate` 등 다른 프로세스가 안 떠 있어도 그대로 실행 가능함.

**준비물**:
- Follower 팔 전원 켜기
- CAN 인터페이스가 bring-up 상태일 것(`can_follower1` 등) — `scripts/1__init_can.sh` 또는 GUI의 CAN Setup 패널로 미리 이름/bitrate만 잡아두면 됨(그 뒤 GUI는 꺼도 무방). `configs/recording.env`의 `FOLLOWER_PORT` 값 확인
- `ugrp` conda 환경

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp

# 기본 실행 (1회 측정)
python3 scripts/tools/kinematics_check.py --port can_follower1

# 여러 자세에서 반복 확인 (더 신뢰도 높음 — 그 사이 leader를 손으로 움직이면 됨)
python3 scripts/tools/kinematics_check.py --port can_follower1 --samples 5 --interval 3.0
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--port` | `can_follower1` | follower CAN 인터페이스 |
| `--samples` | `1` | 비교 샘플 개수 |
| `--interval` | `2.0` | 샘플 사이 대기 시간(초) |

**안전**: 팔을 움직이지 않음 — 연결하고 읽기만 함(`EnableFkCal()`도 계산만 켜는 것, 명령 전송 아님). 아무 때나 돌려도 안전함.

## 판정 기준

- **위치(x/y/z) 차이가 1mm 이하, 각도(rx/ry/rz) 차이가 1도 이하** → SDK 계산과 펌웨어 피드백이 같은 모델을 씀. `CalFK()`/`GetFK()`를 실측 피드백 대신 그대로 써도 안전하다는 뜻(예: EEF action을 leader의 joint 목표값에서 FK로 계산해도 신뢰 가능).
- **차이가 크게 남** → 펌웨어가 우리가 모르는 추가 보정(캘리브레이션 오프셋 등)을 갖고 있다는 신호. 이 경우 EEF state는 `CalFK()` 계산 대신 `GetArmEndPoseMsgs()` 실측값을 우선해야 함.

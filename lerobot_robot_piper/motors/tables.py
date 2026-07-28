MODEL_BAUDRATE_TABLE = {
    "AGILEX-M": 1190,
    "AGILEX-S": 1191,
}

MODEL_ENCODINGS_TABLE = {}

MODEL_ENCODING_TABLE = {
    "AGILEX-M": MODEL_ENCODINGS_TABLE,
    "AGILEX-S": MODEL_ENCODINGS_TABLE,
}

MODEL_RESOLUTION_TABLE = {
    "AGILEX-M": 4096,
    "AGILEX-S": 4096,
}

MODEL_NUMBER_TABLE = {
    "AGILEX-M": 1190,
    "AGILEX-S": 1191,
}

MODEL_CONTROL_TABLE = {
    "AGILEX-M": 1190,
    "AGILEX-S": 1191,
}

# parking()이 set_action(INITIALIZE_POSITION, is_conv=True)로 호출되므로 이 값들은
# 정규화값(-100~100)임. joint2/joint3/joint6은 calibration 범위가 0 기준 비대칭이라
# (joint2: 0~180000, joint3: -170000~0, joint6: -100000~130000) 정규화 0이 실제
# 물리 각도 0도가 아님 — 아래 값은 "관절 raw=0(실제 물리 각도 0도, gripper는
# 닫힘 0mm)"에 정확히 대응하도록 각 joint의 calibration min/max로 역산한 값.
INITIALIZE_POSITION = {
    "joint1":    0,
    "joint2": -100,
    "joint3":  100,
    "joint4":    0,
    "joint5":    0,
    "joint6": -13.043478260869563,
    "gripper":   0,
}

# torque를 풀 때 손목(joint5)을 미리 내려둘 각도(도).
# release_torque_safely(mode="lower")가 사용한다.
#
# 실기 측정으로 얻은 값(2026-07-28, 파킹 자세에서):
#   - torque를 풀어도 joint1~4/6은 0.00도, 즉 전혀 안 움직인다(감속비가 커서 자기
#     무게로 역구동되지 않음). 팔 전체가 떨어질 거라는 예상과 다름.
#   - 유일하게 움직이는 게 손목 joint5로, 놓는 순간 24.4도가 뚝 떨어진다.
#     "쿵" 하고 놓이는 느낌의 정체가 이것.
#   - 놓기 전에 joint5를 미리 이 각도만큼 내려두면 해제 시 움직임이 0.6도로 줄었다
#     (24.4 -> 0.6도, 약 40배). 즉 팔을 어디로 옮길 필요 없이 손목만 미리 내리면 된다.
#
# 팔 자세에 따라 손목에 걸리는 중력 방향이 달라지므로 이 값이 항상 최적은 아니다 —
# GUI의 "Measure Wrist Drop"으로 지금 쓰는 종료 자세에서 다시 재서 recording.env
# (PARK_RELEASE_WRIST_DROP_DEG)에 저장할 수 있다.
WRIST_RELEASE_DROP_DEG = 24.4

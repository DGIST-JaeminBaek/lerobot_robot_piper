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

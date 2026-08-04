import logging
import math
import time
from typing import Any

from piper_sdk import C_PiperInterface_V2
from wego_piper.port_handler import PortHandler

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.motors_bus import MotorsBus

from .tables import (
    INITIALIZE_POSITION,
    MODEL_BAUDRATE_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION_TABLE,
    WRIST_RELEASE_REST_DEG,
)

logger = logging.getLogger(__name__)

# ModeCtrl(0x151)의 MOVE 모드 (piper_sdk ModeCtrl docstring 참고).
#
# MOVE_J는 "이 목표로 이동하라"는 점대점 명령이다 — 컨트롤러가 목표마다 가속/감속
# 궤적을 새로 계획한다. 목표를 하나 주고 도착을 기다리는 방식(파킹, 캘리브레이션,
# 사람이 리더를 천천히 움직이는 텔레옵)에는 맞다.
#
# 문제는 정책 추론처럼 30Hz로 목표를 계속 흘려보낼 때다. 33ms마다 새 점대점 명령이
# 들어오면 컨트롤러가 초당 30번 궤적을 처음부터 다시 계획한다 — 움직이다 말고
# 재시작하기를 반복해서 팔이 떤다. MOVE_CPV(연속 위치-속도)는 그런 스트리밍
# setpoint를 위한 모드다. 펌웨어 V1.8-1 이상이 필요하다(이 랩 팔은 S-V1.8-2).
MOVE_P = 0x00
MOVE_J = 0x01
MOVE_L = 0x02
MOVE_C = 0x03
MOVE_M = 0x04
MOVE_CPV = 0x05

# MIT 제어 대상 관절 (그리퍼는 별개 CAN 노드라 제외). JointMitCtrl의 motor_num이
# 1부터 시작하므로 이 순서가 곧 모터 번호다.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class PiperMotorsBus(MotorsBus):

    apply_drive_mode = False
    available_baudrates = [1000000]
    default_baudrate = 1000000
    default_timeout = 1000
    model_baudrate_table = {model: [1000000] for model in MODEL_BAUDRATE_TABLE}
    model_ctrl_table = {model: {} for model in MODEL_NUMBER_TABLE}
    model_encoding_table = MODEL_ENCODING_TABLE
    model_number_table = MODEL_NUMBER_TABLE
    model_resolution_table = MODEL_RESOLUTION_TABLE
    normalized_data = ["Present_Position", "Goal_Position"]
    # __init__이 덮어쓰지만 클래스 기본값으로도 둔다 — 테스트나 진단 도구가
    # __init__을 거치지 않고 버스를 만들어도 set_action()이 동작해야 한다.
    move_mode = MOVE_J
    move_speed_rate = 30

    def __init__(
        self,
        id: str,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        move_mode: int = MOVE_J,
        move_speed_rate: int = 30,
    ):
        super().__init__(port, motors, calibration)

        self.port_handler = PortHandler()
        self.id = id
        self._is_connected = False
        # ModeCtrl(0x151)의 MOVE 모드. set_action()이 매 명령마다 함께 보낸다.
        # 자세한 차이는 MOVE_J / MOVE_CPV 상수 설명 참고.
        self.move_mode = move_mode
        self.move_speed_rate = move_speed_rate
        self.piper = C_PiperInterface_V2(port)
        logger.info(f"{id} : {port} is selected.")

    # ---- MotorsBus abstract implementations ----

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, handshake: bool = True) -> None:
        self.port_handler.setupPort(self.piper)
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open port for {self.id}")
        self._is_connected = True

    def disconnect(
        self,
        disable_torque: bool = True,
        park: bool | None = None,
        release_mode: str | None = None,
        wrist_rest_deg: float = WRIST_RELEASE_REST_DEG,
        release_ramp_s: float = 2.0,
        release_settle_s: float = 0.5,
        gripper_cycle: bool = True,
        gripper_open: float = 100.0,
        gripper_wait_s: float = 1.0,
    ) -> None:
        # park과 disable_torque를 분리 — follower는 항상 parking 자세로는 가되
        # torque 자동 해제 여부만 선택하고 싶은 경우(DISABLE_TORQUE_ON_DISCONNECT=false
        # + scripts/tools/safe_release_torque.py 조합)를 지원하기 위함.
        # park을 명시하지 않으면 기존 동작과 동일하게 disable_torque 값을 따름.
        if park is None:
            park = disable_torque

        # torque를 풀 때는 "어떤 자세로 가서 풀지"를 release_mode가 결정한다
        # (release_torque_safely 참고). release_mode를 안 주면 기존 동작 그대로 —
        # park이면 parking 이동 후 해제, 아니면 그 자리에서 해제.
        if disable_torque:
            mode = release_mode or ("park" if park else "in_place")
            self.release_torque_safely(
                mode=mode,
                wrist_rest_deg=wrist_rest_deg,
                ramp_s=release_ramp_s,
                settle_s=release_settle_s,
                gripper_cycle=gripper_cycle,
                gripper_open=gripper_open,
                gripper_wait_s=gripper_wait_s,
            )
        elif park:
            self.parking()

        self.port_handler.closePort()
        self._is_connected = False

    def read(self, data_name: str, motor: str) -> int | float:
        pos = self.get_action()
        return pos.get(motor, 0)

    def write(self, data_name: str, motor: str, value: int | float) -> None:
        current = self.get_action()
        current[motor] = value
        self.set_action(current, is_conv=True)

    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, int | float]:
        pos = self.get_action()
        if motors is None:
            return pos
        if isinstance(motors, str):
            motors = [motors]
        return {m: pos[m] for m in motors if m in pos}

    def sync_write(self, data_name: str, values: dict[str, int | float]) -> None:
        self.set_action(values, is_conv=True)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        retry = num_retry if num_retry > 0 else 50  # 5 seconds max by default
        while not self.piper.EnablePiper() and retry:
            retry -= 1
            time.sleep(0.1)
        if not retry:
            enable_status = self.piper.GetArmEnableStatus()
            raise ConnectionError(f"{self.id} enable_torque timed out: {enable_status}")
        logger.info(f"{self.piper.GetArmEnableStatus()}")
        logger.info(f"{self.id} torque on.")

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self.piper.DisablePiper()
        # 그리퍼는 별개 노드라 DisablePiper()로 안 풀린다 (disable_gripper 참고)
        self.disable_gripper()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        self.calibration = calibration_dict

    # ---- MotorsBus serial-protocol compatibility ----

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _handshake(self) -> None:
        pass

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        raise NotImplementedError("Piper CAN bus does not support single motor discovery.")

    def configure_motors(self) -> None:
        pass

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        self.disable_torque(num_retry=num_retry)

    def _get_half_turn_homings(self, positions: dict[str | int, int | float]) -> dict[str | int, int | float]:
        raise NotImplementedError("Piper CAN bus uses static calibration ranges.")

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return [(value >> (8 * idx)) & 0xFF for idx in range(length)]

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        return {motor.id: motor.id for motor in self.motors.values()}

    # ---- Piper-specific methods ----

    @property
    def is_calibrated(self) -> bool:
        return True

    def clear_gripper(self):
        self.piper.GripperCtrl(0, 1000, 0x03, 0)

    def parking(self):
        timeout = 100  # 10sec
        self.set_action(INITIALIZE_POSITION, True)
        time.sleep(0.1)
        status = self.piper.GetArmStatus()

        while status.arm_status.motion_status and timeout:
            self.set_action(INITIALIZE_POSITION, True)
            time.sleep(0.1)
            status = self.piper.GetArmStatus()
            timeout -= 1

    def ramp_to(self, target: dict[str, float], ramp_s: float = 2.0, step_s: float = 0.05) -> None:
        """현재 자세에서 target까지 정규화값을 선형 보간해서 천천히 이동.

        set_action()을 목표값으로 한 번 쏘면 컨트롤러가 알아서 가긴 하지만 속도를
        우리가 못 정해서(ModeCtrl 고정 speed) 낮은 자세로 내릴 때 훅 떨어지는
        느낌이 난다 — 여기서 중간 목표를 잘게 쪼개 보내서 감속 이동을 만든다.
        target에 없는 관절은 현재 값을 그대로 유지한다(예: gripper).
        """
        start = self.get_action()
        steps = max(1, int(round(ramp_s / step_s)))
        for i in range(1, steps + 1):
            alpha = i / steps
            self.set_action(
                {m: val + (target.get(m, val) - val) * alpha for m, val in start.items()},
                is_conv=True,
            )
            time.sleep(step_s)

    # ---- 그리퍼 (팔 모터와 별개 노드 — 0x159, GripperCtrl) ----

    def gripper_ctrl(self, angle_norm: float, effort: int = 1000, code: int = 0x03) -> None:
        """그리퍼 단독 제어. angle_norm은 다른 관절과 같은 정규화값(0~100).

        code: 0x00 실능, 0x01 사용, 0x02 실능+에러클리어, 0x03 사용+에러클리어.
        평소 동작(set_action)은 0x03을 쓴다.
        """
        raw = abs(int(self._unnormalize({"gripper": angle_norm})["gripper"]))
        self.piper.GripperCtrl(raw, effort, code, 0)

    def gripper_status_code(self) -> int:
        """그리퍼 피드백 상태코드(0x2A Byte 6). bit[6]이 구동기 사용(1)/실능(0)."""
        return int(self.piper.GetArmGripperMsgs().gripper_state.status_code)

    def is_gripper_enabled(self) -> bool:
        return bool(self.gripper_status_code() & (1 << 6))

    def disable_gripper(self, retries: int = 6, wait_s: float = 0.2) -> bool:
        """그리퍼 힘 풀기(실능). 실제로 풀렸는지 확인될 때까지 재시도하고 결과를 반환.

        DisablePiper()는 팔 모터(0x471)만 내리고 그리퍼는 별개 노드(0x159)라 그대로
        물고 있어서, 종료 후에도 그리퍼가 안 풀리는 원인이 됐다.

        한 번만 쏘고 끝내면 프레임이 씹히거나 드라이버 에러 상태에서 무시될 수 있어서
        (실기에서 실제로 안 풀렸다), 상태코드 bit[6]으로 확인하면서 0x00(실능)과
        0x02(실능+에러클리어)를 번갈아 보낸다 — SDK 데모(piper_ctrl_gripper.py)도
        0x02로 리셋한 뒤 쓴다. 각도는 현재값 그대로, effort=0으로 줘서 실능 직전에
        움직이지 않게 한다.
        """
        current = self.get_action()["gripper"]
        for attempt in range(retries):
            code = 0x00 if attempt % 2 == 0 else 0x02
            self.gripper_ctrl(current, effort=0, code=code)
            time.sleep(wait_s)
            if not self.is_gripper_enabled():
                logger.info(f"{self.id} gripper disabled (code=0x{code:02X}, {attempt + 1}회 시도).")
                return True
        logger.warning(
            f"{self.id} gripper 실능 실패 — status_code=0x{self.gripper_status_code():02X} "
            f"(bit6=1이면 아직 사용 중). 전원을 껐다 켜야 할 수 있음."
        )
        return False

    def cycle_gripper(
        self,
        open_norm: float = 100.0,
        close_norm: float = 0.0,
        wait_s: float = 1.0,
        effort: int = 1000,
    ) -> None:
        """그리퍼를 한 번 열고 다시 닫아서 파킹 위치로 되돌린다.

        물고 있던 물체를 놓게 하고(열기), 보관/시작 자세인 닫힘 상태로 맞춘다.
        주의: 여는 순간 잡고 있던 물체가 떨어지고, 닫을 때 손가락이 끼일 수 있다 —
        사람이 그리퍼 안에 손을 두지 않은 상태에서만 쓸 것.
        """
        logger.info(f"{self.id} gripper cycle: open({open_norm:.0f}) -> close({close_norm:.0f})")
        # 팔이 CAN 제어 모드가 아니면 그리퍼 각도 명령이 무시된다 — set_action()은
        # 매번 이걸 먼저 보내지만, 여기서는 팔 관절 명령 없이 그리퍼만 움직이므로
        # 직접 한 번 보내줘야 한다(실기에서 이거 없이는 각도가 안 변하는 걸 확인).
        self.piper.ModeCtrl(0x01, MOVE_J, self.move_speed_rate, 0x00)
        self.gripper_ctrl(open_norm, effort=effort)
        time.sleep(wait_s)
        self.gripper_ctrl(close_norm, effort=effort)
        time.sleep(wait_s)

    def wrist_rest_target(self, rest_deg: float) -> dict[str, float]:
        """손목(joint5)을 rest_deg(자연 정지각, 절대 각도)로 내린 목표 자세(정규화값).

        상대 델타가 아니라 절대 각도인 게 중요하다 — 상대로 하면 손목이 이미 정지각에
        있을 때 그보다 더 아래로 명령하게 되고, 놓는 순간 그만큼 다시 튕겨 올라온다
        (실기에서 확인: 손목 30도에서 +24.4도를 더 줬다가 해제하니 29.9도로 복귀).

        이미 정지각보다 아래(각도가 큰 쪽)면 그대로 둔다 — 손목을 도로 들어올리면
        놓을 때 다시 떨어지므로.
        """
        cal = self.calibration["joint5"]
        rest_norm = ((rest_deg * 1000.0 - cal.range_min) / (cal.range_max - cal.range_min)) * 200 - 100
        target = {m: v for m, v in self.get_action().items() if m != "gripper"}
        target["joint5"] = max(target["joint5"], rest_norm)
        return target

    def release_torque_safely(
        self,
        mode: str = "in_place",
        wrist_rest_deg: float = WRIST_RELEASE_REST_DEG,
        ramp_s: float = 2.0,
        settle_s: float = 0.5,
        gripper_cycle: bool = True,
        gripper_open: float = 100.0,
        gripper_wait_s: float = 1.0,
    ) -> None:
        """torque를 푸는 세 가지 방식.

        - "in_place": 이동 없이 지금 자세 그대로 해제. 예상 못 한 이동이 아예 없다.
        - "lower": 팔은 그 자리에 두고 손목(joint5)만 wrist_rest_deg(자연 정지각)까지
          미리 내린 뒤 해제. 실기 측정 결과 torque를 풀 때 실제로 떨어지는 건 손목뿐이라
          (joint1~4/6은 0.00도) 그 낙차를 미리 없애는 것 — tables.py의
          WRIST_RELEASE_REST_DEG 주석 참고. 팔을 옮기지 않으므로 이동 위험이 없다.
        - "park": parking 자세로 이동한 뒤 해제. 손목 낙차는 남는다.
        - "park_lower": parking 자세로 이동한 뒤, 거기서 손목까지 정지각으로 내리고
          해제. "park"과 "lower"를 합친 것으로, 실행이 끝났을 때 팔이 보관 자세에
          있으면서 놓는 순간의 손목 낙차도 없다 — 정상 종료의 기본값.

        팔 관절 이동에서는 gripper를 건드리지 않는다(잡고 있는 물체/손이 끼지 않도록).
        gripper_cycle=True면 마지막에 그리퍼를 한 번 열고 닫아서 물고 있던 걸 놓고
        파킹 위치(닫힘)로 되돌린 뒤 실능시킨다.
        """
        mode = (mode or "in_place").lower()
        if mode in ("park", "park_lower"):
            self.parking()
        if mode in ("lower", "park_lower"):
            # park_lower면 parking 자세에 도착한 *뒤* 그 자세 기준으로 손목을
            # 내린다 — wrist_rest_target()이 현재 자세를 읽어 목표를 만들므로
            # 순서가 중요하다.
            self.ramp_to(self.wrist_rest_target(wrist_rest_deg), ramp_s=ramp_s)
            # 마지막 목표에 실제로 도달할 시간을 준 뒤에 풀어야 "거의 다 내려간
            # 상태"가 아니라 "다 내려간 상태"에서 해제된다.
            time.sleep(settle_s)
        elif mode not in ("in_place", "park"):
            logger.warning(f"{self.id} unknown release mode '{mode}' — in_place로 처리")

        # 팔 토크를 내리기 전에 그리퍼를 정리한다 — 팔이 늘어진 뒤에 그리퍼를
        # 여닫으면 반력으로 팔이 흔들린다.
        if gripper_cycle:
            self.cycle_gripper(
                open_norm=gripper_open,
                close_norm=INITIALIZE_POSITION["gripper"],
                wait_s=gripper_wait_s,
            )

        logger.info(f"{self.id} releasing torque (mode={mode}).")
        self.piper.DisablePiper()
        # DisablePiper()는 팔 모터만 내린다 — 그리퍼는 따로 실능시켜야 풀린다.
        self.disable_gripper()

    def measure_wrist_rest(self, settle_s: float = 3.0) -> tuple[float, float]:
        """torque를 풀고 손목(joint5)이 멎는 각도를 측정. (정지각, 낙차) 도 단위.

        GUI의 "Measure Wrist Rest"가 사용 — 종료 자세가 바뀌면 손목에 걸리는 중력
        방향도 바뀌므로 그 자세에서 다시 재서 wrist_rest_deg를 갱신하기 위한 것.
        측정이 끝나면 torque는 풀린 상태로 남는다(다시 잡으려면 enable_torque).
        """
        cal = self.calibration["joint5"]

        def to_deg(norm: float) -> float:
            return (((norm + 100) / 200) * (cal.range_max - cal.range_min) + cal.range_min) / 1000.0

        before = to_deg(self.get_action()["joint5"])
        self.piper.DisablePiper()
        time.sleep(settle_s)
        after = to_deg(self.get_action()["joint5"])
        return after, after - before

    def set_slave(self):
        self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)

    def set_master(self):
        self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)

    def get_action(self) -> dict[str, Any]:
        msg_joint = self.piper.GetArmJointMsgs()
        msg_gripr = self.piper.GetArmGripperMsgs()
        rlt = {
            "joint1": float(msg_joint.joint_state.joint_1),
            "joint2": float(msg_joint.joint_state.joint_2),
            "joint3": float(msg_joint.joint_state.joint_3),
            "joint4": float(msg_joint.joint_state.joint_4),
            "joint5": float(msg_joint.joint_state.joint_5),
            "joint6": float(msg_joint.joint_state.joint_6),
            "gripper": float(msg_gripr.gripper_state.grippers_angle),
        }
        return self._normalize(rlt)

    def get_effort(self) -> dict[str, float]:
        # GetArmHighSpdInfoMsgs().motor_N.effort: current 기반 고정계수 변환값, 단위 0.001 N·m.
        # 실제 토크 센서 값이 아니라 자세(중력/마찰)에 따라 접촉 없이도 값이 바뀔 수 있음.
        msg_hs = self.piper.GetArmHighSpdInfoMsgs()
        msg_gripr = self.piper.GetArmGripperMsgs()
        return {
            "joint1": msg_hs.motor_1.effort * 0.001,
            "joint2": msg_hs.motor_2.effort * 0.001,
            "joint3": msg_hs.motor_3.effort * 0.001,
            "joint4": msg_hs.motor_4.effort * 0.001,
            "joint5": msg_hs.motor_5.effort * 0.001,
            "joint6": msg_hs.motor_6.effort * 0.001,
            "gripper": msg_gripr.gripper_state.grippers_effort * 0.001,
        }

    def get_velocity(self) -> dict[str, float]:
        # motor_speed: 0.001 rad/s. 안전 컷오프가 자세/속도로 인한 effort 노이즈를
        # 걸러내려면 effort와 같은 타임스탬프의 속도가 필요 — get_effort()와 짝지어 로깅.
        msg_hs = self.piper.GetArmHighSpdInfoMsgs()
        return {
            "joint1": msg_hs.motor_1.motor_speed * 0.001,
            "joint2": msg_hs.motor_2.motor_speed * 0.001,
            "joint3": msg_hs.motor_3.motor_speed * 0.001,
            "joint4": msg_hs.motor_4.motor_speed * 0.001,
            "joint5": msg_hs.motor_5.motor_speed * 0.001,
            "joint6": msg_hs.motor_6.motor_speed * 0.001,
        }

    def is_overloaded(self, effort: dict[str, float], limit: float) -> bool:
        # effort: get_effort()가 반환하는 dict(N·m). 관절 하나라도 임계값을 넘으면 True.
        return any(abs(val) > limit for val in effort.values())

    def enter_mit_compliance(
        self,
        pos_ref: dict[str, float],
        kp: float = 10.0,
        kd: float = 0.8,
    ) -> None:
        # 위치 유지형 컴플라이언스(t_ref=0)만 구현 — 실제 순응 힘/kp-kd 튜닝은
        # 실기에서. is_overloaded() 감지 시 우선은 send_action 쪽에서 명령을
        # 그냥 보류(정지)하는 게 기본 경로이고, 이 메서드는 그다음 단계 확장용.
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, is_mit_mode=0xAD)
        for motor_num, joint in enumerate(
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], start=1
        ):
            self.piper.JointMitCtrl(motor_num, pos_ref[joint], 0.0, kp, kd, 0.0)

    def get_control(self) -> dict[str, Any]:
        msg_joint = self.piper.GetArmJointCtrl()
        msg_gripr = self.piper.GetArmGripperCtrl()
        rlt = {
            "joint1": float(msg_joint.joint_ctrl.joint_1),
            "joint2": float(msg_joint.joint_ctrl.joint_2),
            "joint3": float(msg_joint.joint_ctrl.joint_3),
            "joint4": float(msg_joint.joint_ctrl.joint_4),
            "joint5": float(msg_joint.joint_ctrl.joint_5),
            "joint6": float(msg_joint.joint_ctrl.joint_6),
            "gripper": float(msg_gripr.gripper_ctrl.grippers_angle),
        }
        return self._normalize(rlt)

    def set_action(self, action: dict[str, Any], is_conv: bool = True) -> dict[str, Any]:
        if is_conv:
            action_denormalized = self._unnormalize(action)
        else:
            action_denormalized = action

        self.piper.ModeCtrl(0x01, self.move_mode, self.move_speed_rate, 0x00)
        self.piper.JointCtrl(
            int(action_denormalized["joint1"]),
            int(action_denormalized["joint2"]),
            int(action_denormalized["joint3"]),
            int(action_denormalized["joint4"]),
            int(action_denormalized["joint5"]),
            int(action_denormalized["joint6"]),
        )
        self.piper.GripperCtrl(abs(int(action_denormalized["gripper"])), 1000, 0x03, 0)
        return self.get_control()

    # ---- MIT (임피던스) 제어 ----

    # SDK JointMitCtrl docstring의 참고값. kp가 낮으면 팔이 중력에 무너지고,
    # 높으면 격렬하게 진동한다. 반드시 낮은 값에서 시작해 올릴 것.
    MIT_KP_DEFAULT = 10.0
    MIT_KD_DEFAULT = 0.8
    MIT_KP_MAX = 500.0
    MIT_KD_MAX = 5.0

    def _norm_to_rad(self, motor: str, value: float) -> float:
        """정규화 위치(-100~100)를 라디안으로. calibration 범위는 0.001도 단위."""
        milli_deg = self._unnormalize({motor: value})[motor]
        return milli_deg * 0.001 * math.pi / 180.0

    def _norm_rate_to_rad_s(self, motor: str, value_per_s: float) -> float:
        """정규화 속도(단위/초)를 rad/s로. 오프셋 없이 배율만 적용한다."""
        cal = self.calibration[motor]
        # RANGE_M100_100: 정규화 200 구간이 (max-min) 0.001도에 대응
        span_rad = (cal.range_max - cal.range_min) * 0.001 * math.pi / 180.0
        return value_per_s * span_rad / 200.0

    @staticmethod
    def _gain_for(gain: "float | dict[str, float]", motor: str, fallback: float) -> float:
        """관절별 게인을 허용한다. float면 전 관절 공통, dict면 관절별.

        중력 부담이 관절마다 다르므로 공통 게인으로는 맞출 수 없다 — 실측에서
        같은 kp=10에 joint1/4/6은 처짐 0.00인데 joint2는 1.83이었다. 처짐은
        중력토크/kp라 무거운 관절만 kp를 올려야 한다.
        """
        if isinstance(gain, dict):
            return float(gain.get(motor, fallback))
        return float(gain)

    def set_action_mit(
        self,
        goal_pos: dict[str, float],
        goal_vel: dict[str, float] | None = None,
        kp: "float | dict[str, float]" = MIT_KP_DEFAULT,
        kd: "float | dict[str, float]" = MIT_KD_DEFAULT,
    ) -> dict[str, float]:
        """MIT(임피던스) 모드로 관절 목표를 스트리밍한다.

        MOVE J와 달리 궤적을 계획하지 않는다 — 매 명령이 그냥
        `토크 = kp*(pos_ref - pos) + kd*(vel_ref - vel) + t_ref` 를 갱신할 뿐이라,
        33ms마다 새 목표를 흘려보내도 재계획이 일어나지 않는다. 30Hz 스트리밍에
        맞는 인터페이스가 이것이다.

        속도를 함께 보내는 게 핵심이다. 정책 chunk는 30fps로 시간 매개화된
        궤적이라 각 시점의 의도된 속도를 이미 담고 있는데, 위치만 보내면 그
        정보를 버리게 된다.

        ⚠ 위험: 이건 토크 제어다. kp가 낮으면 팔이 중력에 무너지고 높으면
        진동한다. max_relative_target(위치 명령 클램프)과 effort 컷오프는 위치
        제어를 전제로 만들어진 것이라 여기서는 의미가 달라진다. 반드시 낮은
        게인에서 관절 하나씩 확인할 것 — scripts/tools/piper_mit_probe.py 참고.

        그리퍼는 MIT 대상이 아니다(별개 노드) — 기존 GripperCtrl을 그대로 쓴다.
        """
        goal_vel = goal_vel or {}
        gains = {}
        for motor in JOINT_NAMES:
            motor_kp = self._gain_for(kp, motor, self.MIT_KP_DEFAULT)
            motor_kd = self._gain_for(kd, motor, self.MIT_KD_DEFAULT)
            if not (0.0 <= motor_kp <= self.MIT_KP_MAX):
                raise ValueError(f"{motor} kp must be in [0, {self.MIT_KP_MAX}]; got {motor_kp}")
            if not (-self.MIT_KD_MAX <= motor_kd <= self.MIT_KD_MAX):
                raise ValueError(
                    f"{motor} kd must be in [-{self.MIT_KD_MAX}, {self.MIT_KD_MAX}]; got {motor_kd}"
                )
            gains[motor] = (motor_kp, motor_kd)

        self.piper.ModeCtrl(0x01, MOVE_M, self.move_speed_rate, 0xAD)
        for index, motor in enumerate(JOINT_NAMES, start=1):
            if motor not in goal_pos:
                continue
            motor_kp, motor_kd = gains[motor]
            self.piper.JointMitCtrl(
                index,
                self._norm_to_rad(motor, goal_pos[motor]),
                self._norm_rate_to_rad_s(motor, goal_vel.get(motor, 0.0)),
                motor_kp,
                motor_kd,
                0.0,  # t_ref — 중력 보상은 컨트롤러에 맡긴다
            )
        if "gripper" in goal_pos:
            gripper = self._unnormalize({"gripper": goal_pos["gripper"]})["gripper"]
            self.piper.GripperCtrl(abs(int(gripper)), 1000, 0x03, 0)
        return self.get_control()

    def leave_mit_mode(self) -> None:
        """MIT를 끄고 위치 제어(MOVE J)로 되돌린다. 종료 경로에서 반드시 부를 것 —
        MIT 상태로 parking()을 부르면 위치 명령이 먹지 않는다."""
        self.piper.ModeCtrl(0x01, MOVE_J, self.move_speed_rate, 0x00)

    # ---- Normalization ----

    def _normalize(self, ids_values: dict[str, int]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for motor, val in ids_values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[motor] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[motor] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION_TABLE[self.motors[motor].model] - 1
                normalized_values[motor] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[str, float]) -> dict[str, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for motor, val in ids_values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[motor] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[motor] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION_TABLE[self.motors[motor].model] - 1
                unnormalized_values[motor] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

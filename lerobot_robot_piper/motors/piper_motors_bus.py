import logging
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
    WRIST_RELEASE_DROP_DEG,
)

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        id: str,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)

        self.port_handler = PortHandler()
        self.id = id
        self._is_connected = False
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
        wrist_drop_deg: float = WRIST_RELEASE_DROP_DEG,
        release_ramp_s: float = 2.0,
        release_settle_s: float = 0.5,
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
                wrist_drop_deg=wrist_drop_deg,
                ramp_s=release_ramp_s,
                settle_s=release_settle_s,
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

    def wrist_drop_target(self, drop_deg: float) -> dict[str, float]:
        """현재 자세에서 손목(joint5)만 drop_deg만큼 내린 목표 자세(정규화값)."""
        cal = self.calibration["joint5"]
        # 정규화 스케일: 전체 범위(raw)가 -100~100(=200)에 대응
        delta_norm = (drop_deg * 1000.0) / (cal.range_max - cal.range_min) * 200
        target = {m: v for m, v in self.get_action().items() if m != "gripper"}
        target["joint5"] += delta_norm
        return target

    def release_torque_safely(
        self,
        mode: str = "in_place",
        wrist_drop_deg: float = WRIST_RELEASE_DROP_DEG,
        ramp_s: float = 2.0,
        settle_s: float = 0.5,
    ) -> None:
        """torque를 푸는 세 가지 방식.

        - "in_place": 이동 없이 지금 자세 그대로 해제. 예상 못 한 이동이 아예 없다.
        - "lower": 팔은 그 자리에 두고 손목(joint5)만 wrist_drop_deg만큼 미리 내린
          뒤 해제. 실기 측정 결과 torque를 풀 때 실제로 떨어지는 건 손목뿐이라
          (joint1~4/6은 0.00도) 그 낙차를 미리 없애는 것 — tables.py의
          WRIST_RELEASE_DROP_DEG 주석 참고. 팔을 옮기지 않으므로 이동 위험이 없다.
        - "park": 기존 동작 — parking 자세로 이동한 뒤 해제.

        어느 쪽이든 gripper는 건드리지 않는다(뭔가 잡고 있을 때 손/물체가 끼지 않도록).
        """
        mode = (mode or "in_place").lower()
        if mode == "park":
            self.parking()
        elif mode == "lower":
            self.ramp_to(self.wrist_drop_target(wrist_drop_deg), ramp_s=ramp_s)
            # 마지막 목표에 실제로 도달할 시간을 준 뒤에 풀어야 "거의 다 내려간
            # 상태"가 아니라 "다 내려간 상태"에서 해제된다.
            time.sleep(settle_s)
        elif mode != "in_place":
            logger.warning(f"{self.id} unknown release mode '{mode}' — in_place로 처리")

        logger.info(f"{self.id} releasing torque (mode={mode}).")
        self.piper.DisablePiper()

    def measure_wrist_drop(self, settle_s: float = 3.0) -> float:
        """지금 자세에서 torque를 풀고 손목(joint5)이 얼마나 떨어지는지 측정(도).

        GUI의 "Measure Wrist Drop"이 사용 — 종료 자세가 바뀌면 손목에 걸리는 중력
        방향도 바뀌므로, 그 자세에서 다시 재서 wrist_drop_deg를 갱신하기 위한 것.
        측정이 끝나면 torque는 풀린 상태로 남는다(다시 잡으려면 enable_torque).
        """
        cal = self.calibration["joint5"]
        before = self.get_action()["joint5"]
        self.piper.DisablePiper()
        time.sleep(settle_s)
        after = self.get_action()["joint5"]
        return (after - before) / 200 * (cal.range_max - cal.range_min) / 1000.0

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

        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
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

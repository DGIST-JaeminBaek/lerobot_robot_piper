import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_piper import PiperFollowerConfig
from .depth_utils import depth_to_colormap
from .motors import PiperMotorsBus

logger = logging.getLogger(__name__)


class PiperFollower(Robot):

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.port = config.port
        self.cameras: dict[str, Camera] = {}
        self.bus = PiperMotorsBus(
            id=config.id,
            port=config.port,
            motors={
                "joint1": Motor(1, "AGILEX-M", MotorNormMode.RANGE_M100_100),
                "joint2": Motor(2, "AGILEX-M", MotorNormMode.RANGE_M100_100),
                "joint3": Motor(3, "AGILEX-M", MotorNormMode.RANGE_M100_100),
                "joint4": Motor(4, "AGILEX-S", MotorNormMode.RANGE_M100_100),
                "joint5": Motor(5, "AGILEX-S", MotorNormMode.RANGE_M100_100),
                "joint6": Motor(6, "AGILEX-S", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(7, "AGILEX-S", MotorNormMode.RANGE_0_100),
            },
            calibration={
                "joint1": MotorCalibration(1, 0, 0, -150000, 150000),
                "joint2": MotorCalibration(2, 0, 0, 0, 180000),
                "joint3": MotorCalibration(3, 0, 0, -170000, 0),
                "joint4": MotorCalibration(4, 0, 0, -100000, 100000),
                "joint5": MotorCalibration(5, 0, 0, -65000, 65000),
                "joint6": MotorCalibration(6, 0, 0, -100000, 130000),
                "gripper": MotorCalibration(7, 0, 0, 0, 68000),
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._action_offset: dict[str, float] | None = None
        self._action_offset_start_time: float | None = None
        self._action_offset_reported = False
        self._camera_executor: ThreadPoolExecutor | None = None
        self._ensure_camera_executor()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _effort_ft(self) -> dict[str, type]:
        if not self.config.use_effort:
            return {}
        return {f"{motor}.effort": float for motor in self.bus.motors}

    @property
    def _velocity_ft(self) -> dict[str, type]:
        # NEXT(외력 추정)는 effort와 같은 타임스탬프의 속도가 필요 — 별도 플래그를
        # 늘리지 않고 use_effort에 묶어서 같이 켠다. 그리퍼는 SDK에 motor_speed가 없음.
        if not self.config.use_effort:
            return {}
        return {f"{motor}.vel": float for motor in self.bus.motors if motor != "gripper"}

    @property
    def _depth_cam_keys(self) -> list[str]:
        # RealSense depth 스트림 자체(realsense_use_depth 등)가 켜진 카메라만 대상.
        # use_depth_observation이 꺼져 있으면 스트림이 있어도 observation엔 안 넣음.
        if not self.config.use_depth_observation:
            return []
        return [cam for cam, obj in self.cameras.items() if getattr(obj, "use_depth", False)]

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        ft = {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}
        for cam in self._depth_cam_keys:
            ft[f"{cam}_depth"] = (self.cameras[cam].height, self.cameras[cam].width, 3)
        return ft

    @cached_property
    def observation_features(self) -> dict:
        # lerobot의 hw_to_dataset_features()가 float 타입 키를 전부 observation.state로
        # 합치므로, effort는 별도 스키마 배선 없이 .pos 옆에 얹히기만 하면 됨(TA-VLA STATE/DePre와 동일 취급).
        return {**self._motors_ft, **self._effort_ft, **self._velocity_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def get_cameras(self) -> dict[str, Camera]:
        return self.cameras

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        logger.info(f"{self} connected.")
        self.bus.enable_torque()
        logger.info(f"{self} torque on.")

        # 재시작 시 현재 자세 유지
        if calibrate and self.config.park_on_connect:
            logger.info(f"{self} go to origin.")
            self.bus.parking()

        # 카메라별 connect()가 각자 warmup_s(예: RealSense 10초)만큼 블로킹해서
        # 카메라 개수만큼 그대로 곱해짐(top+wrist 2대면 20초+) — record/teleoperate
        # 시작 직후 그동안 teleop이 응답 없는 것처럼 보이는 원인. 병렬 연결로 줄여봤으나
        # RealSense 2대를 정확히 동시에 초기화하면 USB 대역폭 경합으로 한쪽이
        # "read failed"/타임아웃 나는 게 실제 하드웨어에서 확인됨 — 그래서 순차 연결
        # 유지. 대기 시간을 줄이고 싶으면 REALSENSE_WARMUP_S를 낮추는 쪽으로 접근할 것
        # (단, 너무 낮추면 원래 있었던 "녹화마다 카메라 타임아웃" 문제가 재현될 수 있음).
        for cam in self.cameras.values():
            cam.connect(warmup=self.config.camera_connect_warmup)

        if self.cameras and not self.config.camera_connect_warmup:
            # 동시 RealSense stream 안정화 대기
            time.sleep(self.config.camera_post_connect_wait_s)
        self._ensure_camera_executor()

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.clear_gripper()

    def _load_calibration(self, fpath: Path | None = None) -> None:
        pass

    def _save_calibration(self, fpath: Path | None = None) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_slave()

    def _read_depth_colormap(self, cam_key: str, cam) -> Any:
        # lerobot RealSenseCamera는 async_read()가 color만 반환하고 depth용 공개 API가
        # 없음(코어 소스에 "Missing implementation for depth for now" 표시됨) — background
        # read thread가 채워두는 latest_depth_frame을 frame_lock으로 직접 읽는다.
        # lerobot 코어가 depth API를 추가하면 이 부분을 그쪽으로 교체할 것.
        with cam.frame_lock:
            depth_raw = cam.latest_depth_frame
        if depth_raw is None:
            raise RuntimeError(f"{cam} has no depth frame yet (warmup 중이거나 use_depth 미설정).")
        if self.config.depth_raw_dir:
            self._save_raw_depth(cam_key, depth_raw)
        return depth_to_colormap(
            depth_raw,
            depth_scale=self.config.depth_scale,
            dmin=self.config.depth_min_m,
            dmax=self.config.depth_max_m,
        )

    def _save_raw_depth(self, cam_key: str, depth_raw) -> None:
        # 세션 전체(에피소드 무관) 단조 증가 인덱스 + wall-clock time으로 저장.
        # dataset의 frame_index와 정확히 대응하지 않으니, 나중에 맞출 땐 파일명의
        # unix time을 데이터셋 타임스탬프 쪽과 매칭할 것.
        idx = getattr(self, "_depth_raw_idx", 0)
        out_dir = Path(self.config.depth_raw_dir) / cam_key
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{idx:08d}_{time.time():.6f}.npy", depth_raw)
        self._depth_raw_idx = idx + 1

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.get_action()
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        if self.config.use_effort:
            effort = self.bus.get_effort()
            obs_dict.update({f"{motor}.effort": val for motor, val in effort.items()})
            velocity = self.bus.get_velocity()
            obs_dict.update({f"{motor}.vel": val for motor, val in velocity.items()})

        # Capture images from cameras (parallel)
        if self.cameras:
            futures = {
                cam_key: (self._camera_executor.submit(cam.async_read), time.perf_counter())
                for cam_key, cam in self.cameras.items()
            }
            for cam_key, (future, start) in futures.items():
                obs_dict[cam_key] = future.result()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

                if cam_key in self._depth_cam_keys:
                    obs_dict[f"{cam_key}_depth"] = self._read_depth_colormap(cam_key, self.cameras[cam_key])

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                goal_pos[key.removesuffix(".pos")] = val
            else:
                goal_pos[key] = val

        if self.config.use_action_offset:
            present_pos = self.bus.sync_read("Present_Position")
            if self._action_offset is None:
                if self.config.use_manual_action_offset:
                    # recording.env 기준 수동 offset
                    self._action_offset = self._manual_action_offset(goal_pos)
                    logger.info(f"{self} manual action offset applied: {self._action_offset}")
                    self._report_action_offset(goal_pos, present_pos, self._action_offset)
                else:
                    # leader control frame이 시작 직후 늦게 안정화될 수 있어 바로 고정하지 않는다.
                    self._action_offset_start_time = time.perf_counter()
                    self._action_offset = {key: present_pos[key] - val for key, val in goal_pos.items()}
                    logger.info(
                        f"{self} action offset warmup started "
                        f"({self.config.action_offset_warmup_s:.1f}s): {self._action_offset}"
                    )
            elif not self.config.use_manual_action_offset and self._action_offset_start_time is not None:
                elapsed_s = time.perf_counter() - self._action_offset_start_time
                self._action_offset = {key: present_pos[key] - val for key, val in goal_pos.items()}
                if elapsed_s >= self.config.action_offset_warmup_s:
                    logger.info(f"{self} action offset locked: {self._action_offset}")
                    self._report_action_offset(goal_pos, present_pos, self._action_offset)
                    self._action_offset_start_time = None

            # follower 현재 자세 기준 상대 추종
            goal_pos = {key: val + self._action_offset.get(key, 0.0) for key, val in goal_pos.items()}

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 실시간 안전 컷오프. use_effort(데이터셋 로깅 플래그)와 무관하게 항상 동작 —
        # get_effort()를 여기서 직접 읽는다. 리플레이/정책 출력이 관절 명령으로 바뀌어
        # 로봇에 나가는 지점이 바로 이 set_action 직전이라 여기서 끊는다.
        # 컴플라이언스(MIT 모드) 전환 대신 우선은 이 스텝의 명령을 그냥 보류(마지막으로
        # 실제 전송된 목표를 유지) — 컴플라이언스 kp/kd 튜닝은 다음 단계.
        if self.config.safety_enabled:
            effort = self.bus.get_effort()
            if self.bus.is_overloaded(effort, self.config.safety_effort_limit):
                logger.warning(
                    f"{self} safety cutoff: effort {effort} exceeds "
                    f"{self.config.safety_effort_limit} N·m, holding last commanded position"
                )
                held_pos = getattr(self, "_last_sent_goal_pos", None) or self.bus.get_action()
                return {f"{motor}.pos": val for motor, val in held_pos.items()}

        self.bus.set_action(goal_pos, is_conv=True)
        self._last_sent_goal_pos = goal_pos
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _manual_action_offset(self, goal_pos: dict[str, float]) -> dict[str, float]:
        return {
            key: getattr(self.config, f"action_offset_{key}", 0.0)
            for key in goal_pos
        }

    def _report_action_offset(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
        action_offset: dict[str, float],
    ) -> None:
        # 시작 자세 차이 1회 출력
        if self._action_offset_reported:
            return

        threshold = self.config.action_offset_report_threshold
        lines = [f"{self} action offset report"]
        for key in sorted(action_offset):
            diff = action_offset[key]
            mark = " CHECK" if abs(diff) >= threshold else ""
            lines.append(
                f"  {key}: leader={goal_pos[key]:.3f}, follower={present_pos[key]:.3f}, offset={diff:.3f}{mark}"
            )
        lines.append("  set USE_MANUAL_ACTION_OFFSET=true and ACTION_OFFSET_* in recording.env to fix values")
        logger.info("\n".join(lines))
        self._action_offset_reported = True

    def parking(self):
        self.bus.parking()

    def _ensure_camera_executor(self) -> None:
        if self.cameras and self._camera_executor is None:
            self._camera_executor = ThreadPoolExecutor(
                max_workers=len(self.cameras),
                thread_name_prefix="cam_read",
            )

    def _disconnect_cameras(self) -> None:
        if self._camera_executor is not None:
            self._camera_executor.shutdown(wait=True, cancel_futures=True)
            self._camera_executor = None

        for cam_name, cam in self.cameras.items():
            if not cam.is_connected:
                continue
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning(f"{self} failed to disconnect camera '{cam_name}': {exc}")

    def disconnect(self, disable_torque: bool | None = None) -> None:
        if disable_torque is None:
            disable_torque = self.config.disable_torque_on_disconnect
        self._disconnect_cameras()
        # torque 자동 해제 여부와 무관하게 follower는 항상 parking 자세로 이동.
        # DISABLE_TORQUE_ON_DISCONNECT=false로 두면 parking만 하고 torque는
        # 켜진 채로 남아 scripts/tools/safe_release_torque.py로 수동 해제 가능.
        self.bus.disconnect(disable_torque, park=True)

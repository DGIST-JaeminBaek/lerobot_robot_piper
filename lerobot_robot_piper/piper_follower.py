import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.cameras import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_piper import PiperFollowerConfig
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
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

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

        for cam in self.cameras.values():
            # 모든 camera pipeline 먼저 시작
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

        self.bus.set_action(goal_pos, is_conv=True)
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
        self.bus.disconnect(disable_torque, park=True)

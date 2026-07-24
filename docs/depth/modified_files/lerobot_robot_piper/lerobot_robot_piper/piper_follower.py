import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.cameras import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.datasets.depth_utils import DepthFeature
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_piper import PiperFollowerConfig
from .motors import PiperMotorsBus

logger = logging.getLogger(__name__)

# 임시 진단용 — PIPER_LOG_TIMING=1이면 lerobot의 콘솔 로그 레벨(기본 INFO)과
# 무관하게 get_observation()/send_action()의 logger.debug 타이밍 로그를 강제로
# stdout에 찍히게 함. 15Hz 병목 원인 찾는 용도, 확인 끝나면 지워도 됨.
if os.environ.get("PIPER_LOG_TIMING") == "1":
    logger.setLevel(logging.DEBUG)
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


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
        features = {}
        for cam_key, cam in self.cameras.items():
            features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = DepthFeature(cam.height, cam.width)
        return features

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

        # 카메라별 connect()가 각자 warmup_s(예: RealSense 10초)만큼 블로킹해서
        # 카메라 개수만큼 그대로 곱해짐(top+wrist 2대면 20초+, 순차일 때) — record/
        # teleoperate 시작 직후 그동안 teleop이 응답 없는 것처럼 보이는 원인이었음.
        # 예전에 병렬 연결을 시도했다가 RealSense 2대를 정확히 동시에 초기화하면
        # 한쪽이 "read failed"/타임아웃 나는 걸 실제 하드웨어에서 확인해서 순차로
        # 되돌렸었는데, 그 원인이 USB 대역폭 경합이 아니라 그 당시 CPU 쿨링 문제였을
        # 가능성이 제기돼(2026-07-24) scripts/tools/camera_parallel_connect_test.py로
        # 로봇 없이 카메라만 병렬 connect 3회 재검증 — 매번 성공(~10.3~10.4s, 카메라
        # 1대 warmup_s와 거의 동일해서 실제로 겹쳐서 도는 것도 확인됨), depth 프레임도
        # 정상. 그래서 병렬로 되돌림 — 만약 나중에 이 재현 실패가 다시 나타나면
        # 아래를 순차 for 루프로 되돌릴 것.
        if self.cameras:
            with ThreadPoolExecutor(max_workers=len(self.cameras), thread_name_prefix="cam_connect") as executor:
                futures = [
                    executor.submit(cam.connect, warmup=self.config.camera_connect_warmup)
                    for cam in self.cameras.values()
                ]
                for future in futures:
                    future.result()

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

        # Capture images from cameras (parallel) — RGB와 depth를 각각 순차 단계로
        # 나눠 제출하면 (RGB 전부 끝날 때까지 depth를 시작조차 안 해서) 두 단계
        # 시간이 그대로 더해짐(실측 25ms+33ms≈58ms, 목표 33ms의 거의 2배) —
        # 카메라당 color/depth 요청을 한 번에 다 같이 제출해서 전부 병렬로 겹치게 함.
        if self.cameras:
            futures = {}
            for cam_key, cam in self.cameras.items():
                futures[cam_key] = (self._camera_executor.submit(cam.async_read), time.perf_counter())
                if getattr(cam, "use_depth", False):
                    futures[f"{cam_key}_depth"] = (
                        self._camera_executor.submit(cam.read_depth, timeout_ms=0),
                        time.perf_counter(),
                    )

            for key, (future, start) in futures.items():
                result = future.result()
                obs_dict[key] = result[..., None] if key.endswith("_depth") else result
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {key}: {dt_ms:.1f}ms")

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
            _t0 = time.perf_counter()
            present_pos = self.bus.sync_read("Present_Position")
            logger.debug(f"{self} sync_read(offset): {(time.perf_counter() - _t0) * 1e3:.1f}ms")
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

            # follower 현재 자세 기준 상대 추종:
            # leader 절대 자세로 점프하지 않고, leader의 이동 변화량을 follower 현재 자세에 얹는다.
            goal_pos = {key: val + self._action_offset.get(key, 0.0) for key, val in goal_pos.items()}

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            _t0 = time.perf_counter()
            present_pos = self.bus.sync_read("Present_Position")
            logger.debug(f"{self} sync_read(clip): {(time.perf_counter() - _t0) * 1e3:.1f}ms")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        _t0 = time.perf_counter()
        self.bus.set_action(goal_pos, is_conv=True)
        logger.debug(f"{self} set_action: {(time.perf_counter() - _t0) * 1e3:.1f}ms")
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
            # 카메라당 color 요청 1개 + (depth 켜져있으면) depth 요청 1개까지 동시에
            # 제출하므로, worker 수도 그만큼 있어야 실제로 다 같이 병렬로 돎
            # (get_observation() 참고).
            num_workers = len(self.cameras) + sum(
                1 for cam in self.cameras.values() if getattr(cam, "use_depth", False)
            )
            self._camera_executor = ThreadPoolExecutor(
                max_workers=num_workers,
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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


def _opencv_index_or_path(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value)


@RobotConfig.register_subclass("piper_follower")
@dataclass(kw_only=True)
class PiperFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # 재실행 시 follower 강제 parking 방지
    park_on_connect: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # LeRobot CLI dict 파싱 우회용 카메라 필드
    camera_type: str = "opencv"
    top_cam_type: str = ""
    wrist_cam_type: str = ""
    top_cam: str = ""
    wrist_cam: str = ""
    cam_width: int = 640
    cam_height: int = 480
    camera_fps: int = 30
    realsense_use_depth: bool = False
    realsense_warmup_s: float = 5.0
    camera_connect_warmup: bool = False
    camera_post_connect_wait_s: float = 2.0
    top_realsense_use_depth: bool = False
    wrist_realsense_use_depth: bool = False

    # observation에 관절별 effort(전류 기반 추정 토크, N·m) 포함 여부.
    # piper_sdk의 current-derived 값이라 자세에 따른 중력/마찰 성분이 섞여 있음(진짜 토크 센서 아님).
    use_effort: bool = False

    # observation에 depth(turbo 컬러맵) 이미지 포함 여부. 카메라별 RealSense
    # use_depth(realsense_use_depth 등)가 켜진 카메라에 한해서만 적용됨 —
    # 이게 꺼져 있으면 depth 스트림 자체가 없어서 무시된다.
    use_depth_observation: bool = False
    depth_min_m: float = 0.20
    depth_max_m: float = 0.80
    # RealSense depth unit. D400 계열 기본 0.001(=1mm) — 실제 장비에서 재확인 필요.
    depth_scale: float = 0.001
    # turbo 컬러맵(8bit, dmax-dmin 범위를 256단계로 양자화)은 Evo-Depth IDEM의 보조
    # supervision으로 쓰기엔 정밀도가 부족할 수 있음(예: 0.6m 범위면 계단당 ~2.3mm).
    # 채워두면 raw uint16 depth(미터 변환 전, depth_scale 곱하기 전 원본)를 프레임마다
    # <depth_raw_dir>/<cam>/<session-global index>_<unix time>.npy로 추가 저장.
    # 세션 전체에 걸친 순번이라 에피소드별 frame_index와 정확히 대응하진 않음 —
    # 나중에 매칭하려면 파일명의 unix time을 데이터셋 타임스탬프와 맞출 것. 재수집이
    # 어려운 1회성 촬영이라 정밀도 부족이 드러난 뒤엔 늦으므로 미리 켜두는 걸 권장.
    depth_raw_dir: str = ""

    # 실시간 안전 컷오프(NEXT와 무관, use_effort OFF여도 항상 독립 동작).
    # 리플레이/정책 출력이 관절 명령으로 변환되어 로봇에 나가기 직전(send_action)에 검사.
    safety_enabled: bool = True
    # N·m. get_effort()가 current 기반 추정치라 절대 토크값이 아니므로, 처음엔 보수적으로
    # 잡고 랩 PC에서 자유운동(무접촉) 시 관측되는 effort 노이즈 상한을 본 뒤 튜닝할 것.
    safety_effort_limit: float = 8.0

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = 5.0

    # leader/follower 시작 자세 차이 보정
    use_action_offset: bool = True
    use_manual_action_offset: bool = False
    action_offset_warmup_s: float = 1.5
    action_offset_report_threshold: float = 3.0
    action_offset_joint1: float = 0.0
    action_offset_joint2: float = 0.0
    action_offset_joint3: float = 0.0
    action_offset_joint4: float = 0.0
    action_offset_joint5: float = 0.0
    action_offset_joint6: float = 0.0
    action_offset_gripper: float = 0.0

    def __post_init__(self) -> None:
        # 직접 넘긴 cameras 우선
        if self.cameras or not (self.top_cam or self.wrist_cam):
            return

        camera_type = self.camera_type.lower()
        top_cam_type = (self.top_cam_type or camera_type).lower()
        wrist_cam_type = (self.wrist_cam_type or camera_type).lower()

        if self.top_cam:
            self.cameras["top"] = self._make_camera_config(
                top_cam_type, self.top_cam, self.top_realsense_use_depth or self.realsense_use_depth
            )
        if self.wrist_cam:
            self.cameras["wrist"] = self._make_camera_config(
                wrist_cam_type, self.wrist_cam, self.wrist_realsense_use_depth or self.realsense_use_depth
            )

    def _make_camera_config(self, camera_type: str, value: str, use_depth: bool) -> CameraConfig:
        # 단순 CLI 값을 실제 CameraConfig로 변환
        if camera_type == "opencv":
            return OpenCVCameraConfig(
                index_or_path=_opencv_index_or_path(value),
                width=self.cam_width,
                height=self.cam_height,
                fps=self.camera_fps,
            )
        if camera_type in {"intelrealsense", "realsense"}:
            return RealSenseCameraConfig(
                serial_number_or_name=value,
                width=self.cam_width,
                height=self.cam_height,
                fps=self.camera_fps,
                use_depth=use_depth,
                warmup_s=self.realsense_warmup_s,
            )
        raise ValueError(f"Unsupported camera type '{camera_type}'. Use opencv or intelrealsense.")

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass(kw_only=True)
class PiperFollowerArmConfig:
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

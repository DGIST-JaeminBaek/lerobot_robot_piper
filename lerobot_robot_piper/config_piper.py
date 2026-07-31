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

    # observation에 관절별 effort(전류 기반 추정 토크, N·m) + velocity 포함 여부.
    # piper_sdk의 current-derived 값이라 자세에 따른 중력/마찰 성분이 섞여 있음(진짜 토크 센서 아님).
    # 기본 ON — 안 찍은 effort는 되살릴 수 없는 반면, 켜는 비용은 프레임당 52바이트에
    # 비디오 인코딩과 무관하다(observation.state가 pos7+effort7+vel6=20차원이 됨).
    # 주의: 학습한 데이터셋과 추론 시 이 값이 다르면 state 차원이 안 맞는다 —
    # 7차원으로 학습한 옛 체크포인트를 돌릴 때만 false로 내릴 것.
    use_effort: bool = True

    # 실시간 안전 컷오프(use_effort OFF여도 항상 독립 동작).
    # 리플레이/정책 출력이 관절 명령으로 변환되어 로봇에 나가기 직전(send_action)에 검사.
    safety_enabled: bool = True
    # N·m. get_effort()가 current 기반 추정치라 절대 토크값이 아니므로, 처음엔 보수적으로
    # 잡고 랩 PC에서 자유운동(무접촉) 시 관측되는 effort 노이즈 상한을 본 뒤 튜닝할 것.
    safety_effort_limit: float = 8.0
    # 임계값을 넘었을 때 무엇을 할지. 둘 다 리더/정책 명령은 무시하되 토크는 유지한다
    # (트립 자세를 계속 재전송 — safety_hold_resend 참고).
    #   "hold" — 트립 순간의 자세에서 그대로 정지.
    #   "park" — parking 자세로 천천히 복귀한 뒤 그 자세로 정지.
    safety_on_overload: str = "park"
    # "park"으로 복귀할 때 걸릴 시간(초). 0 이하면 기존 bus.parking()처럼 목표를
    # 한 번에 쏴서 컨트롤러 최고속으로 이동한다 — parking 자세가 "팔이 수직으로
    # 뻗은" 자세라서 그렇게 하면 트립 순간 팔이 확 뻗는 것처럼 보이고, 사람 손이
    # 팔에 닿아 있을 수 있는 상황(외력으로 트립된 직후)에서 위험하다. 그래서
    # 기본은 ramp_to로 천천히 복귀.
    safety_park_ramp_s: float = 4.0
    # 트립 후 "얼어붙힌 자세"를 매 스텝 다시 보낼지. Piper는 JointCtrl 목표를 계속
    # 스트리밍해야 자세를 잡으므로, 끄면(=명령 완전 중단) 팔이 힘을 잃고 늘어진다.
    # 늘어짐의 원인이 우리 쪽인지 컨트롤러 자체 보호인지 가려낼 때만 false로 둘 것.
    safety_hold_resend: bool = True

    # torque 해제 방식 (motors/piper_motors_bus.py release_torque_safely 참고).
    #   "in_place" — 이동 없이 그 자리에서 해제
    #   "lower"    — 팔은 그대로 두고 손목(joint5)만 미리 내린 뒤 해제 (기본)
    #   "park"     — 기존 동작: parking 자세로 이동 후 해제
    # 실기 측정 결과 torque를 풀 때 실제로 떨어지는 건 손목뿐이고(joint1~4/6은
    # 0.00도) 그 낙차가 24.4도였다 — 미리 내려두면 0.6도로 줄어든다. 팔을 옮기지
    # 않으므로 lower가 기본값이어도 이동 위험이 없다(tables.py 주석 참고).
    park_release_mode: str = "lower"
    park_release_ramp_s: float = 2.0
    park_release_settle_s: float = 0.5
    # "lower"에서 손목을 내려둘 각도(도) — 상대 델타가 아니라 절대 각도(자연 정지각).
    # 기본값은 motors/tables.py의 WRIST_RELEASE_REST_DEG와 동일 — 여기 그대로 적어둔
    # 이유는 config가 piper_sdk를 import하는 motors 패키지에 의존하지 않게 하기 위함.
    park_release_wrist_rest_deg: float = 24.4
    # torque를 풀기 전에 그리퍼를 한 번 열고 닫아서 물고 있던 것을 놓고 파킹
    # 위치(닫힘)로 되돌릴지. 그리퍼는 팔 모터와 별개 노드(0x159)라 DisablePiper()로
    # 풀리지 않으므로, 해제 시 실능(GripperCtrl code=0x00)은 이 값과 무관하게 항상 한다.
    # 주의: 여는 순간 잡고 있던 물체가 떨어지고 닫을 때 손가락이 끼일 수 있다.
    park_release_gripper_cycle: bool = True
    park_release_gripper_open: float = 100.0
    park_release_gripper_wait_s: float = 1.5

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

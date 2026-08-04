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

    # 종료 시 어떤 자세로 가서 torque를 풀지 (motors/piper_motors_bus.py의
    # release_torque_safely 참고).
    #   "in_place"   — 이동 없이 그 자리에서 해제
    #   "lower"      — 팔은 그대로 두고 손목(joint5)만 미리 내린 뒤 해제
    #   "park"       — parking 자세로 이동 후 해제 (손목 낙차는 남음)
    #   "park_lower" — parking 자세로 간 뒤 거기서 손목까지 내리고 해제 (기본)
    #
    # 실기 측정 결과 torque를 풀 때 실제로 떨어지는 건 손목뿐이고(joint1~4/6은
    # 0.00도) 그 낙차가 24.4도였다 — 미리 내려두면 0.6도로 줄어든다(tables.py 주석).
    # 예전 기본값은 "lower"였는데 그러면 팔이 있던 자리에 그대로 늘어져서, 추론이
    # 끝난 위치(보드 앞 등)에 팔이 남았다. 보관 자세로 돌아간 뒤 손목을 내리는
    # "park_lower"가 둘을 다 만족한다.
    park_release_mode: str = "park_lower"
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

    # ModeCtrl(0x151)의 MOVE 모드. set_action()이 관절 명령마다 함께 보낸다.
    #   1 = MOVE J   점대점 이동. 목표마다 컨트롤러가 가속/감속 궤적을 새로 계획.
    #   5 = MOVE CPV 연속 위치-속도. 스트리밍 setpoint용. 펌웨어 V1.8-1 이상.
    #
    # 기본값은 MOVE J다 — 텔레옵/녹화/파킹은 이걸로 잘 동작하고 있어 건드리지 않는다.
    # 정책 추론처럼 30Hz로 목표를 계속 흘려보낼 때만 문제가 된다: 33ms마다 새
    # 점대점 명령이 들어오면 컨트롤러가 초당 30번 궤적을 다시 계획해서 팔이 떤다.
    #
    # 2026-08-04 실물 시험: MOVE CPV(5)로 바꿨더니 팔이 전혀 움직이지 않았다
    # (명령은 나가지만 실측 위치가 제자리 — 클램프가 100% 포화). piper_sdk의
    # JointCtrl은 위치만 보내는데 CPV는 속도 setpoint까지 필요한 것으로 보이고,
    # SDK에 그걸 보내는 API가 없다. 그래서 5는 현재 쓸 수 없다 — 30Hz 스트리밍의
    # 재계획 문제는 다른 방법으로 풀어야 한다.
    move_mode: int = 1
    # ModeCtrl의 속도 백분율(0~100). 팔의 물리적 최고 속도 상한.
    move_speed_rate: int = 30

    # MIT(임피던스) 제어. 켜면 set_action()이 JointCtrl 대신 JointMitCtrl을 쓴다.
    # 궤적 재계획이 없어 30Hz 스트리밍에 맞고, 정책 chunk에서 뽑은 속도를 함께
    # 넘길 수 있다(위치 제어는 속도 setpoint를 못 받는다).
    #
    # ⚠ 토크 제어다. kp가 낮으면 팔이 중력에 무너지고 높으면 진동한다.
    # max_relative_target(위치 명령 클램프)과 effort 컷오프는 위치 제어를 전제로
    # 만들어진 것이라 여기서는 의미가 달라진다. 기본 꺼짐 — 켜기 전에
    # scripts/tools/piper_mit_probe.py로 관절 하나씩 확인할 것.
    use_mit_control: bool = False
    mit_kp: float = 10.0   # SDK 참고값 — 전 관절 공통 기본값
    mit_kd: float = 0.8    # SDK 참고값
    # 관절별 kp 덮어쓰기. "joint2=30,joint3=20" 형식 (LeRobot의 dict CLI 파서를
    # 피하려고 문자열로 둔다 — 이 파일의 다른 카메라 필드와 같은 이유).
    #
    # 왜 필요한가: MIT의 정상상태 오차 = 중력토크 / kp 이므로, 중력 부담이 다른
    # 관절에 같은 kp를 주면 처짐이 제각각이 된다. 실측(kp=10, 뻗은 자세):
    #   joint1/4/6  처짐 0.00     joint5  0.17
    #   joint2      처짐 1.83(≈1.65°)     joint3  0.72(≈0.61°)
    # 무거운 두 관절만 kp를 올려 처짐을 맞춘다.
    mit_kp_overrides: str = "joint2=30,joint3=20"
    mit_kd_overrides: str = ""

    # send_action() 단계의 EMA 스무딩. 1.0이면 꺼짐(원본 목표 그대로) — 기본값은
    # 꺼짐이라 텔레옵/replay 동작은 그대로다. 0에 가까울수록 더 부드럽고 더 느리게
    # 따라간다: smoothed = alpha*target + (1-alpha)*prev.
    #
    # 이건 "바닥"일 뿐이다. 추론의 주된 스무딩은 temporal ensemble인데, 그건 action
    # chunk 단위라 여기(한 스텝씩 받는 자리)서는 불가능하고 piper_infer_runner.py가
    # 처리한다. 실측상 total variation을 절반으로 줄인 것도 ensemble 쪽이다
    # (docs/policy/smoothing.md). 여기 EMA는 chunk가 없는 경로 — lerobot-record로
    # 도는 Record나 lerobot-replay — 에서 고주파 노이즈를 깎는 용도다.
    #
    # rate limit은 따로 두지 않는다 — 아래 max_relative_target이 이미 스텝당 최대
    # 변화량을 제한하고 있어서 중복이다.
    action_ema_alpha: float = 1.0

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

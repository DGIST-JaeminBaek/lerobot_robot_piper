#!/usr/bin/env python3
"""
piper_replay_record.py — 기존 LeRobotDataset의 저장된 액션(예: 합성/생성된 궤적)을
실제 follower 로봇에 재생하면서, 그 동안의 카메라+관절 관측을 새 LeRobotDataset으로
녹화하는 도구.

배경: lerobot 자체의 `lerobot-replay`는 액션을 로봇에 보내기만 하고 아무것도
저장하지 않고, `lerobot-record`는 teleop이나 policy로부터만 액션을 받을 수 있어
"저장된 액션을 그대로 재생 + 새 관측을 녹화"하는 경로가 없음. 이 스크립트는
lerobot/scripts/lerobot_record.py의 record_loop()과 lerobot_replay.py의 액션
로딩 방식을 그대로 따르되, 액션 소스를 teleop/policy 대신 source dataset의
저장된 액션으로 바꿔서 둘을 합친 것.

주의:
- source dataset의 fps와 output dataset의 fps는 항상 동일하게 맞춤(= source의
  fps를 그대로 씀) — 서로 다르면 프레임간 간격이 원본 녹화 당시와 달라져서
  PiperFollower의 max_relative_target 클리핑이 매 프레임 걸릴 수 있음.
- PiperFollower.send_action()의 action offset(첫 프레임에서 "현재 위치 - 첫
  액션" 차이를 1회 계산해 이후 프레임에 동일하게 더함)이 그대로 적용되므로,
  로봇이 source 궤적의 시작 자세와 다른 곳에 있어도 급격한 스냅 없이 자연스럽게
  궤적을 따라감. 안전을 위해 max_relative_target 클리핑도 항상 적용됨.
- source episode 1개 = output dataset 1개(episode 1개)를 녹화. 여러 episode를
  합성 궤적에서 뽑아 반복 수확하려면 --source_dataset.episode를 바꿔가며 여러 번
  실행 (teleop_ui.py의 Dataset Browser에서 episode를 골라 재실행하는 것과 동일한
  흐름).

사용법 (lerobot-record/lerobot-replay와 동일한 CLI 스타일):
    python3 piper_replay_record.py \\
        --robot.type=piper_follower --robot.port=can_follower \\
        --robot.top_cam=0 --robot.wrist_cam=1 \\
        --source_dataset.repo_id=local/synth_traj --source_dataset.root=records/local/synth_traj \\
        --source_dataset.episode=0 \\
        --dataset.repo_id=local/replay_record_out --dataset.root=records/local/replay_record_out \\
        --dataset.single_task="pick the cube" \\
        --display_data=true \\
        --robot.discover_packages_path=lerobot_robot_piper
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class SourceDatasetConfig:
    # 재생할 액션이 저장된 소스 데이터셋
    repo_id: str
    episode: int
    root: str | Path | None = None


@dataclass
class OutputDatasetConfig:
    # 새로 녹화될 출력 데이터셋
    repo_id: str
    single_task: str
    root: str | Path | None = None
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self):
        if not self.single_task:
            raise ValueError("You need to provide a task as argument in `dataset.single_task`.")


@dataclass
class ReplayRecordConfig:
    robot: RobotConfig
    source_dataset: SourceDatasetConfig
    dataset: OutputDatasetConfig
    display_data: bool = False
    play_sounds: bool = True


@parser.wrap()
def replay_record(cfg: ReplayRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="replay_record")

    robot = make_robot_from_config(cfg.robot)

    # teleop/policy가 없으므로 전부 IdentityProcessor — record_loop()과 동일한 조합
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    logger.info(f"소스 데이터셋 로드 중: {cfg.source_dataset.repo_id} (episode {cfg.source_dataset.episode})")
    source = LeRobotDataset(
        cfg.source_dataset.repo_id, root=cfg.source_dataset.root, episodes=[cfg.source_dataset.episode]
    )
    # episode가 chunk로 나뉘어 있을 수 있어 명시적으로 필터 (lerobot_replay.py와 동일)
    episode_frames = source.hf_dataset.filter(lambda x: x["episode_index"] == cfg.source_dataset.episode)
    actions_col = episode_frames.select_columns(ACTION)
    action_names = source.features[ACTION]["names"]
    fps = source.fps
    logger.info(f"{len(episode_frames)} frame, fps={fps} (source 기준 — output도 동일하게 맞춤)")

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    robot.connect()

    try:
        with VideoEncodingManager(dataset):
            log_say("Replay-record episode", cfg.play_sounds, blocking=True)
            for idx in range(len(episode_frames)):
                start_t = time.perf_counter()

                action_array = actions_col[idx][ACTION]
                source_action = {name: action_array[i] for i, name in enumerate(action_names)}

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

                robot_action_to_send = robot_action_processor((source_action, obs))
                # 클리핑/오프셋 등이 적용된 뒤 실제로 로봇에 보낸 액션을 저장
                # (record_loop()과 동일한 원칙: 저장값 == 실제 전송값)
                sent_action = robot.send_action(robot_action_to_send)

                action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": cfg.dataset.single_task}
                dataset.add_frame(frame)

                if cfg.display_data:
                    log_rerun_data(observation=obs_processed, action=sent_action)

                dt_s = time.perf_counter() - start_t
                busy_wait(1 / fps - dt_s)

                if (idx + 1) % 50 == 0 or idx == len(episode_frames) - 1:
                    logger.info(f"  {idx + 1}/{len(episode_frames)} frame 재생-녹화 완료")

            dataset.save_episode()
    finally:
        robot.disconnect()

    log_say("Replay-record 완료", cfg.play_sounds)

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main():
    register_third_party_devices()
    replay_record()


if __name__ == "__main__":
    main()

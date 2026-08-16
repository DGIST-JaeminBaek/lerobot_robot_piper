#!/usr/bin/env python3
"""piper_record_one.py — 에피소드 딱 1개만 녹화하는 전용 스크립트.

lerobot-record(공용 lerobot_record.py)는 건드리지 않고, RecordConfig/
DatasetRecordConfig/record_loop()을 그대로 재사용해서 별도로 만든 것 — 여러
로봇 타입이 공유하는 lerobot 핵심 스크립트에 Piper 전용 동작(조기 종료 시
parking 생략)을 섞고 싶지 않아서 분리함.

동작:
    - episode_time_s 타이머가 다 돼서 자연 종료되면: parking 이동 후 disconnect
    - "End Episode" 핫키(exit_early, 오른쪽 화살표)나 Esc(stop_recording)로
      사람이 조기 종료시키면: parking 없이 그 자리에서 바로 disconnect
    - 왼쪽 화살표(rerecord_episode)로 취소하면: 그 에피소드는 저장 안 하고
      역시 parking 없이 종료 (재시도 루프는 없음 — 이 스크립트는 1회성)
    - Ctrl+C/SIGINT로 죽으면: 안전하게 기본값(parking) 유지

--robot.*/--teleop.*/--dataset.* 인자는 lerobot-record와 100% 동일(같은
RecordConfig를 그대로 파싱). --dataset.num_episodes/reset_time_s/resume 값은
읽기는 하되 이 스크립트 로직에서는 안 씀(항상 정확히 1개만, reset 구간 없이).

사용법:
    python scripts/tools/piper_record_one.py \\
        --robot.type=piper_follower --robot.port=can_follower1 \\
        --teleop.type=piper_leader --teleop.port=can_leader1 \\
        --dataset.repo_id=local/test --dataset.root=/path/to/root \\
        --dataset.single_task="pick up the pen" --dataset.episode_time_s=30 \\
        --robot.discover_packages_path=lerobot_robot_piper \\
        --teleop.discover_packages_path=lerobot_robot_piper
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, sanity_check_dataset_name
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


@parser.wrap()
def record_one(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
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

    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        encoder_threads=cfg.dataset.encoder_threads,
    )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    # 예외/SIGINT로 죽는 등 정상적으로 record_loop()을 못 빠져나온 경우엔
    # 안전하게 parking(기본 동작)을 유지 — 아래에서 record_loop()이 정상
    # 리턴했을 때만 실제 종료 사유를 보고 이 값을 재계산함.
    park = True
    saved = False

    try:
        with VideoEncodingManager(dataset):
            log_say("Recording episode", cfg.play_sounds)
            start_t = time.perf_counter()
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )
            elapsed_s = time.perf_counter() - start_t

            # exit_early(End Episode 핫키/Esc)로 조기 종료되면 record_loop()이
            # episode_time_s보다 눈에 띄게 일찍 리턴함 — 1초 여유를 두고 판단.
            # 왼쪽 화살표(rerecord_episode)로 취소한 경우도 사람이 개입한 조기
            # 종료라 parking 생략 대상으로 취급.
            timer_finished_naturally = elapsed_s >= cfg.dataset.episode_time_s - 1.0
            park = timer_finished_naturally and not events["rerecord_episode"]

            if events["rerecord_episode"]:
                log_say("Discarding episode", cfg.play_sounds)
                dataset.clear_episode_buffer()
            else:
                dataset.save_episode()
                saved = True
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if robot.is_connected:
            robot.disconnect(park=park)
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

    if saved and cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record_one()


if __name__ == "__main__":
    main()

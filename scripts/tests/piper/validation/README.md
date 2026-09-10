# Replay 검증 테스트

`piper_replay_player.py`의 일반/RViz CLI 경계와 Piper 관절값의 ROS 단위 변환을
하드웨어와 ROS2 없이 확인합니다.

`test_effort_common.py`는 effort dataset schema와 smooth-start 오염 판정을 두 effort
CLI가 같은 기준으로 쓰는지 확인합니다.

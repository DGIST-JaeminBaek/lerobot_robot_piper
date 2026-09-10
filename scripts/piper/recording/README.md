# 범용 녹화 도구

이 폴더에는 특정 작업의 장면이나 성공 기준과 무관하게, Piper 조작 데이터를
녹화하고 데이터셋을 보정하는 도구를 둡니다.

- `piper_record_one.py`: GUI에서 호출하는 단일 episode 녹화기입니다.
- `smooth_start_frames.py`: 녹화 초반의 위치만 parking 자세에서 보간하도록 보정합니다.
  effort·velocity 같은 측정값은 수정하지 않습니다.
- `fix_action_offset.py`: 녹화 action의 시간 지연·offset을 분석해 새 데이터셋으로 보정합니다.
- `retask_dataset.py`: 영상·state·action을 바꾸지 않고 단일-task 데이터셋의 task 문구만 바꿉니다.

특정 task를 위해 데이터셋을 합치거나 라벨을 생성하는 도구는
`scripts/tasks/<task>/dataset/`에 둡니다.

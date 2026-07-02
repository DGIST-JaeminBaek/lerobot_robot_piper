# 데이터 수집 프로토콜

Task 문장: `write AIIII`

## Episode 시작 조건

1. 화이트보드 위치 고정
2. 펜 시작 위치 고정 또는 표시
3. 상방 카메라가 보드 전체, 펜, follower 팔을 볼 수 있는지 확인
4. 팔목 카메라가 펜촉과 보드 접촉 영역을 볼 수 있는지 확인
5. leader와 follower를 같은 home pose에서 시작

## Episode 내용

각 episode에 포함할 동작:

1. 펜 위치로 이동
2. 펜 잡기
3. 글씨 시작 위치로 이동
4. `AIIII` 쓰기
5. 펜을 보드에서 떼기
6. 안전한 종료 자세로 이동

## 품질 확인

- 상방 카메라에서 `AIIII` 글자가 보이는지 확인
- 팔목 카메라에서 펜촉 접촉 장면이 심하게 흔들리지 않는지 확인
- episode 시작 시 follower가 튀지 않는지 확인
- 실패한 시연은 따로 표시하거나 일관되게 폐기

## 첫 수집 목표

처음에는 5 episode만 수집.

큰 데이터셋을 만들기 전에 다음 항목 확인:

- 영상 저장 여부
- `observation.state` 저장 여부
- `action` 저장 여부
- `task`가 `write AIIII`로 저장되었는지 여부

```bash
python3 scripts/wego_dataset_check.py --dataset-repo-id your_hf_name/piper_write_light --episode 0
```

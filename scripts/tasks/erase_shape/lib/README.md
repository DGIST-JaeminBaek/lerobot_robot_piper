# 도형 지우기 공용 라이브러리

이 폴더에는 도형 지우기 task의 여러 하위 범주에서 함께 쓰지만, Piper 범용 코드로
승격할 만큼 다른 task에서 재사용되지는 않은 보조 모듈을 둡니다.

- `plot_ko.py`: 헤드리스 Matplotlib에서 한글 폰트와 마이너스 기호를 올바르게 설정하는
  plotting 보조 모듈입니다. 분석·평가 도구가 공통으로 사용합니다.
- `ink_metric.py`: 보드·도형·잉크·가림을 측정하는 task 공용 영상 분석 로직입니다.
  runtime 종료 게이트, dataset done-label, analysis, 사후 평가가 함께 사용합니다.

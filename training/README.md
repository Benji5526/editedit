# 학습 프로그램 (탐지 모델 추가 학습)

모자이크 앱과 따로 도는 스크립트 묶음이다. 앱은 여기서 만든 모델 파일을 "탐지 모델 파일" 칸에서 고르기만 한다.
모든 과정은 이 PC 안에서만 돈다. 학습 도구(Ultralytics)는 온라인 확인·다운로드·통계 전송을 끈 상태로 실행한다(`common.py`).

## 준비 (한 번)

학습 도구는 앱 환경(`.venv`)이 아니라 학습 전용 환경(`.venv-train`)에 설치한다. PyTorch·OpenCV는 앱 환경 것을 같이 쓴다
(`.venv-train/Lib/site-packages/_app_env.pth`가 앱 환경의 패키지 폴더를 가리킨다).

    <기본 파이썬> -m venv .venv-train
    (위 .pth 파일 만들기)
    .venv-train\Scripts\python.exe -m pip install ultralytics onnx

Ultralytics는 AGPL-3.0이다. 이 폴더의 스크립트와 여기서 만든 모델을 남에게 배포하면 그 조건이 따라온다.

## 순서

1. **정답 만들기**: 원본과, 그 원본을 이미 모자이크한 영상의 짝에서 "가려진 자리"를 뽑는다.

       .venv\Scripts\python.exe tools\pair_labels.py 원본.mp4 모자이크본.mp4

   출력된 요약을 먼저 본다. `background_difference`가 크면(대략 6 이상) 두 영상이 잘 안 맞는 것이고,
   `static_places`에 뭔가 있으면 로고·자막이거나 움직이지 않는 대상이다(로고가 맞을 때만 `--drop-static`).

2. **학습 자료 만들기**: 원본에서 프레임을 뽑아 이미지와 정답 파일로 만든다.

       .venv\Scripts\python.exe training\make_dataset.py 이름 --video 원본1.mp4 --video 원본2.mp4

   `training/data/이름/`에 생긴다. 영상의 프레임이 그대로 들어 있으므로 이 폴더는 git에 올리지 않는다.

3. **학습**:

       .venv-train\Scripts\python.exe training\train.py 이름 --base 기본가중치.pt --epochs 100

   `--base`는 이미 이 PC에 있는 가중치 파일이어야 한다(자동으로 받지 않는다). 가중치 없이 `--base yolov8n.yaml`로도
   돌릴 수 있지만 자료가 아주 많아야 쓸 만해진다. 결과는 `training/runs/이름/train/`의 `model.onnx`,
   `model.labels.json`(클래스 이름과 입력 크기, 앱이 onnx 옆에서 읽는다), `report.json`.

4. **비교**: 앱이 쓰는 탐지 코드로 검증용 프레임을 돌려 놓친 곳과 쓸데없이 잡은 곳을 센다.

       .venv\Scripts\python.exe training\evaluate.py 이름 training\runs\이름\train\model.onnx --standard

   놓친 곳(`missed_regions`)이 줄고 쓸데없는 상자(`extra_boxes`)가 크게 늘지 않았을 때만 새 모델을 쓴다.

5. **앱에서 쓰기**: `model.onnx`와 `model.labels.json`을 같은 폴더(예: `models/`)에 같은 이름으로 두고
   (`내모델.onnx`, `내모델.labels.json`), 처리 설정 창의 "탐지 모델 파일"에서 고른다.

## 알아 둘 것

- 클래스는 `censor` 하나다. 짝 비교는 "가려졌다"는 것만 알려 주고 그것이 얼굴인지 성기인지 체액인지는 알려 주지 않는다.
- 여기서 만든 모델에는 얼굴 클래스가 없다. 이 모델을 고르면 "얼굴" 옵션은 꺼야 한다. 표준 NudeNet 모델과 이 모델을
  한 작업에서 같이 돌리는 기능은 아직 없다.
- 검증용 프레임은 영상마다 시간 구간째로 떼어 낸다. 그래도 같은 영상에서 나온 것이므로, 수치는 "이 자료에서"의 성능이다.
  새 영상에서의 성능은 그 영상으로 직접 확인해야 한다.
- 영상 한두 개로 학습하면 그 영상에만 맞는 모델이 된다.
- 이 프로그램이 스스로 만든 모자이크 결과를 짝으로 넣어 학습하면 안 된다(자기 실수를 그대로 배운다). 사람이 직접
  가렸거나 검수해서 고친 결과만 쓴다.

# 딸기 병해 진단 (Strawberry Disease Detection)

딸기 잎·과실 사진을 보고 병해를 진단하는 2단계 AI 파이프라인입니다.
YOLO로 잎·과실 위치를 먼저 찾고, 잘라낸 영역을 EfficientNet-B0로 분류합니다.

멋쟁이사자처럼 CV 부트캠프 팀 프로젝트 · 2026.01.09 ~ 01.15 (7일) · 4명 · Google Colab

<br>

## 파이프라인

```
이미지 → [YOLO] 잎·과실 탐지 → 영역 잘라내기 → [EfficientNet-B0] 병해 분류 → 결과 표시
```

분류 클래스 5종: 정상, 잎끝마름, 역병, 시들음병, 황화

<br>

## 결과

**분류 (EfficientNet-B0)**

정상 이미지에 비해 병해 이미지가 훨씬 적어서, 학습 때 병해 클래스에 가중치를 주고 Test는 클래스별 81장씩(총 405장) 같은 수로 맞춰 평가했습니다.

| Accuracy | Macro F1 | Precision | Recall |
|---|---|---|---|
| 83.5% | 0.833 | 0.838 | 0.835 |

클래스별 F1은 시들음병 0.92, 역병 0.90, 정상 0.84, 잎끝마름 0.77, 황화 0.72였습니다.

**탐지**

| 실험 | 모델 | mAP50 | mAP50-95 |
|---|---|---|---|
| 기본 학습 | YOLO11s | 0.496 | 0.348 |
| 어텐션 추가 | YOLOv8s + Triplet Attention (KTD-YOLOv8) | 0.772 | 0.542 |

실험마다 평가 이미지 구성이 조금씩 달라 두 수치는 참고용 비교입니다. `inference.py`는 YOLO11s 가중치를 사용합니다.

<br>

## 내가 맡은 부분

탐지 모델(YOLO) 실험과 2단계 파이프라인 구축을 맡았습니다.

- **데이터 준비**: AI Hub 딸기 데이터가 Colab 디스크에 다 들어가지 않아 Validation 세트만 골라 학습용으로 재구성
- **모델 비교**: YOLOv5·v8·11을 같은 데이터로 돌려 보고, 속도와 성능이 무난한 YOLO11s를 기본 모델로 선택
- **불균형 대응**: mixup·mosaic·색감(HSV) 증강, 에폭 조정을 해도 mAP50이 0.4 근처에서 멈춤 → 데이터가 적은 황화·잎끝마름 클래스를 빼고 학습하는 실험으로 원인 확인
- **해상도 실험**: 얇은 줄기를 잘 못 찾아 입력 해상도를 1024로 올린 YOLO11m 실험
- **메모리 제약**: Colab RAM·디스크 한계 안에서 끊기지 않게 학습하도록 캐시·worker·batch 설정 조정
- **파이프라인**: YOLO가 찾은 영역을 잘라 분류 모델에 넘기는 추론 파이프라인(`inference.py`) 구축

<br>

## 배운 점

- 모델이나 하이퍼파라미터보다 데이터의 양과 질이 성능을 더 크게 좌우했습니다.
- 7일짜리 협업에서는 처음에 전체 개요와 일자별 목표를 정하고, 팀원 진행 상황을 계속 공유받아야 일이 꼬이지 않았습니다.

<br>

## 파일 구성

| 파일 | 내용 |
|---|---|
| `inference.py` | YOLO + EfficientNet 추론 파이프라인 |
| `smartfarm_strawberry_aihub_2_0_training_detection.ipynb` | AI Hub 데이터 전처리, YOLO11s 탐지 모델 학습 |
| `smartfarm_strawberry_KTD_YOLOv8.ipynb` | Triplet Attention을 넣은 YOLOv8 탐지 모델 학습 |
| `smartfarm_strawberry_aihub_with_unzip_res_change.ipynb` | 입력 해상도 변경 실험 |
| `클래스_제거_smartfarm_strawberry_aihub_with_unzip.ipynb` | 클래스 축소 실험 |
| `proj1_smartfarm_2step_classification_augmentation.ipynb` | 분류 모델 비교(ResNet50, EfficientNet-B0), 데이터 증강, 최종 분류 모델 |

<br>

## 실행

```bash
pip install -r requirements.txt
python inference.py   # 파일 상단의 모델·데이터 경로를 환경에 맞게 수정
```

데이터: AI Hub 딸기 병해 이미지

## 팀

[@LegenDUST](https://github.com/LegenDUST) · [@jueuni135-spec](https://github.com/jueuni135-spec) · [@jueun27](https://github.com/jueun27) · [@Kwakjaemin1007](https://github.com/Kwakjaemin1007)

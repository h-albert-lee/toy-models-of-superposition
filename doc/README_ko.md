# Toy Models of Superposition 문서

이 문서는 레포지토리의 구성과 주요 모듈, 실행 방법을 상세히 설명합니다. 연구자는 이 자료를 기반으로 독성 개념(superposition)의 특성을 실험적으로 탐구할 수 있습니다.

## 1. 레포지토리 개요

이 프로젝트는 시각-언어 모델(Vision-Language Model, VLM) 내부 표현 공간에서 독성(Toxic) 특징이 다른 개념과 겹쳐(superposed) 존재한다는 가설을 검증하기 위한 도구를 제공합니다. LLaVA 계열과 같은 사전 학습된 VLM을 대상으로 활성화 벡터를 추출하고, 특정 방향을 제거하거나 추가하는 방식으로 모델의 반응 변화를 관찰합니다.

## 2. 디렉터리 구조

```
configs/                  # 실험을 제어하는 YAML 설정 파일
  base_config.yaml
data/                     # JSONL 형식의 예시 데이터
scripts/                  # 커맨드라인 실행 스크립트
src/                      # 핵심 파이썬 모듈
  vlm_wrapper.py          # 모델 로딩 및 활성화 추출, 개입 기능
  feature_extractor.py    # GTV/ITV 계산을 위한 고수준 유틸리티
  interventions.py        # 벡터 연산(예: projection subtraction)
  ...
tests/                    # pytest 단위 테스트
```

## 3. 설정 파일(`configs/base_config.yaml`)

설정 파일은 모델 정보와 데이터 경로, 배치 크기 등을 한 곳에서 관리합니다.

| 키 | 설명 |
|----|------|
| `model_name` | 사용하고자 하는 Hugging Face 모델 ID |
| `device` | 모델을 올릴 디바이스 (`cuda`/`cpu`) |
| `model_kwargs` | `from_pretrained` 에 전달할 추가 인자 |
| `layers` | 여러 레이어를 순회하며 평가할 레이어 목록 |
| `extraction_layer` | 활성화를 추출할 레이어 이름 |
| `batch_size` | `VectorExtractor` 가 한 번에 처리할 배치 크기 |
| `data_sources.toxic_neutral_pairs` | 일반 독성 벡터 계산용 텍스트 페어 파일 |
| `data_sources.intrinsic_toxicity_cases` | 내재 독성 벡터 계산용 멀티모달 사례 파일 |
| `intervention_layer` | 개입(intervention)을 적용할 레이어 |
| `intervention_multiplier` | 개입 강도 조절 인자 |
| `output_dir` | 결과 파일이 저장될 디렉터리 |

## 4. 핵심 모듈 설명

### 4.1 `src/vlm_wrapper.py`
- Hugging Face 모델을 로딩하고 `_register_hooks` 로 특정 레이어의 활성화를 저장합니다.
- `get_activations` 는 텍스트와 이미지를 **리스트 형태로 입력**받아 배치 처리하며, 결과는 레이어 이름을 키로 하는 딕셔너리로 반환됩니다.
- `generate_with_intervention` 을 통해 지정한 레이어에 개입 함수를 후크로 걸어 출력 텍스트를 생성할 수 있습니다.

### 4.2 `src/feature_extractor.py`
- `VectorExtractor` 는 `compute_gtv` 와 `compute_itv` 두 메서드를 제공합니다.
- 두 메서드 모두 설정된 `batch_size` 단위로 데이터를 나누어 `VLM_Wrapper` 를 호출하므로, 대규모 데이터도 효율적으로 처리할 수 있습니다.
- 로깅을 통해 각 배치가 처리되는 시점을 확인할 수 있습니다.

### 4.3 `src/interventions.py`
- 현재는 `vector_subtraction` 함수 하나를 제공하며, 주어진 방향 벡터를 활성화에서 제거합니다.

## 5. 실행 스크립트 흐름

1. **`00_define_general_toxicity_vector.py`**
   - `data_sources.toxic_neutral_pairs` 에서 텍스트 페어를 읽어 일반 독성 벡터(GTV)를 계산합니다.
2. **`01_define_intrinsic_toxicity_vector.py`**
   - 멀티모달 사례를 바탕으로 내재 독성 벡터(ITV)를 산출합니다.
3. **`03_analyze_vector_similarity.py`**
   - GTV와 ITV의 코사인 유사도를 계산하여 두 벡터의 관계를 분석합니다.
4. **`04_evaluate_precision_intervention.py`**
   - 독성 분류기를 이용해 개입 전후의 텍스트 독성 점수를 비교합니다.
5. **`05_run_multi_layer_experiments.py`**
   - 여러 레이어를 순회하며 유사도를 기록하고 가장 높은 레이어에 대해 정밀 개입 평가를 수행합니다.

## 6. 배치 처리와 성능

기존 구현은 데이터 항목마다 모델을 반복 호출하여 비효율적이었으나, 현재 버전에서는 `get_activations` 가 입력 리스트를 한 번에 처리합니다. 이로 인해 GPU/CPU 활용률이 개선되고 전체 실행 시간이 크게 감소합니다.

## 7. 로깅

모든 스크립트와 주요 모듈은 Python `logging` 모듈을 사용합니다. 기본 설정은 INFO 레벨이며, 필요에 따라 `logging.basicConfig(level=logging.DEBUG)` 와 같이 조정할 수 있습니다.

## 8. 테스트

`tests/` 디렉터리에는 벡터 연산 및 배치 기반 추출 로직을 검증하는 pytest 스위트가 포함되어 있습니다. 다음 명령으로 실행합니다.

```bash
pytest
```

테스트는 코드 변경 시 회귀(regression)를 방지하는 최소한의 안전장치를 제공합니다.

## 9. 추가 자료

- `notebooks/` 디렉터리에는 실험 결과를 시각적으로 탐색할 수 있는 주피터 노트북이 포함되어 있습니다.
- `requirements.txt` 를 참고하여 필요한 라이브러리를 설치하십시오.

## 10. 라이선스

본 프로젝트는 [MIT 라이선스](../LICENSE) 하에 배포됩니다.


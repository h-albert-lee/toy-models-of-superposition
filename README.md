# Toy Models of Superposition

This repository provides a small research toolkit for studying how toxic and benign concepts may be superposed in the latent space of vision–language models (VLMs).

## 주요 기능

- **설정 파일 기반 실험 관리**: `configs/base_config.yaml` 파일을 통해 모델 이름, 추출 레이어, 배치 크기, 데이터 경로 등을 손쉽게 변경할 수 있습니다.
- **배치 단위 활성화 추출**: `VLM_Wrapper.get_activations` 와 `VectorExtractor` 가 텍스트/이미지를 배치로 처리하여 GPU 활용률을 높입니다.
- **구조화된 로깅**: 모든 스크립트는 `logging` 모듈을 사용하여 진행 상황을 정보 레벨로 기록합니다.
- **단위 테스트**: `tests/` 디렉터리의 pytest 스위트를 통해 핵심 벡터 연산과 특성 추출 로직을 검증할 수 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

1. **일반 독성 벡터(GTV) 계산**
   ```bash
   python scripts/00_define_general_toxicity_vector.py --config configs/base_config.yaml
   ```
2. **내재 독성 벡터(ITV) 계산**
   ```bash
   python scripts/01_define_intrinsic_toxicity_vector.py --config configs/base_config.yaml
   ```
3. **벡터 유사도 분석**
   ```bash
   python scripts/03_analyze_vector_similarity.py --config configs/base_config.yaml
   ```
4. **정밀 개입 성능 평가**
   ```bash
   python scripts/04_evaluate_precision_intervention.py --config configs/base_config.yaml
   ```

모든 스크립트는 설정 파일을 통해 모델과 데이터 경로를 지정하며, 실행 결과는 `output_dir` 로 지정된 디렉터리에 저장됩니다.

## 설정 파일 예시

```yaml
model_name: "llava-hf/llava-1.5-7b-hf"
device: "cuda"
model_kwargs: {}
extraction_layer: "language_model.layers.0"
batch_size: 8
data_sources:
  toxic_neutral_pairs: "data/general_toxic_neutral_text_pairs.jsonl"
  intrinsic_toxicity_cases: "data/intrinsic_toxicity_cases.jsonl"
intervention_layer: "language_model.layers.0"
intervention_multiplier: 1.0
output_dir: "results"
```

## 개발 및 테스트

```bash
pytest
```

## 라이선스

[MIT](LICENSE)

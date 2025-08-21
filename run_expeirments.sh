#!/bin/bash

# =================================================================
# VLM 독성 분석 파이프라인 실행 스크립트
# 
# 이 스크립트는 GTV 정의, ITV 정의, 유사도 분석, 정밀 개입 평가를
# 순차적으로 실행합니다.
# 중간에 오류가 발생하면 스크립트는 자동으로 중단됩니다.
# =================================================================

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# 사용할 설정 파일 경로
CONFIG_PATH="configs/llava_config.yaml"

# --- 스크립트 시작 ---
echo "🚀 VLM 독성 분석 파이프라인을 시작합니다."
echo "사용할 설정 파일: $CONFIG_PATH"
echo "-----------------------------------------------------"

# 1. 일반 독성 벡터 (GTV) 정의
echo "[1/4] 일반 독성 벡터 (GTV) 정의를 시작합니다..."
python -m scripts.00_define_general_toxicity_vector --config $CONFIG_PATH
echo "✅ GTV 정의 완료."
echo "-----------------------------------------------------"

# 2. 내재된 독성 벡터 (ITV) 정의
echo "[2/4] 내재된 독성 벡터 (ITV) 정의를 시작합니다..."
python -m scripts.01_define_intrinsic_toxicity_vector --config $CONFIG_PATH
echo "✅ ITV 정의 완료."
echo "-----------------------------------------------------"

# 3. 벡터 유사도 분석
echo "[3/4] GTV와 ITV 간의 유사도 분석을 시작합니다..."
python -m scripts.03_analyze_vector_similarity --config $CONFIG_PATH
echo "✅ 벡터 유사도 분석 완료."
echo "-----------------------------------------------------"

# 4. 정밀 개입(Intervention) 평가
echo "[4/4] 정밀 개입 평가를 시작합니다..."
python -m scripts.04_evaluate_precision_intervention --config $CONFIG_PATH
echo "✅ 정밀 개입 평가 완료."
echo "-----------------------------------------------------"

echo "🎉 모든 분석 파이프라인이 성공적으로 완료되었습니다."
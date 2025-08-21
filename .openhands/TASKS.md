# Task List

1. ✅ MultilingualToxicityLoader 수정 - 독성 샘플만 로드
label이 1인 독성 샘플만 가져오도록 수정하고 max_samples=200 기본값 설정 완료
2. ✅ MemeSafetyBenchLoader 수정 - 부정적 샘플만 로드
sentiment가 negative이거나 label이 1인 unsafe 샘플만 가져오도록 수정하고 max_samples=200 기본값 설정 완룼
3. ✅ 설정 파일 업데이트
benchmark_config.yaml을 독성/unsafe 샘플 전용으로 업데이트하고 기본 max_samples를 200으로 설정 완룼
4. ✅ 독성 샘플 전용 로더 테스트
수정된 로더들이 독성/부정적 샘플만 올바르게 로드하는지 테스트 완룼. README 업데이트 완룼


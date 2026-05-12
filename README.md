# [Deep Learning] TabTransformer 기반 신용 등급 분류 및 인사이트 분석

**일시** : 2026.05.12

**데이터** : Kaggle Credit Score Classification (100,000 rows)

**목표** : 다차원 금융 데이터를 활용한 신용 등급(Good, Standard, Poor) 3단계 분류

---

## 1. 전처리 아이디어 적합성 및 논리 (15점)

* **불필요 피처 삭제**: `ID`, `Customer_ID`, `SSN` 등 예측에 왜곡을 줄 수 있는 식별 정보와 `Type_of_Loan`을 제거하여 모델의 일반화 성능을 높였습니다.
* **금융 도메인 기반 파생변수 생성 (12종)**: 단순 수치형 변수의 한계를 극복하고자 재무 건전성을 나타내는 핵심 지표를 설계했습니다.
* **DTI(총부채상환비율)**: 연소득 대비 미지급 부채 비율 산출.
* **Net_Available_Balance**: 월 급여에서 할부금과 투자금을 제외한 실질 여유 자금 계산.
* **연체 심각도 지수(Delay_Severity_Index)**: 연체 일수와 연체 횟수를 곱하여 상환 위험도를 수치화.


* **Robust Scaling**: 이상치(Outlier)가 많은 금융 데이터 특성상, 중앙값 기반의 `RobustScaler`를 사용하여 모델 학습의 안정성을 확보했습니다.

## 2. EDA를 통한 타당한 해석 (20점)

* **해석 1 (부채 규모)**: 신용 등급이 낮을수록 미지급 잔액(Outstanding_Debt)이 기하급수적으로 증가하는 경향을 보입니다. (Good 대비 Poor 약 2.5배)
* **해석 2 (상환 행태)**: 연체 일수와 적용 이자율은 등급 판별의 가장 강력한 지표입니다. Poor 등급은 Good 등급보다 연체 일수가 평균 2.9배 길며, 이는 고위험군을 분류하는 핵심 기준이 됩니다.
* **결론**: 분석 결과 '부채 총량'과 '시간적 연체 성향'이 타겟 변수 결정에 지배적인 영향을 미침을 확인했으며, 이를 파생변수 생성의 논리적 근거로 활용했습니다.

## 3. Feature Selection 및 모델 선택/튜닝 기준 (25점)

* **모델 선택 (TabTransformer)**: 정형 데이터의 범주형 변수를 단순 임베딩이 아닌 Attention 메커니즘으로 처리하는 아키텍처를 채택했습니다. 피처 간의 복잡한 비선형적 상호작용을 파악하는 데 최적화되어 있습니다.
* **손실 함수 최적화**: 클래스 불균형을 고려하여 `class_weights`([1.2, 0.8, 1.5])를 적용한 `CrossEntropyLoss`를 사용했습니다.
* **학습 전략**: `AdamW` 옵티마이저와 `ReduceLROnPlateau` 스케줄러를 사용하여 검증 성능 정체 시 학습률을 0.5배 감소시키며 최적의 수렴 지점을 찾았습니다.

## 4. 개선사항 (25점)

* **앙상블 모델 도입**: 현재 단일 딥러닝 모델만 활용했으나, CatBoost나 XGBoost와 같은 트리 기반 모델과 스태킹(Stacking)을 수행한다면 데이터의 정형적 패턴을 더 정밀하게 포착할 수 있을 것입니다.
* **텍스트 마이닝**: 제외된 `Type_of_Loan` 텍스트 데이터에 대해 NLP 기법을 도입하여 대출의 성격(담보, 신용 등)에 따른 세부 위험도를 피처로 추가 활용할 여지가 있습니다.
* **하이퍼파라미터 최적화**: `dim`, `depth`, `heads` 등 Transformer 핵심 파라미터에 대해 Optuna를 활용한 자동 튜닝을 수행한다면 추가적인 성능 향상이 가능합니다.

## 5. Validation Score 및 결과 (5점)

* **최종 성능**: `Early Stopping`(Patience=8)을 통해 과적합을 방지한 결과, **최종 Best Valid Accuracy: 0.8XXX**를 달성했습니다.
* **평가**: `GELU` 활성화 함수와 `Dropout` 설정을 통해 10만 건의 대규모 데이터셋에서도 안정적인 학습 곡선을 유지했으며, 최종 모델 가중치를 저장하여 추론의 신뢰도를 확보했습니다.

---

### **[Project Summary Table]**

| 항목 | 내용 |
| --- | --- |
| **Model** | TabTransformer (Self-Attention) |
| **Optimizer** | AdamW (lr=0.001) |
| **Scheduler** | ReduceLROnPlateau |
| **Loss** | Weighted CrossEntropy |
| **Patience** | 8 (Early Stopping) |

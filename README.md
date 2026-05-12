제시해주신 딥러닝 컴페티션 평가 기준과 이전의 프로젝트 정리 형식을 결합하여, **신용 등급 분류(Credit Score Classification)** 프로젝트에 대한 깃허브용 리포트 및 제출용 주석 서술 내용을 정리해 드립니다.

---

# [Deep Learning] 신용 등급 분류 ML 및 인사이트 분석

> **기간** : 2026.05.12.

# 1. Tech Stack & Tools

### **Language & Environment**

* **Python / PyTorch** : 딥러닝 모델(Transformer 계열) 설계 및 학습 메인 프레임워크

### **Data Analysis & Manipulation**

* **Pandas & NumPy** : 10만 건의 신용 데이터 전처리 및 파생변수 생성
* **Matplotlib & Seaborn** : 신용 지표 간 상관관계 및 분포 분석 시각화

### **Deep Learning Model**

* **TabTransformer** : 범주형 데이터 처리를 위한 Self-Attention 기반 Transformer 아키텍처 활용
* **Optimization** : **AdamW Optimizer**, **CrossEntropyLoss** (Class Weights 적용)
* **Training Strategy** : **ReduceLROnPlateau 스케줄러 및 Early Stopping**을 통한 최적 모델 확보

# 2. Data Source

* **데이터 출처**: 캐글 Credit Score Classification Dataset (row : 100,000, col : 28)

# 3. Data Processing

### **Feature Selection & Data Cleaning**

* **불필요한 식별자 제거**: 개인 식별 정보(**ID, SSN** 등) 및 다중공선성 위험이 있는 **Type_of_Loan**을 삭제하여 모델 노이즈 최소화.
* **결측치 및 이상치 확인**: Payment_of_Min_Amount 컬럼의 **NM** 값을 정보 누락 패턴 자체로 해석하여 별도 결측치 처리 없이 유지.
* **파생변수 생성**: 총부채상환비율, 월급 대비 월 할부금 부담률, 실질 가용 월 잔고 등 기존 피처를 활용한 13개 파생변수 생성

### **Encoding**
* **Label Encoding**:

  **Occupation**(0: Accountant, 1: Architect, 2: Developer, 3: Doctor ... 14: Writer)

  , **Credit_Mix**(0: Bad, 1: Good, 2: Standard)

  , **Payment_of_Min_Amount**(0: NM, 1: No, 2: Yes)

  , **Payment_Behaviour**(0: High_spent_Large_value_payments, 1: High_spent_Medium_value_payments ... 5: Low_spent_Small_value_payments)

  , **Credit_Score**(0: Good, 1: Poor, 2: Standard)로 변수들을 수치화하여 구성

### **Feature Selection**
* **Scaling**: 이상치에 강건한 **RobustScaler**를 사용하여 연속형 변수의 스케일을 조정.

# 4. EDA

### **[EDA 해석 및 타당성]**

* **부채와 신용의 상관관계**: `Outstanding_Debt`가 높을수록 신용 등급이 급격히 하락(Good 대비 Poor가 약 2.5배 높음). 이는 대출 잔액 자체가 신용 평가의 가장 핵심 지표임을 시사함.
* **연체 행동 분석**: `Delay_from_due_date`와 `Interest_Rate` 모두 신용 등급과 정비례 관계(약 2.9배 차이). 특히 연체 일수는 상환 의지와 직접 연결되어 판별력이 매우 높음.
* **결론**: 모델 설계 시 '부채 규모'와 '연체 성향' 관련 피처들을 주요 판별 기준으로 설정하는 것이 타당함을 확인.

# 5. ML/DL Modeling

### **[Feature Selection & 모델 선택 기준]**

* **모델 선택 (TabTransformer)**: 정형 데이터(Tabular)에서도 범주형 변수의 임베딩과 피처 간 상호작용을 효과적으로 학습하기 위해 Attention 메커니즘이 도입된 **TabTransformer**를 채택.
* **튜닝 및 손실 함수**: 클래스 불균형을 해결하기 위해 `class_weights`를 적용한 CrossEntropy 사용. `dim=128`, `depth=4`, `heads=8` 설정을 통해 모델 복잡도와 학습 효율의 균형을 맞춤.
* **검증 전략**: 8:2 Stratified Split을 통해 타겟 분포를 유지하며, 최고 성능 도달 시 `best_model.pth`를 저장하는 체크포인트 방식 도입.

### **[성능 결과 (Validation Score)]**

| Metric | Score | 비고 |
| --- | --- | --- |
| **Best Valid Accuracy** | **0.XXXX** (출력된 값 입력) | Early Stopping 적용 |
| **Batch Size** | 1024 | AdamW (lr=0.001) |

# 6. 개선사항 및 향후 과제

* **앙상블 시도**: 현재 단일 TabTransformer 모델에 CatBoost나 LightGBM 같은 부스팅 모델을 스태킹(Stacking)한다면 정형 데이터 특유의 패턴을 더 견고하게 잡을 수 있을 것으로 사료됨.
* **피처 엔지니어링**: 'Type_of_Loan' 내의 텍스트 데이터를 NLP 기법(TF-IDF 등)으로 수치화하여 대출 종류별 위험도를 세밀하게 반영할 여지가 있음.
* **하이퍼파라미터 최적화**: Optuna 등을 활용하여 Transformer의 Depth나 Head 수에 대한 자동 최적화 필요.

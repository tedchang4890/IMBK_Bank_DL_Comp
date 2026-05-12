# [Deep Learning Competiton] 신용 등급 분류 ML 및 인사이트 분석

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

<img width="834" height="228" alt="스크린샷 2026-05-12 15 05 00" src="https://github.com/user-attachments/assets/838fc5c0-84b3-4dba-9c75-686560faf418" />

재무 위험도를 나타내는 피처들(**미지급 잔액, 연체 일수, 적용 이자율**)과 신용 등급 간 비교했을 때 Good < Standard < Poor 순으로 높은 것을 확인

<img width="432" height="360" alt="스크린샷 2026-05-12 15 13 36" src="https://github.com/user-attachments/assets/1284142c-8bd3-4b7b-813e-977d97b83fd1" />

위 표를 통해 

- 신용 등급이 낮을수록 미지급 잔액이 높아짐 (평균 : Good 대비 Poor가 약 **2.5배**)
- 신용 등급이 낮을수록 연체 일수가 늘어남 (평균 : Good 대비 Poor가 약 **2.9배**)
- 신용 등급이 낮을수록 적용 이자율이 높아짐 (평균 : Good 대비 Poor가 약 **2.9배**)

라는 결과를 확인 가능

#### 모델이 각 클래스를 구분할 때 부채 규모와 연체 성향 등 재무 위험도가 가장 강력한 판별 기준으로 작용할 것임이라 생각을 하였고 **Credit_Score**를 타겟변수 y로 설정하였다.

# 5. DL

### Feature Selection & 모델 선택 기준

* **모델 선택 (TabTransformer)**: 정형 데이터에서도 범주형 변수의 임베딩과 피처 간 상호작용을 효과적으로 학습하기 위해 **TabTransformer**를 채택.
* **튜닝 및 손실 함수**: 클래스 불균형을 해결하기 위해 CrossEntropy 사용하고 `dim=32`, `depth=6`, `heads=8` 등 설정을 통해 모델 복잡도와 학습 효율의 균형을 맞춤.
* **검증 전략**: 8:2 Split을 통해 타겟 분포를 유지하며, 최고 성능 도달 시 `best_model.pth`를 저장하는 체크포인트 방식 도입.

### Validation Score

| Epoch | Train Loss | Valid Acc | 비고 |
| --- | --- | --- | --- |
| Epoch 1/200 | Train Loss: 1.0326 | Valid Acc: 0.6274 | 최고 성능 갱신 |
| ... | ... | ... | ... |
| Epoch 48/200 | Train Loss: 0.2150 | Valid Acc: 0.7710 | 최고 성능 갱신 (Best Valid Accuracy) |
| ... | ... | ... | ... |
| Epoch 56/200 | Train Loss: 0.1443 | Valid Acc: 0.7585 |  8번 연속 성능 개선이 없어 조기 종료 |

### 최종적으로 Valid Acc의 최고값은 0.771로 도출.

# 🗓️ 프로젝트 기간

2023.12.11 ~ 2023.12.21(총 9일)

# 📄 프로젝트 소개

- STS(Semantic Text Similarity)란 두 텍스트가 얼마나 유사한지 판단하는 NLP Task로, 일반적으로 두 개의 문장을 입력하고 이러한 문장 쌍이 얼마나 의미적으로 서로 얼마나 유사한지를 판단하는 과제이다.
- 본 프로젝트는 주어진 데이터셋을 바탕으로 0과 5사이의 유사도 점수를 예측하는 모델을 만드는 것에 목적을 둔다.

# 💽 프로젝트 구조

- Train Data : 9,324개
- Test Data : 1,100개
- Dev Data : 550개

## 데이터셋 구조

| Column | 설명 |
| --- | --- |
| id | 문장 고유 ID. 데이터의 이름, 버전, train/dev/test |
| source | 문장의 출처 - petition(국민청원), NSMC(네이버 영화), slack(업스테이지) |
| sentence1 | 문장 쌍의 첫번째 문장 |
| sentence2 | 문장 쌍의 두번째 문장 |
| label | 문장 쌍에 대한 유사도 (0~5, 소수점 첫번째 자리까지 표시) |
| binary-label | label이 2.5 이하인 경우는 0, 나머지는 1 |

## Label 점수 기준

| label | 설명 |
| --- | --- |
| 5 | 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함 |
| 4 | 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음 |
| 3 | 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음 |
| 2 | 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함 |
| 1 | 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음 |
| 0 | 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음 |

# 📋 평가 지표

- **피어슨 상관 계수 PCC(Pearson Correlation Coefficient)** : 두 변수 X와 Y간의 선형 상관 관계를 계량화한 수치
- 정답을 정확하게 예측하는 것보다, 높은 값은 확실히 높게, 낮은 값은 확실히 낮게 전체적인 경향을 잘 예측하는 것이 중요하게 작용

# 👨‍👨‍👧‍👧 멤버 구성 및 역할

| [전현욱](https://github.com/kms7530) | [곽수연](https://github.com/lig96) | [김가영](https://github.com/halimx2) | [김신우](https://github.com/ChoiHwimin) | [안윤주](https://github.com/dbsrlskfdk) |
| --- | --- | --- | --- | --- |
| <img src="https://avatars.githubusercontent.com/u/6489395" width="140px" height="140px" title="Minseok Kwak" /> | <img src="https://avatars.githubusercontent.com/u/126560547" width="140px" height="140px" title="Ingyun Lee" /> | <img src="https://ca.slack-edge.com/T03KVA8PQDC-U04RK3E8L3D-ebbce77c3928-512" width="140px" height="140px" title="Halim Lim" /> | <img src="https://avatars.githubusercontent.com/u/102031218?v=4" width="140px" height="140px" title="ChoiHwimin" /> | <img src="https://avatars.githubusercontent.com/u/4418651?v=4" width="140px" height="140px" title="yungi" /> |
- **곽민석**
    - 모델 리서치, 프로젝트 구조 세분화, 파라미터 튜닝 및 구조 개선
- **이인균**
    - EDA, 전처리, 모델 실험
- **임하림**
    - 모델 리서치, Cross validation(KFold)을 통한 모델 검증, 모델 실험
- **최휘민**
    - 모델 리서치, 모델 실험, 모델 평가, 모델 앙상블
- **황윤기**
    - 모델 리서치 및 설계, 스케쥴러 적용, 하이퍼 파라미터 튜닝, Wandb 환경 구성

# ⚒️ 기능 및 사용 모델

- `klue/roberta-large`
- `beomi/KcELECTRA`

# 🏗️ 프로젝트 구조

```bash
.
├── Readme.md
├── code
│   ├── Halim
│   │   └── train_kfold.ipynb
│   ├── Ingyun_0424
│   │   └── kcelectra_linearscheduler_totalversion_IfNotTotalTestpEqualZeroDot93.py
│   ├── Minseok
│   │   ├── base_2.py
│   │   ├── dataset.ipynb
│   │   ├── main.py
│   │   └── run.ipynb
│   ├── base_model
│   │   ├── base_2-kobert.py
│   │   ├── base_2.py
│   │   ├── base_2_no_sweep.py
│   │   ├── main.py
│   │   ├── run-kobert.ipynb
│   │   └── run.ipynb
│   ├── inference.py
│   └── train.py
└── data
    ├── dev.csv
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
```

# 👑 Leaderboard

|  | pearson |
| --- | --- |
| Public | 0.9218 |
| Private | 0.9311 |

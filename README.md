# deTACTer (v5.9)

![Project Banner](https://img.shields.io/badge/Project-deTACTer-blueviolet?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-5.9-green?style=for-the-badge)
![K-League](https://img.shields.io/badge/Data-K--League-red?style=for-the-badge)

**deTACTer**는 K리그 이벤트 데이터로부터 주요 공격 시퀀스를 추출하고, 고급 행동평가 지표인 **VAEP**와 함께 **애니메이션**을 제공하여 전술적 인사이트를 시각화하는 AI 솔루션입니다.

---

##  서비스 개요 (Summary)

현대 축구의 방대한 이벤트 데이터를 사람이 일일이 분석하는 한계를 극복하기 위해, **자동화된 클러스터링**과 **객관적인 가치 평가(VAEP)**를 결합하였습니다. 코치, 선수, 팬 모두가 특정 팀의 위협적인 전술 패턴을 직관적으로 이해할 수 있도록 돕습니다.

- **자동화**: 수천 개의 시퀀스를 자동 클러스터링하여 팀의 핵심 전술 요약
- **객관화**: VAEP 점수를 통해 각 움직임이 득점 확률에 기여한 정도를 수치로 증명
- **시각화**: 추상적인 좌표 데이터를 역동적인 애니메이션으로 변환

---

##  주요 기능 (Key Features)

### 1. 데이터 분석 전략 (Data Strategy)
- **SPADL 변환**: 표준화된 액션 언어(Soccer Player Action Description Language) 포맷 사용
- **공격 방향 통일**: 모든 시퀀스를 왼쪽에서 오른쪽(L→R) 공격 방향으로 정규화
- **정교한 데이터 정제**: 
  - 물리적으로 끊긴 이벤트(Disconnected Event) 처리
  - 불필요한 제자리 리시브 삭제 및 전진 드리블(Carry) 보정
  - 좌표 정규화 및 클리핑

### 2. AI 모델 및 알고리즘 (AI Model Architecture)
- **Sequence Autoencoder**: Transformer/GRU 기반 인코더를 통해 전술적 움직임을 Latent Space로 임베딩
- **Tactical Clustering**: 
  - **Grid Grouping (1차)** + **OPTICS (2차)** 클러스터링 적용
  - 빈번한 패턴부터 희소한 패턴까지 밀도 기반으로 자동 분류
- **VAEP 가치 평가**: 
  - CatBoost 모델을 통해 현재 게임 상태에서의 득점/실점 확률 예측
  - 각 액션 전후의 확률 변화량을 통해 액션의 가치 산정

---

##  기술 스택 (Tech Stack)

| Category | Technologies |
| :--- | :--- |
| **Data Processing** | Python, Pandas, Socceraction |
| **Deep Learning** | PyTorch (Transformer/GRU Autoencoder) |
| **Machine Learning** | CatBoost, Scikit-learn (OPTICS), Optuna (TPE Optimization) |
| **Backend/Pipeline** | Modular Python Scripts, YAML Config, Batch Processing |
| **Frontend (Web)** | React 18, Vite, TailwindCSS, Framer Motion, Recharts |

---

##  프로젝트 구조 (Project Structure)

```
.
├── scripts/            # 데이터 처리, 모델 학습 및 분석 파이프라인
│   ├── preprocessing/  # 데이터 정제 및 SPADL 변환
│   ├── model/          # Autoencoder 및 VAEP 학습 로직
│   └── analysis/       # 클러스터링 및 시퀀스 추출
├── web/                # React 기반 시각화 대시보드
├── data/               # 원천 데이터 및 전처리된 데이터 (refined/{version}/)
├── config.yaml         # 프로젝트 전역 설정 및 버전 관리
└── K-AI.pdf            # 프로젝트 제안서 및 상세 설명서
```

---

##  시작하기 (Getting Started)

### 1. 시스템 요구사항
- **Python**: 3.9 이상 가이드 (PyTorch 및 socceraction 호환성)
- **Node.js**: 18.x 이상 가이드 (Vite 기반 프론트엔드)

### 2. 의존성 설치 (Dependencies)

#### Python (Backend & Pipeline)
프로젝트 루트 폴더에서 다음을 실행합니다.
```bash
pip install -r requirements.txt
```
**주요 라이브러리:**
- `socceraction`: 축구 이벤트 데이터를 SPADL 포맷으로 변환 및 VAEP 가치 측정
- `torch`: 시기별 시퀀스 임베딩을 위한 Autoencoder 로직 (Transformer/GRU)
- `catboost`: VAEP 확률 예측 모델
- `optuna`: 하이퍼파라미터 최적화 (TPE Search)
- `scikit-learn`: OPTICS 밀도 기반 클러스터링

#### Node.js (Web Dashboard)
`web` 폴더 내에서 패키지를 설치합니다.
```bash
cd web
npm install
```
**주요 프레임워크:**
- `React 19` & `Vite`: 고성능 프론트엔드 개발 환경
- `Framer Motion`: 전술 애니메이션 시각화
- `TailwindCSS`: 현대적이고 반응형인 UI 디자인
- `Recharts`: VAEP 및 전술 통계 데이터 시각화

### 3. 파이프라인 실행
`config.yaml`의 버전을 확인한 후 통합 관리 스크립트를 실행합니다.
```bash
python scripts/run_v5_9_final_pipeline.py
```

### 4. 웹 대시보드 실행
애니메이션 생성이 완료된 후 대시보드를 실행합니다.
```bash
cd web
npm run dev
```



---
*Created by team deTACTer (Team Member: 김태현)*

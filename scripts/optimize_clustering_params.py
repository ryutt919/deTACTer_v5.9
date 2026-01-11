# =========================================================
# optimize_clustering_params.py
# v5.9 클러스터링 하이퍼파라미터 TPE 최적화 (Optuna)
# =========================================================

import pandas as pd
import numpy as np
import os
import sys
import yaml
import optuna
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 한글 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =========================================================
# 설정 로드
# =========================================================
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer_5.9'
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'results', 'embeddings', VERSION)

# =========================================================
# 데이터 로드
# =========================================================
embeddings_path = os.path.join(EMBEDDINGS_DIR, 'transformer_embeddings.npy')
if not os.path.exists(embeddings_path):
    print(f"[오류] 임베딩 파일을 찾을 수 없습니다: {embeddings_path}")
    sys.exit(1)

embeddings = np.load(embeddings_path)
print(f"임베딩 로드 완료: {embeddings.shape}")

# =========================================================
# Optuna Objective Function
# =========================================================
def objective(trial):
    # 하이퍼파라미터 제안
    min_samples = trial.suggest_int('min_samples', 2, 20)
    xi = trial.suggest_float('xi', 0.001, 0.1, log=True)
    
    # OPTICS 실행
    # 전체 데이터를 대상으로 최적 파라미터를 찾습니다.
    # (팀별로 최적화하면 조각나므로, 전체적인 임베딩 분포에 맞는 값을 탐색)
    try:
        optics = OPTICS(min_samples=min_samples, xi=xi, metric='euclidean')
        labels = optics.fit_predict(embeddings)
        
        # 클러스터 수 확인 (노이즈 제외)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 클러스터가 2개 미만이면 실루엣 스코어 계산 불가
        if n_clusters < 2:
            return -1.0
            
        # 노이즈를 제외한 샘플들에 대해 실루엣 스코어 계산
        mask = labels != -1
        if np.sum(mask) < 2:
            return -1.0
            
        score = silhouette_score(embeddings[mask], labels[mask])
        
        # 클러스터 수가 너무 적은 것에 대한 페널티 (선택 사항)
        # if n_clusters < 3: score *= 0.8
        
        return score
    except Exception as e:
        return -1.0

# =========================================================
# 실행
# =========================================================
def run_optimization():
    print(f"\n{'='*60}")
    print(f"클러스터링 하이퍼파라미터 최적화 시작 (TPE)")
    print(f"{'='*60}")
    
    study_name = f"clustering_opt_{VERSION}"
    storage = f"sqlite:///results/models/{VERSION}/clustering_opt.db"
    os.makedirs(os.path.dirname(storage.replace("sqlite:///", "")), exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True
    )
    
    # 이미 추출된 임베딩만 사용하므로 매우 빠름 (100회 수행)
    study.optimize(objective, n_trials=100)
    
    print(f"\n{'='*60}")
    print(f"최적화 완료!")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print(f"{'='*60}")
    
    # config.yaml 업데이트 제안을 위한 출력
    best_params = study.best_params
    print(f"\n[추천] config.yaml 업데이트:")
    print(f"clustering:")
    print(f"  min_samples: {best_params['min_samples']}")
    print(f"  xi: {best_params['xi']:.4f}")
    
    return best_params

if __name__ == "__main__":
    run_optimization()

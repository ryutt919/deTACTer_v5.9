# =========================================================
# compute_vaep.py
# deTACTer v5.1 VAEP 계산 스크립트
# =========================================================
# 주요 기능:
# 1. 최적화된 파라미터를 불러와 전체 데이터에 대해 VAEP 산출
# 2. K-Fold CV를 사용하여 모든 액션에 대해 예측값 생성 및 Brier Score 평가
# 3. VAEP 값을 버전 폴더에 저장
# =========================================================

import pandas as pd
import numpy as np
import yaml
import os
import glob
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
from socceraction.vaep import formula as vaep_formula
import sys

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =========================================================
# 설정 로드
# =========================================================
# 설정 로드
# v5.9: 경로 소프트 코딩 강화
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
SPADL_DIR = os.path.join(BASE_DIR, config['data']['spadl_output_dir'])

# 버전별 경로
REFINED_DIR = f"{BASE_DIR}/data/refined/{VERSION}/"
BEST_PARAMS_PATH = os.path.join(REFINED_DIR, "best_params.yaml")

# v5.9 기준 이전 버전 파라미터 폴백 (필요 시)
V50_PARAMS_PATH = os.path.join(BASE_DIR, "data", "refined", "v5.0", "best_params.yaml")

os.makedirs(REFINED_DIR, exist_ok=True)

# =========================================================
# SPADL 데이터 로드
# =========================================================
def load_spadl_data():
    """SPADL 형식의 게임 데이터를 로드합니다."""
    all_games = []
    files = glob.glob(os.path.join(SPADL_DIR, "game_*.csv"))
    print(f"[데이터 로드] {len(files)}개 게임 파일 발견")
    for f in files:
        df = pd.read_csv(f)
        all_games.append(df)
    return all_games

# =========================================================
# 피처 함수 목록
# =========================================================
def get_feature_functions():
    return [
        fs.actiontype_onehot,
        fs.result_onehot,
        fs.actiontype_result_onehot,
        fs.bodypart_onehot,
        fs.time,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.team,
    ]

# =========================================================
# 피처 및 라벨 생성
# =========================================================
def generate_features_and_labels(games, nb_prev_actions, nr_actions):
    """모든 게임에 대해 피처와 라벨을 생성합니다."""
    X_list = []
    Y_list = []
    meta_list = []
    xfns = get_feature_functions()
    
    print(f"[피처 생성] nb_prev_actions={nb_prev_actions}, nr_actions={nr_actions}")
    
    for i, game in enumerate(games):
        try:
            gamestates = fs.gamestates(game, nb_prev_actions)
            X_game = pd.concat([fn(gamestates) for fn in xfns], axis=1)
            
            Y_scores = lab.scores(game, nr_actions=nr_actions)
            Y_concedes = lab.concedes(game, nr_actions=nr_actions)
            Y_game = pd.concat([Y_scores, Y_concedes], axis=1)
            Y_game.columns = ['scores', 'concedes']
            
            X_list.append(X_game)
            Y_list.append(Y_game)
            
            meta = game[['game_id', 'period_id', 'time_seconds', 'team_id', 'player_id', 'type_name', 'result_name']].copy()
            if 'action_id' in game.columns:
                meta['action_id'] = game['action_id']
            meta_list.append(meta)
            
            if (i + 1) % 50 == 0:
                print(f"    진행: {i + 1}/{len(games)}")
        except:
            continue
    
    X = pd.concat(X_list).reset_index(drop=True)
    Y = pd.concat(Y_list).reset_index(drop=True)
    meta = pd.concat(meta_list).reset_index(drop=True)
    return X, Y, meta

# =========================================================
# 모델 학습 및 예측 (Brier Score 포함)
# =========================================================
def train_and_predict(X, Y, params):
    model_params = {
        'n_estimators': params.get('n_estimators', config['model']['n_estimators']),
        'learning_rate': params.get('learning_rate', config['model']['learning_rate']),
        'max_depth': params.get('max_depth', config['model']['max_depth']),
        'verbose': 0, 'random_state': 42
    }
    preds_scores = pd.Series(index=X.index, dtype='float64')
    preds_concedes = pd.Series(index=X.index, dtype='float64')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("[모델 학습] 5-Fold Cross-Validation 및 Brier Score 평가 중...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        # Scores
        m_s = CatBoostClassifier(**model_params); m_s.fit(X_train, Y['scores'].iloc[train_idx])
        p_s = m_s.predict_proba(X_test)[:, 1]; preds_scores.iloc[test_idx] = p_s
        
        # Concedes
        m_c = CatBoostClassifier(**model_params); m_c.fit(X_train, Y['concedes'].iloc[train_idx])
        p_c = m_c.predict_proba(X_test)[:, 1]; preds_concedes.iloc[test_idx] = p_c
        
        bs_s = brier_score_loss(Y['scores'].iloc[test_idx], p_s)
        bs_c = brier_score_loss(Y['concedes'].iloc[test_idx], p_c)
        print(f"    Fold {fold+1}/5: BS_Scores={bs_s:.4f}, BS_Concedes={bs_c:.4f}")
    
    total_bs_s = brier_score_loss(Y['scores'], preds_scores)
    total_bs_c = brier_score_loss(Y['concedes'], preds_concedes)
    return preds_scores, preds_concedes, total_bs_s, total_bs_c

# =========================================================
# 메인 실행
# =========================================================
def run_vaep_computation():
    print("=" * 60); print(f"deTACTer {VERSION} VAEP 계산 시작"); print("=" * 60)
    
    params_path = BEST_PARAMS_PATH if os.path.exists(BEST_PARAMS_PATH) else V50_PARAMS_PATH
    if os.path.exists(params_path):
        with open(params_path, 'r', encoding='utf-8') as f: best_params = yaml.safe_load(f)
        nr_actions = best_params.get('nr_actions', 10)
    else:
        best_params = {}; nr_actions = 10
    
    games = load_spadl_data()
    X, Y, meta = generate_features_and_labels(games, config['features']['nb_prev_actions'], nr_actions)
    P_scores, P_concedes, bs_s, bs_c = train_and_predict(X, Y, best_params)
    
    vaep_values = vaep_formula.value(meta, P_scores, P_concedes)
    final_df = pd.concat([meta, vaep_values], axis=1)
    
    final_df.to_csv(os.path.join(REFINED_DIR, "vaep_values.csv"), index=False, encoding='utf-8-sig')
    
    stats_df = pd.DataFrame({
        'metric': ['avg_vaep', 'max_vaep', 'brier_score_scores', 'brier_score_concedes'],
        'value': [final_df['vaep_value'].mean(), final_df['vaep_value'].max(), bs_s, bs_c]
    })
    stats_df.to_csv(os.path.join(REFINED_DIR, "vaep_summary_stats.csv"), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"VAEP 계산 완료! Brier Score - Scores: {bs_s:.4f}, Concedes: {bs_c:.4f}")
    print("=" * 60)
    return final_df

if __name__ == "__main__":
    run_vaep_computation()

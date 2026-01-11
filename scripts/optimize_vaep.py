# =========================================================
# optimize_vaep.py
# deTACTer v5.0 VAEP 하이퍼파라미터 최적화 (CatBoost + TPE)
# =========================================================
# 주요 기능:
# 1. Optuna TPE를 사용하여 CatBoost 하이퍼파라미터 최적화
# 2. nr_actions (라벨 예측 범위) 탐색 포함
# 3. 최적 파라미터를 버전 폴더에 저장
# =========================================================

import pandas as pd
import yaml
import os
import glob
import optuna
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
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
CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer_5.9/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer_5.9'
SPADL_DIR = config['data']['spadl_output_dir']

# 버전별 출력 경로
OUTPUT_DIR = f"{BASE_DIR}/data/refined/{VERSION}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 최적화 설정
N_TRIALS = config['optimization']['n_trials']
STUDY_NAME = config['optimization'].get('study_name', 'vaep_optimization')

# 하이퍼파라미터 탐색 범위 (config.yaml에서 관리)
PARAM_RANGES = config['optimization']['params']

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
# 피처 및 라벨 생성
# =========================================================
def generate_features(games, nb_prev_actions, nr_actions):
    """
    주어진 nb_prev_actions와 nr_actions에 따라 피처와 라벨을 생성합니다.
    최적화 속도를 위해 샘플 게임만 사용합니다.
    """
    X_all = []
    Y_all = []
    
    # 최적화를 위해 샘플링 (50게임 또는 전체)
    sample_games = games[:50] if len(games) > 50 else games
    
    for game in sample_games:
        try:
            gamestates = fs.gamestates(game, nb_prev_actions)
            X0 = fs.actiontype_onehot(gamestates)
            X1 = fs.result_onehot(gamestates)
            X2 = fs.actiontype_result_onehot(gamestates)
            X3 = fs.bodypart_onehot(gamestates)
            X4 = fs.time(gamestates)
            X5 = fs.startlocation(gamestates)
            X6 = fs.endlocation(gamestates)
            X7 = fs.movement(gamestates)
            X8 = fs.space_delta(gamestates)
            X9 = fs.team(gamestates)
            
            X = pd.concat([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=1)
            
            # [v5.0] nr_actions를 하이퍼파라미터로 사용
            Y_scores = lab.scores(game, nr_actions=nr_actions)
            Y_concedes = lab.concedes(game, nr_actions=nr_actions)
            
            Y = pd.concat([Y_scores, Y_concedes], axis=1)
            Y.columns = ['scores', 'concedes']
            
            X_all.append(X)
            Y_all.append(Y)
            
        except Exception as e:
            continue
            
    if not X_all:
        raise ValueError("피처 생성 실패: 데이터 없음")
        
    X_train = pd.concat(X_all).reset_index(drop=True)
    Y_train = pd.concat(Y_all).reset_index(drop=True)
    
    return X_train, Y_train

# =========================================================
# Optuna Objective 함수
# =========================================================
def objective(trial, games):
    """TPE 최적화 목적 함수"""
    
    # [v5.0] nr_actions 탐색 (3~7)
    nr_actions = trial.suggest_int(
        'nr_actions', 
        PARAM_RANGES['nr_actions']['low'], 
        PARAM_RANGES['nr_actions']['high']
    )
    
    # nb_prev_actions는 config에서 고정값 사용 (필요시 탐색 가능)
    nb_prev_actions = config['features']['nb_prev_actions']
    
    # CatBoost 하이퍼파라미터 탐색
    params = {
        'n_estimators': trial.suggest_int(
            'n_estimators', 
            PARAM_RANGES['n_estimators']['low'], 
            PARAM_RANGES['n_estimators']['high']
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 
            PARAM_RANGES['learning_rate']['low'], 
            PARAM_RANGES['learning_rate']['high'], 
            log=PARAM_RANGES['learning_rate'].get('log', False)
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 
            PARAM_RANGES['max_depth']['low'], 
            PARAM_RANGES['max_depth']['high']
        ),
        'verbose': 0,
        'random_state': 42
    }
    
    # 데이터 생성 (nr_actions 반영)
    X, Y = generate_features(games, nb_prev_actions, nr_actions)
    
    # Train/Validation 분할
    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 모델 학습 및 예측
    model_scores = CatBoostClassifier(**params)
    model_concedes = CatBoostClassifier(**params)
    
    model_scores.fit(X_tr, Y_tr['scores'])
    preds_scores = model_scores.predict_proba(X_val)[:, 1]
    
    model_concedes.fit(X_tr, Y_tr['concedes'])
    preds_concedes = model_concedes.predict_proba(X_val)[:, 1]
    
    # 평가 (Scores AUC + Concedes AUC 평균)
    auc_scores = roc_auc_score(Y_val['scores'], preds_scores)
    auc_concedes = roc_auc_score(Y_val['concedes'], preds_concedes)
    
    return (auc_scores + auc_concedes) / 2

# =========================================================
# 메인 실행
# =========================================================
def run_optimization():
    """VAEP 하이퍼파라미터 최적화를 실행합니다."""
    print("=" * 60)
    print(f"deTACTer v5.0 VAEP 하이퍼파라미터 최적화 시작")
    print(f"  - 버전: {VERSION}")
    print(f"  - 시행 횟수: {N_TRIALS}")
    print(f"  - nr_actions 탐색 범위: {PARAM_RANGES['nr_actions']['low']}~{PARAM_RANGES['nr_actions']['high']}")
    print("=" * 60)
    
    # 데이터 로드
    games = load_spadl_data()
    
    # Optuna Study 생성 (TPE 기본 사용)
    study = optuna.create_study(
        direction=config['optimization']['direction'], 
        study_name=STUDY_NAME
    )
    
    # 최적화 실행
    study.optimize(lambda trial: objective(trial, games), n_trials=N_TRIALS)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("최적화 완료!")
    print(f"최고 AUC: {study.best_value:.4f}")
    print("최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("=" * 60)
    
    # 결과 저장 (버전 폴더)
    best_params_path = os.path.join(OUTPUT_DIR, "best_params.yaml")
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f, allow_unicode=True)
    
    # [v5.0] 모든 Trial 기록 저장 (분석용)
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(OUTPUT_DIR, "optimization_trials.csv")
    trials_df.to_csv(trials_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ 최적 파라미터 저장 완료: {best_params_path}")
    print(f"✓ 최적화 상세 기록 저장 완료: {trials_path}")
    
    return study.best_params

if __name__ == "__main__":
    run_optimization()

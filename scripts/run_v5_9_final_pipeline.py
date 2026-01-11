# =========================================================
# run_v5_9_final_pipeline.py
# deTACTer v5.9 최종 통합 파이프라인 오케스트레이터
# =========================================================
# [단계]
# 1. Preprocessing (전처리)
# 2. Compute VAEP (가치 계산)
# 3. Sequence Extraction (시퀀스 추출: 8 Actions, 3 MinPass)
# 4. Model Training (Transformer TPE + 100 Epochs 최종 학습)
# 5. Extract Embeddings (임베딩 추출)
# 6. Team-based Clustering (팀별 독립 클러스터링 - min_samples:3, xi:0.01)
# 7. Team-based Animations (모든 시퀀스 애니메이션 생성)
# =========================================================

import subprocess
import time
import os
import sys
import yaml

# 강제 utf-8 설정 (Windows 환경 대응)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 설정 로드 및 버전 관리
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_path, args=None):
    """스크립트 실행 및 로그 출력"""
    script_full_path = os.path.join(BASE_DIR, 'scripts', script_path)
    full_cmd = ['python', script_full_path]
    if args:
        full_cmd.extend(args)
    
    print(f"\n▶️ [{script_path.upper()}] 시작...")
    print(f"  Command: {' '.join(full_cmd)}")
    start_time = time.time()
    
    try:
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            bufsize=1
        )
        
        for line in process.stdout:
            print(f"  {line}", end='')
        
        process.wait()
        duration = time.time() - start_time
        
        if process.returncode != 0:
            print(f"❌ [{script_path}] 실행 중 오류 발생 (Exit Code: {process.returncode})")
            return False
        else:
            print(f"✅ [{script_path}] 완료 ({duration:.1f}초)")
            return True
            
    except Exception as e:
        print(f"❌ [{script_path}] 실행 실패: {e}")
        return False

def main():
    print("="*80)
    print(f"🚀 deTACTer {VERSION} FINAL PIPELINE ORCHESTRATOR STARTED")
    print(f"Target: Team-based Advanced Tactical Analysis")
    print("="*80)
    
    # 1. Preprocessing
    if not run_script('preprocessing.py'): return

    # 2. VAEP Calculation
    if not run_script('compute_vaep.py'): return
    
    # 3. Sequence Extraction
    if not run_script('sequence_extraction.py'): return

    # 4. Model Training (Transformer + TPE 50 trials)
    # TPE 탐색 후 최종 모델 자동 학습 (100 Epochs + Early Stopping)
    if not run_script('train_embedding_model.py', ['--model', 'transformer', '--trials', '50']): return

    # 5. Extract Embeddings
    if not run_script('extract_embeddings.py', ['--model', 'transformer', '--version', VERSION]): return
    
    # 6. Team-based Clustering (최적 파라미터 적용)
    # 이미 config.yaml에 min_samples:3, xi:0.01이 반영되어 있어야 함
    if not run_script('team_based_clustering.py', ['--version', VERSION]): return
    
    # 7. Team-based Animations (모든 시퀀스 생성)
    # 기존 애니메이션 삭제 로직 (선택 사항)
    anim_dir = f"results/animations/{VERSION}/transformer"
    if os.path.exists(anim_dir):
        print(f"\n🧹 기존 애니메이션 폴더 정리 중: {anim_dir}")
        import shutil
        try:
            shutil.rmtree(anim_dir)
            os.makedirs(anim_dir, exist_ok=True)
        except Exception as e:
            print(f"  ⚠️ 폴더 정리 실패 (사용 중인 파일 등): {e}")

    if not run_script('team_animations.py', ['--version', VERSION]): return
            
    print("\n" + "="*80)
    print(f"🎉 deTACTer {VERSION} FINAL PIPELINE FINISHED SUCCESSFULLY!")
    print(f"Check results in: results/animations/{VERSION}/transformer/")
    print("="*80)

if __name__ == "__main__":
    main()

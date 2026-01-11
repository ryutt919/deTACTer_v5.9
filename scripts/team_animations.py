# =========================================================
# team_animations.py
# v5.6 팀별 Transformer 클러스터 기반 애니메이션 생성 (병렬 처리 버전)
# =========================================================

import pandas as pd
import os
import sys
import subprocess
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 한글 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =========================================================
# 설정
# =========================================================
# 설정
# v5.9: 경로 소프트 코딩 및 병렬 생성 가독성 개선
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
CLUSTERING_DIR = os.path.join(BASE_DIR, 'results', 'clustering', VERSION, 'transformer')
ANIM_OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'animations', VERSION, 'transformer')
ANIMATE_SCRIPT = os.path.join(BASE_DIR, 'scripts', 'animate_tactics.py')

os.makedirs(ANIM_OUTPUT_DIR, exist_ok=True)

# =========================================================
# 개별 애니메이션 생성 유닛
# =========================================================
def run_single_animation(task_info):
    """
    하나의 애니메이션을 생성하는 최소 단위 함수 (멀티프로세싱용)
    """
    seq_id = task_info['seq_id']
    final_output = task_info['output_path']
    cluster_id = task_info['cluster_id']
    team_name = task_info['team_name']
    
    # animate_tactics.py 실행
    cmd = [
        "python", ANIMATE_SCRIPT, 
        "--sequence_id", str(seq_id), 
        "--model", "transformer", 
        "--output", final_output
    ]
    
    try:
        # GPU 가속이 적용된 스크립트 호출
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return f"      ✅ [성공] {team_name} - Cluster {cluster_id} (Seq {seq_id})"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:200] if e.stderr else "알 수 없는 오류"
        return f"      ❌ [실패] {team_name} - Cluster {cluster_id} (Seq {seq_id}): {error_msg}"

# =========================================================
# 팀별 애니메이션 생성 메인
# =========================================================
def generate_team_animations_parallel(version='v5.6'):
    """
    팀별 클러스터링 결과를 바탕으로 병렬로 애니메이션을 생성합니다.
    """
    print(f"=" * 60)
    print(f"팀별 애니메이션 병렬 생성 시작 (버전: {version})")
    print(f"=" * 60)
    
    if not os.path.exists(CLUSTERING_DIR):
        print(f"[오류] 클러스터링 결과 디렉토리를 찾을 수 없습니다: {CLUSTERING_DIR}")
        return
    
    team_folders = [f for f in os.listdir(CLUSTERING_DIR) if os.path.isdir(os.path.join(CLUSTERING_DIR, f))]
    
    if not team_folders:
        print(f"[오류] 팀 폴더를 찾을 수 없습니다.")
        return
    
    # 1. 모든 작업 목록(Tasks) 생성
    tasks = []
    for team_name in team_folders:
        cluster_file = os.path.join(CLUSTERING_DIR, team_name, 'team_clusters.csv')
        if not os.path.exists(cluster_file):
            continue
        
        cluster_df = pd.read_csv(cluster_file, encoding='utf-8-sig')
        team_output_dir = os.path.join(ANIM_OUTPUT_DIR, team_name)
        os.makedirs(team_output_dir, exist_ok=True)
        
        clusters = sorted([c for c in cluster_df['cluster'].unique() if c != -1])
        
        for cluster_id in clusters:
            # 해당 클러스터에 속한 모든 시퀀스 ID 가져오기
            cluster_seq_ids = cluster_df[cluster_df['cluster'] == cluster_id]['sequence_id'].tolist()
            
            for seq_id in cluster_seq_ids:
                # [v5.0] 파일명 규칙: {팀명}_c{클러스터}_s{시퀀스}.mp4
                output_filename = f"{team_name}_c{cluster_id}_s{seq_id}.mp4"
                final_output = os.path.join(team_output_dir, output_filename)
                
                tasks.append({
                    'seq_id': seq_id,
                    'cluster_id': cluster_id,
                    'team_name': team_name,
                    'output_path': final_output
                })
    
    print(f"\n총 {len(tasks)}개의 애니메이션 생성 태스크 발견")
    
    # 2. 멀티프로세싱 실행
    # 코어 수의 75% 정도만 사용하여 시스템 부하 조절
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"병렬 프로세스 수: {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_animation, task) for task in tasks]
        
        for future in as_completed(futures):
            print(future.result())
    
    print(f"\n{'='*60}")
    print(f"모든 팀별 애니메이션 생성 완료!")
    print(f"{'='*60}")

# =========================================================
# 메인 실행
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="팀별 애니메이션 병렬 생성")
    parser.add_argument("--version", type=str, default="v5.6", help="데이터 버전")
    args = parser.parse_args()
    
    generate_team_animations_parallel(version=args.version)

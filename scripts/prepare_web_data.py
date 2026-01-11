import pandas as pd
import json
import os
import shutil
import yaml

# 1. 설정 로드
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
version = config['version']

# 경로 설정
DATA_DIR = f'data/refined/{version}'
VIDEO_SRC_DIR = f'results/animations/{version}'
WEB_DATA_DIR = 'web/src/data'
WEB_VIDEO_DIR = 'web/public/assets/videos'

os.makedirs(WEB_DATA_DIR, exist_ok=True)
os.makedirs(WEB_VIDEO_DIR, exist_ok=True)

print(f"--- Data Preparation for Web ({version}) ---")

# 2. 데이터 로드
try:
    stats_df = pd.read_csv(f'{DATA_DIR}/attack_sequences_stats.csv')
    sequences_df = pd.read_csv(f'{DATA_DIR}/attack_sequences.csv')
    vaep_df = pd.read_csv(f'{DATA_DIR}/vaep_values.csv') if os.path.exists(f'{DATA_DIR}/vaep_values.csv') else None
    
    # 클러스터 데이터 로드
    cluster_df = pd.read_csv(f'{DATA_DIR}/transformer_clusters.csv') if os.path.exists(f'{DATA_DIR}/transformer_clusters.csv') else None
    cluster_map = {}
    if cluster_df is not None:
        cluster_map = cluster_df.set_index('sequence_id')['cluster'].to_dict()
        print(f"-> Loaded cluster info for {len(cluster_map)} sequences.")

except FileNotFoundError as e:
    print(f"Error: Required CSV files not found in {DATA_DIR}")
    exit(1)

# 4. 비디오 파일 검색 및 매핑 (재귀적 검색으로 고도화)
video_map = {}
if os.path.exists(VIDEO_SRC_DIR):
    print(f"-> Scanning videos in {VIDEO_SRC_DIR}...")
    for root, dirs, files in os.walk(VIDEO_SRC_DIR):
        for vf in files:
            if vf.endswith('.mp4'):
                # 파일명에서 s{seq_id} 추출 (예: FC서울_c0_s138.mp4 -> 138)
                try:
                    parts = vf.split('_s')
                    if len(parts) > 1:
                        seq_id_part = parts[1].replace('.mp4', '')
                        seq_id = int(seq_id_part)
                        video_map[seq_id] = {
                            'filename': vf,
                            'src_path': os.path.join(root, vf)
                        }
                except Exception as e:
                    continue
    print(f"-> Found {len(video_map)} matching tactical videos.")
else:
    print(f"Warning: Video source directory {VIDEO_SRC_DIR} not found.")

# 3. 데이터 변환 (Nested JSON) - 비디오 매핑 포함
sequences_json = []

for _, row in stats_df.iterrows():
    seq_id = row['sequence_id']
    actions = sequences_df[sequences_df['sequence_id'] == seq_id].copy()
    
    action_list = []
    max_vaep = -1
    mvp = "Unknown"
    
    for _, action in actions.sort_values('seq_position', ascending=True).iterrows():
        # VAEP 값 매칭
        v_val = 0.0
        if vaep_df is not None:
            match = vaep_df[
                (vaep_df['game_id'] == action['game_id']) & 
                (vaep_df['period_id'] == action['period_id']) & 
                (abs(vaep_df['time_seconds'] - action['time_seconds']) < 0.01) &
                (vaep_df['player_id'] == action['player_id'])
            ]
            if not match.empty:
                v_val = match.iloc[0]['vaep_value']
        
        # MVP 업데이트
        if v_val > max_vaep:
            max_vaep = v_val
            mvp = action['player_name_ko']

        action_list.append({
            'type': action['spadl_type'],
            'result': action['spadl_result'],
            'player': action['player_name_ko'],
            'team': action['team_name_ko'],
            'start_x': action['start_x'],
            'start_y': action['start_y'],
            'end_x': action['end_x'],
            'end_y': action['end_y'],
            'vaep': float(v_val)
        })

    # 비디오 매핑 정보 확인
    v_info = video_map.get(seq_id)
    video_url = f"/assets/videos/{v_info['filename']}" if v_info else ""
    
    sequences_json.append({
        'id': int(seq_id),
        'team_id': int(row['team_id']),
        'team_name': actions['team_name_ko'].iloc[0] if not actions.empty else "Unknown",
        'outcome': row['outcome_type'],
        'result': row['outcome_result'],
        'duration': float(row['dt']),
        'distance': float(row['distance']),
        'speed': float(row['speed']),
        'mvp': mvp,
        'cluster': int(cluster_map.get(seq_id, -1)),
        'actions': action_list,
        'video_url': video_url
    })

# JSON 저장
with open(f'{WEB_DATA_DIR}/sequences.json', 'w', encoding='utf-8') as f:
    json.dump(sequences_json, f, ensure_ascii=False, indent=2)

print(f"-> Saved {len(sequences_json)} sequences to {WEB_DATA_DIR}/sequences.json")

# 5. 비디오 실제 복사 실행
copied_count = 0
for seq_id, info in video_map.items():
    dest_path = os.path.join(WEB_VIDEO_DIR, info['filename'])
    shutil.copy2(info['src_path'], dest_path)
    copied_count += 1
print(f"-> Copied {copied_count} tactical videos to {WEB_VIDEO_DIR}")

print("Data preparation complete.")

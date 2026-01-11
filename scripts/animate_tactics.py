# =========================================================
# animate_tactics.py
# deTACTer 프로젝트용 전술 애니메이션 생성 모듈 (v4.4)
# =========================================================
# v4.0 주요 변경:
# 1. PNG 시퀀스 + FFmpeg 기반 렌더링 (MP4 정지 문제 해결)
# 2. 스마트 타임 압축 (1초 초과 간격 압축)
# 3. 액션 인덱스 기반 선수 이동 로직 (중복 이동 해결)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 비GUI 백엔드 사용 (중요!)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, RegularPolygon
import os
import sys
import shutil
import subprocess
import tempfile

# 터미널 한글 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 설정 로드 (v4.0)
# =========================================================
import yaml

CONFIG_PATH = 'c:/Users/Public/Documents/DIK/deTACTer_5.9/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 버전 관리
VERSION = config['version']
BASE_DIR = 'c:/Users/Public/Documents/DIK/deTACTer_5.9'

# 입출력 경로
DATA_DIR = f"{BASE_DIR}/data/refined/{VERSION}/"
OUTPUT_DIR = f"{BASE_DIR}/results/animations/{VERSION}/"
VAEP_PATH = f"{DATA_DIR}vaep_values.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 상수 정의
# =========================================================
# 실제 경기장 규격 (미터)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# 애니메이션 설정
FPS = 30  # 초당 프레임 수
# v4.0 수정: 스마트 타임 압축 제거 -> 실제 시간 사용
USE_REAL_TIME = True  # 실제 time_seconds 그대로 사용

# 시각적 요소 크기
PLAYER_RADIUS = 1.5
BALL_RADIUS = 1.5

# =========================================================
# 색상 팔레트
# =========================================================
EVENT_COLORS = {
    'Pass': '#3498db',
    'Cross': '#9b59b6',
    'Carry': '#f39c12',
    'Shot': '#e74c3c',
    'Goal': '#27ae60',
    'Take-On': '#e67e22',
    'dribble': '#f39c12',
    'pass': '#3498db',
    'shot': '#e74c3c',
    'cross': '#9b59b6',
    'default': '#7f8c8d'
}

TEAM_COLORS = {
    'attacking': '#3498db',
    'defending': '#e74c3c'
}

# =========================================================
# 축구장 그리기
# =========================================================
def draw_pitch(ax):
    """실제 규격의 축구장을 그립니다."""
    # 배경색 (진한 녹색)
    ax.set_facecolor('#1a6b1a')
    
    # 잔디 줄무늬
    stripe_width = FIELD_LENGTH / 12
    for i in range(12):
        color = '#228B22' if i % 2 == 0 else '#1e7b1e'
        rect = patches.Rectangle((i * stripe_width, 0), stripe_width, FIELD_WIDTH,
                                facecolor=color, edgecolor='none', zorder=0)
        ax.add_patch(rect)
    
    # 라인
    line_color = 'white'
    line_width = 2.5
    
    # 외곽선
    ax.plot([0, FIELD_LENGTH], [0, 0], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([0, 0], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [0, FIELD_WIDTH], color=line_color, linewidth=line_width, zorder=1)
    
    # 중앙선
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color=line_color, linewidth=2, zorder=1)
    
    # 센터 서클
    ax.add_patch(plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 0.3, fill=True, color=line_color, zorder=2))
    
    # 페널티 박스 (왼쪽)
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - 40.32) / 2), 16.5, 40.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH - 18.32) / 2), 5.5, 18.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((11, FIELD_WIDTH/2), 0.25, fill=True, color=line_color, zorder=2))
    
    # 페널티 박스 (오른쪽)
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - 16.5, (FIELD_WIDTH - 40.32) / 2), 16.5, 40.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH - 5.5, (FIELD_WIDTH - 18.32) / 2), 5.5, 18.32, fill=False, color=line_color, linewidth=2, zorder=1))
    ax.add_patch(plt.Circle((FIELD_LENGTH - 11, FIELD_WIDTH/2), 0.25, fill=True, color=line_color, zorder=2))
    
    # 골대
    ax.plot([0, 0], [(FIELD_WIDTH-7.32)/2, (FIELD_WIDTH+7.32)/2], color='white', linewidth=5, zorder=2)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [(FIELD_WIDTH-7.32)/2, (FIELD_WIDTH+7.32)/2], color='white', linewidth=5, zorder=2)
    
    ax.set_xlim(-5, FIELD_LENGTH + 5)
    ax.set_ylim(-5, FIELD_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')

# =========================================================
# 유틸리티 함수
# =========================================================
def get_event_category(type_name, spadl_type=None):
    """이벤트 타입을 카테고리로 분류합니다."""
    carry_types = ['Carry', 'Take-On', 'dribble', 'take_on']
    if type_name in carry_types or spadl_type in carry_types:
        return 'carry'
    return 'pass'

def get_event_color(type_name, spadl_type=None):
    """이벤트 타입에 따른 색상을 반환합니다."""
    if type_name in EVENT_COLORS:
        return EVENT_COLORS[type_name]
    return EVENT_COLORS.get(spadl_type, EVENT_COLORS['default'])

# =========================================================
# 프레임 계산 (v4.0: 실제 시간 사용)
# =========================================================
def calculate_frames(events):
    """
    실제 time_seconds를 사용하여 각 이벤트의 프레임 정보를 계산합니다.
    
    - 이벤트 간 간격을 그대로 유지 (실제 경기 시간 재현)
    - 반환: 각 이벤트의 시작 프레임, 종료 프레임, 프레임 수
    """
    frame_info = []
    current_frame = 0
    
    for i, ev in enumerate(events):
        if i < len(events) - 1:
            next_ev = events[i + 1]
            gap = next_ev['time_seconds'] - ev['time_seconds']
        else:
            # 마지막 장면을 1초간 유지
            gap = 1.0
        
        # 이 구간의 프레임 수
        num_frames = int(gap * FPS)
        num_frames = max(num_frames, 1)  # 최소 1프레임
        
        frame_info.append({
            'event_idx': i,
            'start_frame': current_frame,
            'end_frame': current_frame + num_frames - 1,
            'num_frames': num_frames,
            'event': ev
        })
        
        current_frame += num_frames
    
    total_frames = current_frame
    return frame_info, total_frames

# =========================================================
# 선수 위치 계산 (v4.0: 액션 인덱스 기반)
# =========================================================
def build_player_timeline(filtered_seq, events):
    """
    각 선수별로 이벤트 타임라인을 구축합니다.
    
    반환: {player_id: [(event_idx, start_x, start_y, end_x, end_y, category), ...]}
    """
    player_timeline = {}
    
    for i, ev in enumerate(events):
        pid = ev['player_id']
        if pd.isna(pid):
            continue
        
        if pid not in player_timeline:
            player_timeline[pid] = []
        
        # 좌표를 미터 단위로 변환
        start_x = ev['start_x']
        start_y = ev['start_y']
        end_x = ev['end_x']
        end_y = ev['end_y']
        
        player_timeline[pid].append({
            'event_idx': i,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'category': ev['category']
        })
    
    return player_timeline

def get_player_position_v4_1(player_id, player_timeline, current_frame, frame_info):
    """
    v4.1: 글로벌 프레임 기반 선수 위치 계산
    
    - current_frame: 현재 절대 프레임 번호
    - frame_info: 이벤트별 프레임 범위 정보 (start_frame, end_frame)
    """
    if player_id not in player_timeline or not player_timeline[player_id]:
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False
    
    timeline = player_timeline[player_id]
    
    # 1. 현재 수행 중인 이벤트 확인
    for entry in timeline:
        # 해당 이벤트의 프레임 범위 찾기
        ev_info = frame_info[entry['event_idx']]
        if ev_info['start_frame'] <= current_frame <= ev_info['end_frame']:
            # 현재 이벤트 수행 중
            progress = (current_frame - ev_info['start_frame']) / max(ev_info['num_frames'] - 1, 1)
            
            if entry['category'] == 'carry':
                # 드리블: 시작 -> 종료로 이동
                x = entry['start_x'] + (entry['end_x'] - entry['start_x']) * progress
                y = entry['start_y'] + (entry['end_y'] - entry['start_y']) * progress
            else:
                # 패스/슛: 시작 위치 고정 (사용자 요청: 패스 시 움직이지 않음)
                x, y = entry['start_x'], entry['start_y']
            
            return x, y, True
            
    # 2. 이벤트 사이 이동 (공백 구간)
    # 현재 프레임보다 이전에 끝난 마지막 이벤트와, 이후에 시작할 첫 이벤트를 찾음
    last_event = None
    next_event = None
    
    for entry in timeline:
        ev_info = frame_info[entry['event_idx']]
        if ev_info['end_frame'] < current_frame:
            last_event = entry
        elif ev_info['start_frame'] > current_frame:
            next_event = entry
            break # 첫 번째 미래 이벤트 발견 시 중단
    
    if last_event and next_event:
        # 두 이벤트 사이의 글로벌 이동
        last_ev_info = frame_info[last_event['event_idx']]
        next_ev_info = frame_info[next_event['event_idx']]
        
        # 이동 구간: (이전 이벤트 종료) ~ (다음 이벤트 시작)
        start_move_frame = last_ev_info['end_frame']
        end_move_frame = next_ev_info['start_frame']
        
        # 시작 위치 계산 (이전 이벤트 타입에 따라)
        if last_event['category'] == 'carry':
            start_x, start_y = last_event['end_x'], last_event['end_y']
        else:
            # 패스 후에는 start 위치에 있었음
            start_x, start_y = last_event['start_x'], last_event['start_y']
            
        # 목표 위치 (다음 이벤트 시작점)
        target_x, target_y = next_event['start_x'], next_event['start_y']
        
        # 이동이 필요한지 확인
        if abs(start_x - target_x) > 0.01 or abs(start_y - target_y) > 0.01:
            # 글로벌 진행도 계산
            total_move_frames = end_move_frame - start_move_frame
            current_move_progress = (current_frame - start_move_frame) / max(total_move_frames, 1)
            
            x = start_x + (target_x - start_x) * current_move_progress
            y = start_y + (target_y - start_y) * current_move_progress
        else:
            x, y = start_x, start_y
            
        return x, y, False

    elif last_event:
        # 더 이상 이벤트 없음 -> 마지막 위치 유지
        if last_event['category'] == 'carry':
            return last_event['end_x'], last_event['end_y'], False
        else:
            return last_event['start_x'], last_event['start_y'], False
            
    elif next_event:
        # 아직 첫 이벤트 전 -> 첫 이벤트 시작 위치 대기
        return next_event['start_x'], next_event['start_y'], False
        
    return FIELD_LENGTH / 2, FIELD_WIDTH / 2, False

# =========================================================
# PNG 시퀀스 기반 MP4 생성 (v4.5)
# =========================================================
def load_vaep_data():
    """VAEP 데이터를 로드하고 정규화 계수를 계산합니다."""
    if os.path.exists(VAEP_PATH):
        try:
            df = pd.read_csv(VAEP_PATH)
            # -100 ~ 100 정규화를 위한 스케일러 계산
            vaep_min = df['vaep_value'].min()
            vaep_max = df['vaep_value'].max()
            
            # 0을 기준으로 대칭 정규화 (절대값 최대값 사용)
            abs_max = max(abs(vaep_min), abs(vaep_max))
            if abs_max == 0: abs_max = 1.0
            
            # 스케일링 함수: value / abs_max * 100
            return df, abs_max
        except Exception as e:
            print(f"    [WARN] VAEP 데이터 로드 실패: {e}")
    return None, 1.0

def find_vaep_value(vaep_df, event_row, abs_max):
    """시퀀스 이벤트와 매칭되는 정규화된 VAEP score를 반환합니다."""
    if vaep_df is None:
        return 0.0
        
    mask = (
        (vaep_df['game_id'] == event_row['game_id']) &
        (vaep_df['period_id'] == event_row['period_id']) &
        (np.abs(vaep_df['time_seconds'] - event_row['time_seconds']) < 0.01) &
        (vaep_df['type_name'].str.lower() == event_row['type_name'].lower())
    )
    
    match = vaep_df[mask]
    if len(match) > 0:
        raw_val = match.iloc[0]['vaep_value']
        # -100 ~ 100 정규화
        return (raw_val / abs_max) * 100
    return 0.0

def create_animation_v4(seq_df, sequence_id, output_path, title="전술 패턴"):
    """
    v4.6: PNG 시퀀스 + FFmpeg 기반 애니메이션 생성
    - VAEP Score 정규화 (-100 ~ 100)
    - 성과 이벤트 VAEP 표시 스킵
    - 상단 UI 텍스트 확대 (18pt)
    """
    # VAEP 데이터 및 스케일러 로드
    vaep_df, vaep_abs_max = load_vaep_data()
    
    # [v5.0 수정] 시퀀스 데이터를 시간 순으로 확실하게 정렬 (길이 문제 해결의 핵심)
    # 기존 seq_position 기준보다 time_seconds 기준이 더 정확함
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('time_seconds', ascending=True).reset_index(drop=True)
    
    # 리시브 이벤트 필터링 및 Carry 전환
    processed_events = []
    for _, row in seq.iterrows():
        type_name = row['type_name']
        sx, sy = row['start_x'], row['start_y']
        ex, ey = row['end_x'], row['end_y']
        
        if type_name in ['Pass Received', 'Ball Received']:
            if abs(sx - ex) < 0.001 and abs(sy - ey) < 0.01:
                continue  # 정적 리시브 제외
            else:
                row = row.copy()
                row['type_name'] = 'Carry'
                row['spadl_type'] = 'dribble'
                processed_events.append(row)
        else:
            processed_events.append(row)
    
    if not processed_events:
        print(f"    [SKIP] 시퀀스 {sequence_id}: 유효한 이벤트 없음")
        return None
    
    seq = pd.DataFrame(processed_events)
    
    # 동일 선수의 연속 Carry 병합
    merged_events = []
    for _, row in seq.iterrows():
        if not merged_events:
            merged_events.append(row.to_dict())
            continue
        
        last = merged_events[-1]
        if (row['player_id'] == last['player_id'] and
            get_event_category(row['type_name'], row.get('spadl_type', '')) == 'carry' and
            get_event_category(last['type_name'], last.get('spadl_type', '')) == 'carry'):
            last['end_x'] = row['end_x']
            last['end_y'] = row['end_y']
            last['time_seconds'] = row['time_seconds']
            if row.get('is_outcome', False):
                last['is_outcome'] = True
        else:
            merged_events.append(row.to_dict())
    
    merged_seq = pd.DataFrame(merged_events)
    
    # 정적 리시브 필터링 (성과 이벤트 제외)
    mask = merged_seq['type_name'].isin(['Pass Received', 'Ball Received']) & (~merged_seq['is_outcome'])
    filtered_seq = merged_seq[~mask].reset_index(drop=True)
    
    if len(filtered_seq) == 0:
        print(f"    [SKIP] 시퀀스 {sequence_id}: 필터링 후 이벤트 없음")
        return None
    
    # 이벤트 리스트 생성
    attacking_team_id = filtered_seq['team_id'].iloc[-1]
    start_time = filtered_seq['time_seconds'].iloc[0]
    
    events = []
    for idx, row in filtered_seq.iterrows():
        is_attacking = row['team_id'] == attacking_team_id
        
        # VAEP 값 찾기 및 정규화 (-100 ~ 100)
        vaep_score = find_vaep_value(vaep_df, row, vaep_abs_max)
        
        # 부호 문자열 결정
        vaep_sign = "+" if vaep_score >= 0 else "" 
        
        events.append({
            'player_id': row['player_id'],
            'team_id': row['team_id'],
            'is_attacking': is_attacking,
            'start_x': row['start_x'] * FIELD_LENGTH,
            'start_y': row['start_y'] * FIELD_WIDTH,
            'end_x': row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else row['start_x'] * FIELD_LENGTH,
            'end_y': row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else row['start_y'] * FIELD_WIDTH,
            'type_name': row['type_name'],
            'is_outcome': row.get('is_outcome', False),
            'category': get_event_category(row['type_name'], row.get('spadl_type', '')),
            'color': get_event_color(row['type_name'], row.get('spadl_type', '')),
            'team_color': TEAM_COLORS['attacking'] if is_attacking else TEAM_COLORS['defending'],
            'player_name': row.get('player_name_ko', '')[:6] if pd.notna(row.get('player_name_ko', '')) else '',
            'team_name': row.get('team_name_ko', 'Unknown'),
            'time_seconds': row['time_seconds'] - start_time,
            'vaep_score': vaep_score,
            'vaep_sign': vaep_sign
        })
    
    # 스마트 프레임 계산 (v4.0: 실제 시간 사용)
    frame_info, total_frames = calculate_frames(events)
    
    # 선수 타임라인 구축
    player_timeline = build_player_timeline(filtered_seq, events)
    
    print(f"    프레임 생성 중... (총 {total_frames}프레임, {total_frames/FPS:.1f}초)")
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp(prefix='deTACTer_frames_')
    
    try:
        # 각 프레임 생성
        for frame_idx in range(total_frames):
            # 현재 프레임이 어떤 이벤트에 해당하는지 찾기
            current_event_idx = 0
            local_progress = 0.0
            
            for fi in frame_info:
                if fi['start_frame'] <= frame_idx <= fi['end_frame']:
                    current_event_idx = fi['event_idx']
                    local_progress = (frame_idx - fi['start_frame']) / max(fi['num_frames'] - 1, 1)
                    break
            
            curr_ev = events[current_event_idx]
            
            curr_ev = events[current_event_idx]
            
            # Figure 생성 (1280x770: 사용자 요청으로 세로 길이 약 1.3cm/50px 확장)
            fig, ax = plt.subplots(figsize=(12.8, 7.7), dpi=100)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 여백 제거
            fig.patch.set_facecolor('#1a1a2e')
            draw_pitch(ax) # draw_pitch 내부 ylim은 (-5, 73). 필요시 수정 필요하나 여기서는 함수 밖에서 재설정 불가하므로 draw_pitch 수정 필요.
            # 하지만 draw_pitch가 내부함수가 아니므로 아래에서 set_ylim 호출.
            ax.set_ylim(-5, FIELD_WIDTH + 10) # 상단 여백 추가 (텍스트 잘림 방지)
            
            # 공 경로 (지금까지의 경로)
            ball_path_x = []
            ball_path_y = []
            for i in range(current_event_idx + 1):
                ev = events[i]
                ball_path_x.append(ev['start_x'])
                ball_path_y.append(ev['start_y'])
                if i < current_event_idx:
                    ball_path_x.append(ev['end_x'])
                    ball_path_y.append(ev['end_y'])
            ax.plot(ball_path_x, ball_path_y, '-', color='yellow', linewidth=1.5, alpha=0.3, zorder=15)
            
            # 선수 마커
            for pid, timeline in player_timeline.items():
                px, py, active = get_player_position_v4_1(pid, player_timeline, frame_idx, frame_info)
                
                # 팀 색상 결정
                is_att = any(e['is_attacking'] for e in events if e['player_id'] == pid)
                color = TEAM_COLORS['attacking'] if is_att else TEAM_COLORS['defending']
                
                circ = plt.Circle((px, py), PLAYER_RADIUS, facecolor=color, 
                                 edgecolor='yellow' if active else 'white',
                                 linewidth=3 if active else 1.5, zorder=20)
                ax.add_patch(circ)
                
                # 선수 이름
                player_name = ''
                for e in events:
                    if e['player_id'] == pid:
                        player_name = e['player_name']
                        break
                ax.text(px, py - PLAYER_RADIUS - 0.8, player_name, ha='center', fontsize=7, color='white', zorder=25,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.7, edgecolor='none'))
            
            # 공 위치
            if curr_ev['category'] == 'carry':
                px, py, _ = get_player_position_v4_1(curr_ev['player_id'], player_timeline, frame_idx, frame_info)
                bx, by = px + 1.2, py
            else:
                bx = curr_ev['start_x'] + (curr_ev['end_x'] - curr_ev['start_x']) * local_progress
                by = curr_ev['start_y'] + (curr_ev['end_y'] - curr_ev['start_y']) * local_progress
            
            ball = plt.Circle((bx, by), BALL_RADIUS, facecolor='white', edgecolor='black', linewidth=2, zorder=30)
            ax.add_patch(ball)
            ball_pattern = RegularPolygon((bx, by), 5, radius=BALL_RADIUS*0.6, facecolor='black', zorder=31)
            ax.add_patch(ball_pattern)
            
            # 이벤트 라벨 (v4.6: 성과 이벤트는 VAEP 스킵, 텍스트 확대)
            label_text = f"[{curr_ev['team_name']}] {curr_ev['player_name']}: {curr_ev['type_name']}"
            if not curr_ev['is_outcome']:
                label_text += f" ({curr_ev['vaep_sign']}{curr_ev['vaep_score']:.1f} VAEP score)"
            
            ax.text(FIELD_LENGTH/2, FIELD_WIDTH+4, 
                   label_text,
                   ha='center', fontsize=18, color=curr_ev['color'], fontweight='bold', zorder=40,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#333', alpha=0.9))
            
            # 과거 이벤트 마커
            for i in range(current_event_idx):
                ev = events[i]
                ax.scatter([ev['end_x']], [ev['end_y']], color='yellow', s=40, alpha=0.6, edgecolor='none', zorder=18)

            # PNG 저장 (Size 고정)
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
            plt.savefig(frame_path, dpi=100, facecolor='#1a1a2e')
            plt.close(fig)
            
            if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
                print(f"      진행: {frame_idx+1}/{total_frames} ({100*(frame_idx+1)/total_frames:.1f}%)")
        
        # FFmpeg로 MP4 합치기
        print(f"    MP4 인코딩 시작... (총 {len(os.listdir(temp_dir))} 프레임 이미지)")
        
        # [v5.0 수정] 경로 안정성 확보: temp_dir에서 상대 경로로 인코딩 실행
        # Windows의 백슬래시 문제를 피하기 위해 슬래시로 정규화
        output_abs_path = os.path.abspath(output_path).replace('\\', '/')
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', 
            '-framerate', str(FPS),
            '-i', 'frame_%05d.png',
            '-c:v', 'h264_mf',     # Windows Media Foundation 하드웨어 가속
            '-b:v', '5M',          # 비트레이트 설정
            '-pix_fmt', 'yuv420p',
            '-video_size', '1280x770', # 해상도 명시 (세로 770px)
            output_abs_path
        ]
        
        # [v4.9] FFmpeg 실행 시도
        success = False
        try:
            # cwd를 temp_dir로 지정하여 상대 경로 인식 유도
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, cwd=temp_dir)
            if result.returncode == 0:
                success = True
            else:
                # h264_mf 실패 시 CPU(libx264)로 재시도
                print(f"    [WARN] HW 가속 실패 (코드 {result.returncode}), CPU로 재시도합니다.")
                cpu_cmd = [
                    'ffmpeg', '-y', '-framerate', str(FPS),
                    '-i', 'frame_%05d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-preset', 'veryfast', '-crf', '25',
                    output_abs_path
                ]
                result_cpu = subprocess.run(cpu_cmd, capture_output=True, text=True, cwd=temp_dir)
                if result_cpu.returncode == 0:
                    success = True
        except Exception as e:
            print(f"    [오류] FFmpeg 실행 중 예외 발생: {e}")
            
        if not success:
            # FFmpeg 실패 시 imageio 사용
            try:
                import imageio.v2 as imageio
                frames = []
                file_list = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
                if not file_list:
                    print(f"    [오류] 생성된 프레임 이미지가 없습니다! (temp_dir: {temp_dir})")
                else:
                    for f in file_list:
                        frames.append(imageio.imread(os.path.join(temp_dir, f)))
                    imageio.mimsave(output_path, frames, fps=FPS)
                    print(f"    ✓ MP4 저장 완료 (imageio): {output_path}")
            except ImportError:
                print("    [오류] imageio 라이브러리가 설치되어 있지 않습니다.")
            except Exception as e:
                print(f"    [오류] imageio 인코딩 실패: {e}")
    
    finally:
        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path

# =========================================================
# 정적 플롯
# =========================================================
def plot_sequence_static(seq_df, sequence_id, output_path, title="전술 패턴"):
    """시퀀스의 정적 이미지를 생성합니다."""
    seq = seq_df[seq_df['sequence_id'] == sequence_id].sort_values('seq_position', ascending=False).reset_index(drop=True)
    
    if len(seq) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    draw_pitch(ax)
    # ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15) # 사용자 요청으로 타이틀 제거
    
    attacking_team_id = seq['team_id'].iloc[-1]
    px, py = None, None
    
    for i, row in seq.iterrows():
        x, y = row['start_x'] * FIELD_LENGTH, row['start_y'] * FIELD_WIDTH
        ex = row['end_x'] * FIELD_LENGTH if pd.notna(row['end_x']) else x
        ey = row['end_y'] * FIELD_WIDTH if pd.notna(row['end_y']) else y
        
        is_att = row['team_id'] == attacking_team_id
        t_color = TEAM_COLORS['attacking'] if is_att else TEAM_COLORS['defending']
        c = get_event_color(row['type_name'])
        
        if px is not None:
            ax.plot([px, x], [py, y], '-', color='yellow', alpha=0.3)
        
        if get_event_category(row['type_name']) == 'pass':
            ax.add_patch(FancyArrowPatch((x, y), (ex, ey), arrowstyle='fancy', color=c, alpha=0.6))
        
        ax.scatter([x], [y], color=t_color, s=100, alpha=0.7, edgecolor='white', linewidth=1)
        ax.annotate(f"{i+1}. {row['type_name']}", (x, y), color='white', fontsize=8)
        px, py = ex, ey
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return output_path

# =========================================================
# 클러스터별 애니메이션 생성
# =========================================================
def create_cluster_animations(seq_df, cluster_df, cluster_col, n_clusters=5, n_samples=2):
    """클러스터별 샘플 시퀀스의 애니메이션을 생성합니다."""
    from sklearn.cluster import OPTICS
    
    # OPTICS 상세 클러스터링
    dtw_matrix = np.load(DATA_DIR + 'dtw_distance_matrix.npy')
    optics = OPTICS(min_samples=3, xi=0.03, metric='precomputed')
    optics_labels = optics.fit_predict(dtw_matrix)
    cluster_df = cluster_df.copy()
    cluster_df['optics_detailed'] = optics_labels
    
    # 상위 n_clusters개 클러스터 선택
    label_counts = cluster_df[cluster_col].value_counts()
    top_clusters = [c for c in label_counts.index[:n_clusters] if c != -1]
    
    for cid in top_clusters:
        sids = cluster_df[cluster_df[cluster_col] == cid]['sequence_id'].values[:n_samples]
        for i, sid in enumerate(sids):
            mp4_path = f"{OUTPUT_DIR}{cluster_col}_c{cid}_s{i+1}.mp4"
            png_path = f"{OUTPUT_DIR}{cluster_col}_c{cid}_s{i+1}.png"
            
            print(f"\n[애니메이션] 클러스터 {cid}, 샘플 {i+1}")
            create_animation_v4(seq_df, sid, mp4_path, title=f"클러스터 {cid} - 샘플 {i+1}")
            plot_sequence_static(seq_df, sid, png_path, title=f"클러스터 {cid} - 샘플 {i+1}")

# =========================================================
# 메인 함수
# =========================================================
# =========================================================
# 메인 실행 로직
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="deTACTer Animation Generator")
    parser.add_argument("--sequence_id", type=int, help="특정 시퀀스 ID에 대해서만 애니메이션 생성")
    parser.add_argument("--model", type=str, default=None, help="모델 종류 (GRU, Transformer 등)")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로 (지정하지 않으면 기본 경로 사용)")
    args = parser.parse_args()
    
    # 1. 특정 시퀀스 모드 (run_v50_animations.py에서 호출)
    if args.sequence_id is not None:
        seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
        
        # 출력 경로 결정
        if args.output:
            mp4_path = args.output
        else:
            mp4_path = f"{OUTPUT_DIR}sequence_{args.sequence_id}.mp4"
        
        # 모델명 타이틀 구성
        title = "전술 패턴 분석"
        if args.model:
            title = f"{args.model.upper()} 임베딩 기반 {title}"
            
        print(f"[단일 실행] 시퀀스 {args.sequence_id} 애니메이션 생성 시작")
        create_animation_v4(seq_df, args.sequence_id, mp4_path, title=title)
        
    # 2. 전체 데모 모드 (직접 실행 시)
    else:
        print(f"============================================================")
        print(f"deTACTer v4.4 애니메이션 생성 (전체 모드)")
        print(f"  버전: {VERSION}")
        print(f"============================================================")
        
        seq_df = pd.read_csv(DATA_DIR + 'attack_sequences.csv', encoding='utf-8-sig')
        try:
            cluster_df = pd.read_csv(DATA_DIR + 'cluster_labels.csv', encoding='utf-8-sig')
            # DTW 매트릭스 로드 부분 제거 (v5.5는 좌표 기반)
            # 대신 기존 cluster 컬럼을 활용
            print("[알림] 기존 클러스터 결과(cluster_labels.csv)를 사용하여 예시 생성")
            
            # 클러스터별 예시 생성 (단순화: 각 클러스터 첫번째 샘플 하나씩)
            clusters = sorted(cluster_df['cluster'].unique())
            for cid in clusters:
                if cid == -1: continue
                sid = cluster_df[cluster_df['cluster'] == cid]['sequence_id'].iloc[0]
                mp4_path = f"{OUTPUT_DIR}cluster_{cid}_example.mp4"
                print(f"  -> Cluster {cid} Example: {sid}")
                create_animation_v4(seq_df, sid, mp4_path, title=f"Cluster {cid} Example")
                
        except Exception as e:
            print(f"[오류] 클러스터 예시 생성 실패: {e}")
            # 데모 시퀀스 하나라도 생성 시도
            if len(seq_df) > 0:
                sid = seq_df['sequence_id'].iloc[0]
                create_animation_v4(seq_df, sid, OUTPUT_DIR + 'demo_sequence.mp4', title="Demo Sequence")

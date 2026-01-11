# =========================================================
# preprocessing.py
# deTACTer 프로젝트용 통합 데이터 전처리 모듈
# =========================================================
# 주요 기능:
# 1. 인코딩 및 문자열 정제 (utf-8-sig)
# 2. 골키퍼 위치 기반 공격 방향 통일 (L->R)
# 3. 좌표 정규화 (105x68 -> 0~1) 및 클리핑
# 4. SPADL 변환 로직 통합
# 5. 결측치는 원본 유지 (Handle Nothing)
# =========================================================

import pandas as pd
import numpy as np
import yaml
import os
import sys

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =========================================================
# 설정 로드
# =========================================================
# 설정 로드
# 설정 로드
# v5.9: 경로 소프트 코딩 및 단일 소스(config.yaml) 참조 강화
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 버전 설정 로드
VERSION = config['version']

RAW_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, config['data']['raw_data_path']))
MATCH_INFO_PATH = os.path.normpath(os.path.join(BASE_DIR, config['data']['match_info_path']))

# 버전별 출력 경로 설정
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "refined", VERSION))
OUTPUT_PATH = os.path.normpath(os.path.join(OUTPUT_DIR, "preprocessed_data.csv"))

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 필드 규격 (국제 표준)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# =========================================================
# 1. 데이터 로드 (인코딩: utf-8-sig)
# =========================================================
def load_data():
    """raw_data.csv와 match_info.csv를 로드합니다."""
    print(f"[1/5] 데이터 로드 중... ({RAW_DATA_PATH})")
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    match_info = pd.read_csv(MATCH_INFO_PATH, encoding='utf-8-sig')
    
    # 문자열 컬럼 공백 제거
    str_cols = ['player_name_ko', 'team_name_ko', 'position_name', 'main_position']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    print(f"    -> 로드 완료: {len(df):,} rows")
    return df, match_info

# =========================================================
# 2. 공격 방향 통일 (L->R) - 골키퍼 위치 기반
# =========================================================
def unify_attack_direction(df, match_info):
    """
    [v3.1 수정] 팀별 좌표 반전 로직을 제거합니다.
    좌표 반전은 이제 시퀀스 추출 단계에서 '공격 팀' 기준으로 일괄 수행됩니다.
    이 함수는 공격 방향 판별을 위한 기초 데이터(GK 평균 위치 등)가 필요한 경우를 위해
    데이터만 정제하거나, 현재는 로직을 스킵합니다.
    """
    print("[2/5] 공격 방향 통일 로직 스킵 (시퀀스 추출 단계로 이관)...")
    return df

# =========================================================
# 3. 좌표 정규화 (0~1) 및 클리핑
# =========================================================
def normalize_and_clip_coordinates(df):
    """좌표를 0~1로 정규화하고, 범위 밖 값은 클리핑합니다."""
    print("[3/5] 좌표 정규화 및 클리핑...")
    
    coord_cols_x = ['start_x', 'end_x']
    coord_cols_y = ['start_y', 'end_y']
    
    for col in coord_cols_x:
        if col in df.columns:
            # 클리핑 후 정규화
            df[col] = df[col].clip(0, FIELD_LENGTH) / FIELD_LENGTH
    
    for col in coord_cols_y:
        if col in df.columns:
            df[col] = df[col].clip(0, FIELD_WIDTH) / FIELD_WIDTH
    
    # dx, dy도 정규화 (부호 유지)
    if 'dx' in df.columns:
        df['dx'] = df['dx'] / FIELD_LENGTH
    if 'dy' in df.columns:
        df['dy'] = df['dy'] / FIELD_WIDTH
    
    print("    -> 정규화 및 클리핑 완료")
    return df

# =========================================================
# 4. SPADL 변환 로직 (convert_to_spadl.py에서 통합)
# =========================================================
def map_action_type(type_name):
    """이벤트 타입을 SPADL 표준 액션 타입으로 매핑합니다."""
    type_name = str(type_name).lower()
    if 'pass' in type_name:
        if 'freekick' in type_name: return 'freekick_short'
        if 'corner' in type_name: return 'corner_short'
        return 'pass'
    if 'cross' in type_name: return 'cross'
    if 'throw-in' in type_name: return 'throw_in'
    if 'shot' in type_name:
        if 'freekick' in type_name: return 'shot_freekick'
        if 'penalty' in type_name: return 'shot_penalty'
        return 'shot'
    if 'goal' in type_name and 'kick' in type_name: return 'goalkick'
    if 'goal' in type_name: return 'shot'
    if 'carry' in type_name: return 'dribble'
    if 'take-on' in type_name: return 'take_on'
    if 'tackle' in type_name: return 'tackle'
    if 'interception' in type_name: return 'interception'
    if 'clearance' in type_name: return 'clearance'
    if 'foul' in type_name: return 'foul'
    if 'keeper' in type_name or 'save' in type_name or 'catch' in type_name or 'parry' in type_name: 
        return 'keeper_save'
    return 'non_action'

def map_result(result_name):
    """결과를 success/fail로 매핑합니다."""
    if result_name in ['Successful', 'Goal']:
        return 'success'
    return 'fail'

def apply_spadl_mapping(df):
    """SPADL 타입 및 결과 매핑을 적용합니다."""
    print("[4/5] SPADL 매핑 적용...")
    
    df['spadl_type'] = df['type_name'].apply(map_action_type)
    df['spadl_result'] = df['result_name'].apply(map_result)
    
    # non_action 필터링은 여기서 하지 않음 (시퀀스 추출 시 처리)
    print(f"    -> SPADL 매핑 완료 (non_action: {(df['spadl_type'] == 'non_action').sum():,}개)")
    return df

# =========================================================
# 5. 템포 피처 계산 (dx, dy, dt)
# =========================================================
def calculate_tempo_features(df):
    """액션 간 템포 피처를 계산합니다."""
    print("[5/5] 템포 피처 계산...")
    
    # dt 계산 (이전 액션과의 시간 차이)
    df = df.sort_values(['game_id', 'period_id', 'time_seconds'])
    df['dt'] = df.groupby(['game_id', 'period_id'])['time_seconds'].diff().fillna(0)
    
    # 속도 계산 (거리 / 시간) - dt가 0이면 무한대 방지
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed'] = np.where(df['dt'] > 0, df['distance'] / df['dt'], 0)
    
    print("    -> 템포 피처(dt, distance, speed) 추가 완료")
    return df

# =========================================================
# 5.5. Skip Connected(투명) 이벤트 제거
# =========================================================
# =========================================================
# 5.5. Disconnected Event (is_dis) 판별
# =========================================================
def calculate_is_disconnected(df):
    """
    전후 이벤트 위치가 연결되지 않는 지점을 판별합니다.
    - 정의: Prev.End (x, y) != Curr.Start (x, y)
    - 미세한 오차(EPSILON)를 제외하고 좌표가 물리적으로 끊겨 있으면 is_dis = True
    """
    print("[5.5/6] Disconnected Event (단절점) 판별 중...")
    
    # 시간 순 정렬
    df = df.sort_values(['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    # 이전 이벤트의 종료 좌표 (Shift 1)
    prev_end_x = df.groupby(['game_id', 'period_id'])['end_x'].shift(1)
    prev_end_y = df.groupby(['game_id', 'period_id'])['end_y'].shift(1)
    
    # 현재 이벤트의 시작 좌표
    curr_start_x = df['start_x']
    curr_start_y = df['start_y']
    
    # 거리 차이 계산
    EPSILON = 1e-3  # 1mm 정도의 오차 허용 (정규화된 좌표 기준이면 더 작아야 할 수 있음)
    # 현재 좌표가 0~1 정규화되어 있으므로, 1e-3은 약 10cm(105m 기준) 내외
    
    gap_dist = np.sqrt((prev_end_x - curr_start_x)**2 + (prev_end_y - curr_start_y)**2)
    
    # 첫 번째 이벤트(NaN)는 단절로 보지 않음 (시퀀스의 시작일 수는 있어도 '중간 단절'은 아님)
    # 하지만 로직상 True로 두면 '이전과 끊김 -> 여기서부터 시작'이라는 의미로 해석 가능
    # 여기서는 '중간 단절'을 찾는 것이므로 NaN은 False로 처리
    df['is_dis'] = gap_dist > EPSILON
    df['is_dis'] = df['is_dis'].fillna(False)
    
    n_dis = df['is_dis'].sum()
    print(f"    -> 단절점 판별 완료: {n_dis:,}건 발견")
    return df

# =========================================================
# 6. 리시브 로직 보정 및 궤적 연결 (v5.3 이전 통합)
# =========================================================
def refine_receiving_and_trajectories(df):
    """
    리시브 이벤트를 정제하고 파편화된 핵심 궤적을 연결합니다.
    - 제자리 리시브: 삭제
    - 이동 리시브: Carry로 변환
    - 액션 간 간극 보정: Carry 삽입 또는 연결
    """
    print("[6/6] 리시브 보정 및 궤적 연결 중 (v5.3)...")
    
    # 정렬 보장
    df = df.sort_values(['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    refined_rows = []
    rows = [row.to_dict() for _, row in df.iterrows()]
    
    i = 0
    while i < len(rows):
        curr = rows[i]
        
        if curr.get('type_name') == 'Pass Received' or curr.get('type_name') == 'Ball Received':
            start_x, start_y = curr['start_x'], curr['start_y']
            end_x, end_y = curr['end_x'], curr['end_y']
            dist = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
            is_moving = dist > 1e-5
            
            next_node = rows[i+1] if i + 1 < len(rows) else None
            
            # [1] 중복 리시브 삭제 (다음 액션과 위치가 같음)
            if next_node is not None and \
               abs(start_x - next_node['start_x']) < 1e-5 and \
               abs(start_y - next_node['start_y']) < 1e-5:
                i += 1
                continue
            
            # [2] 이동 리시브 처리
            if is_moving:
                if next_node is not None and next_node.get('type_name') == 'Carry':
                    # 다음 Carry의 시작을 이 리시브의 시작으로 연장하고 리시브 삭제
                    rows[i+1]['start_x'] = start_x
                    rows[i+1]['start_y'] = start_y
                    i += 1
                    continue
                else:
                    # 리시브를 Carry로 변환
                    curr['type_name'] = 'Carry'
                    curr['spadl_type'] = 'dribble'
            
            # [3] 정적 리시브 간극 보정
            elif next_node is not None and next_node.get('type_name') != 'Carry':
                gap = np.sqrt((end_x - next_node['start_x'])**2 + (end_y - next_node['start_y'])**2)
                if gap > 1e-5:
                    # 리시브를 Carry로 변환하여 경로 연결
                    curr['type_name'] = 'Carry'
                    curr['spadl_type'] = 'dribble'
                    curr['end_x'] = next_node['start_x']
                    curr['end_y'] = next_node['start_y']
            
            # 여전히 Pass Received이고 안 움직였다면 삭제
            if curr.get('type_name') == 'Pass Received' or curr.get('type_name') == 'Ball Received' and not is_moving:
                i += 1
                continue

        refined_rows.append(curr)
        i += 1
        
    refined_df = pd.DataFrame(refined_rows)
    print(f"    -> 정제 완료: {len(df):,} -> {len(refined_df):,} rows")
    return refined_df

# =========================================================
# 메인 파이프라인
# =========================================================
def run_preprocessing():
    """전체 전처리 파이프라인을 실행합니다."""
    print("=" * 60)
    print("deTACTer 데이터 전처리 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    df, match_info = load_data()
    
    # 2. 공격 방향 통일
    df = unify_attack_direction(df, match_info)
    
    # 3. 좌표 정규화 및 클리핑
    df = normalize_and_clip_coordinates(df)
    
    # 4. SPADL 매핑
    df = apply_spadl_mapping(df)
    
    # 5. 템포 피처
    df = calculate_tempo_features(df)
    
    # 6. 리시브 보정 및 궤적 연결 (v5.3)
    df = refine_receiving_and_trajectories(df)

    # 7. Disconnected Event 판별 (v5.5) - 보정 후 수행
    df = calculate_is_disconnected(df)
    
    # 저장
    print("=" * 60)
    print(f"전처리 완료! 저장 중... ({OUTPUT_PATH})")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {len(df):,} rows")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    run_preprocessing()

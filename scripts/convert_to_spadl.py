
import pandas as pd
import yaml
import os
import numpy as np
import socceraction.spadl.config as spadlc

# 설정 로드
# v5.9: 경로 소프트 코딩 및 전처리된 데이터 연동 강화
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VERSION = config['version']
RAW_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, config['data']['raw_data_path']))
CLEAN_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "data", "refined", VERSION, "preprocessed_data.csv"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, config['data']['spadl_output_dir']))

if not os.path.exists(CLEAN_DATA_PATH):
    print(f"Warning: Cleaned data not found at {CLEAN_DATA_PATH}. Using raw data.")
    DATA_TO_LOAD = RAW_DATA_PATH
else:
    print(f"Using preprocessed data: {CLEAN_DATA_PATH}")
    DATA_TO_LOAD = CLEAN_DATA_PATH

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"Loading data from {DATA_TO_LOAD}...")
    df = pd.read_csv(DATA_TO_LOAD, encoding='utf-8-sig')
    return df

def map_action_type(type_name):
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
    if 'goal' in type_name: return 'shot' # Fallback for just 'Goal'
    if 'carry' in type_name: return 'dribble' # SPADL dribble is a carry
    if 'take-on' in type_name: return 'take_on'
    if 'tackle' in type_name: return 'tackle'
    if 'interception' in type_name: return 'interception'
    if 'clearance' in type_name: return 'clearance'
    if 'foul' in type_name: return 'foul'
    if 'keeper' in type_name or 'save' in type_name or 'catch' in type_name or 'parry' in type_name: return 'keeper_save'
    return 'non_action'

def map_result(result_name):
    if result_name == 'Successful' or result_name == 'Goal':
        return 'success'
    return 'fail'

def convert_to_spadl(df):
    spadl_actions = []
    
    # Ensure sorted
    df = df.sort_values(by=['game_id', 'period_id', 'time_seconds'])
    
    # Renaming and basic transform
    df['type_id'] = df.apply(lambda row: -1, axis=1) # Placeholder, actual SPADL relies on names often or standard IDs
    # Mapping Types
    # We will produce a DF with columns expected by socceraction
    # game_id, period_id, time_seconds, team_id, player_id, start_x, start_y, end_x, end_y, type_id, result_id, bodypart_id, action_id
    
    # Standard SPADL type mapping (simplified for this custom data)
    # Ideally we'd use socceraction's internal mapping but we need to map OUR strings to THEIR schema.
    # For VAEP training, consistent names are key.
    
    spadl_df = pd.DataFrame()
    spadl_df['game_id'] = df['game_id']
    spadl_df['period_id'] = df['period_id']
    spadl_df['time_seconds'] = df['time_seconds']
    spadl_df['team_id'] = df['team_id']
    spadl_df['player_id'] = df['player_id']
    spadl_df['start_x'] = df['start_x']
    spadl_df['start_y'] = df['start_y']
    spadl_df['end_x'] = df['end_x']
    spadl_df['end_y'] = df['end_y']
    if 'sequence_id' in df.columns:
        spadl_df['sequence_id'] = df['sequence_id']
    spadl_df['original_event_type'] = df['type_name']
    
    # Map type_name to standardized SPADL names
    spadl_df['type_name'] = df['type_name'].apply(map_action_type)
    
    # Map result_name to 1 (success) or 0 (fail)
    spadl_df['result_id'] = df['result_name'].apply(lambda x: 1 if x in ['Successful', 'Goal'] else 0)
    spadl_df['result_name'] = df['result_name'].apply(map_result)
    
    # Bodypart - Default to foot (0) as we don't have it
    spadl_df['bodypart_id'] = 0 
    spadl_df['bodypart_name'] = 'foot'
    
    # Filter out non-actions
    spadl_df = spadl_df[spadl_df['type_name'] != 'non_action']
    
    # Add proper type_ids based on spadl config if possible, OR just keep names for VAEP features
    # socceraction VAEP features often use the names directly or ID map. 
    # Let's try to map names to IDs using spadl.config.actiontypes
    
    # In socceraction v1.x, actiontypes is a list.
    action_map = {name: i for i, name in enumerate(spadlc.actiontypes)}
    spadl_df['type_id'] = spadl_df['type_name'].apply(lambda x: action_map.get(x, -1))
    
    # Filter unknown actions
    spadl_df = spadl_df[spadl_df['type_id'] != -1]
    
    return spadl_df

def main():
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    spadl_actions = convert_to_spadl(df)
    print(f"Converted to {len(spadl_actions)} SPADL actions.")
    
    # Save per game
    for game_id, game_actions in spadl_actions.groupby('game_id'):
        output_path = os.path.join(OUTPUT_DIR, f"game_{game_id}.csv")  # Using csv for simpler inspection, or h5
        game_actions.to_csv(output_path, index=False)
        
    print(f"Saved SPADL data to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

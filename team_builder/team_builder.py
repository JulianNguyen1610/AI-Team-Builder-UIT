import pandas as pd
import joblib
import os

from config import FILE_PATH, MODEL_STORAGE_PATH, ID_COLUMN, POSITION_COLUMN, NAME_COLUMN, OVERALL_COLUMN, FORMATION_SLOTS, POSITION_REQUIREMENTS_DETAILED, POSITION_GROUPS
from genetic_optimizer import GeneticTeamBuilder
from utils import calculate_ml_suitability_score, get_player_group
from model_trainer import preprocess_data

def load_models():
    print("Đang tải các mô hình AI chuyên gia...")
    models = {}
    for group_name in POSITION_GROUPS.keys():
        model_filename = os.path.join(MODEL_STORAGE_PATH, f"model_{group_name.lower()}.joblib")
        try:
            models[group_name] = joblib.load(model_filename)
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file mô hình tại '{model_filename}'.")
    return models

def build_team(dataframe, filter_name, formation_key, models_dict, filter_type='team', use_genetic_algo=True, tactic_name='Balanced (Cân bằng)'):
    """
    Hàm chính để xây dựng đội hình (V3 - Có tham số Tactic).
    """
    filter_type_name = "đội bóng" if filter_type == 'team' else "quốc gia"
    algo_name = "Genetic Algorithm" if use_genetic_algo else "Greedy Algorithm"
    
    print(f"\nBắt đầu xây dựng đội hình cho {filter_type_name} '{filter_name}'")
    print(f"Sơ đồ: {formation_key} | Thuật toán: {algo_name} | Chiến thuật: {tactic_name}")
    
    # Lọc cầu thủ
    if filter_type == 'nation':
        if 'Nation' not in dataframe.columns:
             return None
        potential_players = dataframe[dataframe['Nation'].str.contains(filter_name, case=False, na=False)].copy()
    else:
        if 'team_color' not in dataframe.columns:
             return None
        potential_players = dataframe[dataframe['team_color'].str.contains(filter_name, case=False, na=False)].copy()
    
    if potential_players.empty:
        return None
    
    print(f"Tìm thấy {len(potential_players)} ứng viên tiềm năng.")
    slots_to_fill = FORMATION_SLOTS[formation_key]
    
    # --- NHÁNH 1: GENETIC ALGORITHM ---
    if use_genetic_algo:
        optimizer = GeneticTeamBuilder(potential_players, slots_to_fill, models_dict, tactic_name)
        final_team = optimizer.run()
        return final_team

    # --- NHÁNH 2: GREEDY ALGORITHM ---
    final_team = []
    used_player_ids = set()

    for position_slot in slots_to_fill:
        best_player_for_slot = None
        max_score = -100

        acceptable_native_positions = POSITION_REQUIREMENTS_DETAILED.get(position_slot, {}).get('main_positions', [])
        candidates = potential_players[potential_players[POSITION_COLUMN].isin(acceptable_native_positions)]
        
        if candidates.empty:
            required_pos_group = get_player_group(position_slot)
            if required_pos_group:
                group_positions = POSITION_GROUPS.get(required_pos_group, [])
                candidates = potential_players[potential_players[POSITION_COLUMN].isin(group_positions)]
        
        if candidates.empty: candidates = potential_players

        for index, current_player in candidates.iterrows():
            if current_player[ID_COLUMN] in used_player_ids: continue
            
            score = calculate_ml_suitability_score(current_player, position_slot, models_dict, tactic_name)
            
            if score > max_score:
                max_score = score
                best_player_for_slot = current_player
        
        if best_player_for_slot is not None:
            final_team.append(best_player_for_slot)
            used_player_ids.add(best_player_for_slot[ID_COLUMN])
            
    return final_team

if __name__ == "__main__":
    # Test chạy thử (không cần sửa)
    pass
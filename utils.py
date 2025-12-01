import pandas as pd
from config import (
    POSITION_COLUMN, OVERALL_COLUMN, FEATURES_COLUMNS, 
    POSITION_GROUPS, POSITION_REQUIREMENTS_DETAILED
)

# --- CÁC HẰNG SỐ CẤU HÌNH ---
PASSING_THRESHOLD = 83          
INVERTED_FB_THRESHOLD = 82      
CB_MIN_HEIGHT = 183             
MIN_WINGER_PACE = 83            

# --- CẤU HÌNH LOGIC CHÂN THUẬN ---
STRICT_RIGHT_SIDED_POSITIONS = ['RB', 'RWB', 'RM']
STRICT_LEFT_SIDED_POSITIONS = ['LB', 'LWB', 'LM']

def get_player_group(player_position):
    """Hàm phụ để tìm ra nhóm vị trí của một cầu thủ"""
    if pd.isna(player_position): return None
    for group, positions in POSITION_GROUPS.items():
        if player_position in positions:
            return group
    return None

def calculate_ml_suitability_score(player_row, required_position_key, models_dict):
    """
    Phiên bản Tối ưu hóa: Kiểm tra xem đã có điểm dự đoán sẵn chưa.
    """
    player_pos_str = player_row[POSITION_COLUMN]
    player_pos_group = get_player_group(player_pos_str)
    required_pos_group = get_player_group(required_position_key)

    # --- TỐI ƯU HÓA TẠI ĐÂY ---
    # Kiểm tra xem trong dữ liệu cầu thủ đã có sẵn điểm dự đoán (base_ml_score) chưa
    # Nếu có rồi (do tính batch bên genetic_optimizer), lấy dùng luôn -> Cực nhanh
    if 'base_ml_score' in player_row and pd.notna(player_row['base_ml_score']):
         predicted_score = player_row['base_ml_score']
    else:
        # Nếu chưa có (trường hợp chạy lẻ), thì mới gọi mô hình để dự đoán -> Chậm hơn
        model = models_dict.get(required_pos_group)
        if model is None: return player_row[OVERALL_COLUMN]
        
        player_features = pd.DataFrame([player_row[FEATURES_COLUMNS]])
        predicted_score = model.predict(player_features)[0]
    # ---------------------------

    # ... (TOÀN BỘ PHẦN LOGIC THƯỞNG/PHẠT BÊN DƯỚI GIỮ NGUYÊN) ...
    # Copy y nguyên phần logic Height, Inverted FB, Pace, v.v. từ phiên bản cũ vào đây
    
    # --- LOGIC THỂ HÌNH CHO CB ---
    if required_position_key in ['CB', 'RCB', 'LCB']:
        height = player_row.get('Height_clean', 180)
        if height < CB_MIN_HEIGHT:
            predicted_score -= (CB_MIN_HEIGHT - height) * 1.5
        else:
            predicted_score += (height - CB_MIN_HEIGHT) * 0.2
            
    # ... (Giữ nguyên các logic khác của bạn ở đây) ...
    # Để ngắn gọn tôi không paste lại hết, hãy đảm bảo bạn giữ lại đầy đủ logic V7
    
    # Copy logic Hậu vệ đá tiền vệ...
    is_defender_origin = player_pos_group == 'Defender'
    is_midfield_target = required_pos_group == 'Midfielder'

    if is_defender_origin and is_midfield_target:
        # A. Hậu vệ biên (RB/LB) -> Tiền vệ
        if player_pos_str in ['RB', 'LB', 'RWB', 'LWB']:
            short_pass = player_row.get('Short Passing', 0)
            long_pass = player_row.get('Long Passing', 0)
            vision = player_row.get('Vision', 0)
            dribbling = player_row.get('Dribbling', 0)
            agility = player_row.get('Agility', 0)
            long_shots = player_row.get('Long Shots', 0)
            shot_power = player_row.get('Shot Power', 0)
            finishing = player_row.get('Finishing', 0) 

            inverted_ability = (
                ((short_pass + long_pass + vision) / 3 * 0.5) + 
                ((dribbling + agility) / 2 * 0.3) + 
                ((long_shots + shot_power + finishing) / 3 * 0.2) 
            )

            if inverted_ability >= INVERTED_FB_THRESHOLD: 
                predicted_score += 4.0 
            else:
                predicted_score *= 0.6 

        # B. Trung vệ (CB) -> Tiền vệ
        elif player_pos_str in ['CB', 'RCB', 'LCB', 'SW']:
            short_pass = player_row.get('Short Passing', 0)
            long_pass = player_row.get('Long Passing', 0)
            ball_control = player_row.get('Ball Control', 0)
            if (short_pass + long_pass + ball_control)/3 >= 75: predicted_score += 3.0
            else: predicted_score *= 0.7

    # --- LOGIC PHÂN ĐỊNH VAI TRÒ CDM vs CM ---
    if required_position_key in ['CDM', 'RDM', 'LDM']:
        if player_pos_str == 'CDM': predicted_score += 3.0
        shooting = player_row.get('SHO', 0)
        if shooting > 78: predicted_score -= (shooting - 78) * 0.5
            
    if required_position_key in ['CM', 'RCM', 'LCM']:
        shooting = player_row.get('SHO', 0)
        if shooting > 75: predicted_score += 2.0

    # --- LOGIC ƯU TIÊN WEAK FOOT CHO CAM ---
    if required_position_key == 'CAM':
        weak_foot = float(player_row.get('Weak foot', 3))
        if weak_foot >= 5: predicted_score += 3.0 
        elif weak_foot == 4: predicted_score += 1.5 
        elif weak_foot <= 2: predicted_score -= 2.0 
        shooting = player_row.get('SHO', 0)
        if shooting > 75: predicted_score += 1.5

    # --- LOGIC PHẠT TỐC ĐỘ WINGER ---
    if required_position_key in ['RW', 'LW', 'RM', 'LM', 'RAM', 'LAM']:
        pace = player_row.get('PAC', 0)
        if pace < MIN_WINGER_PACE:
            predicted_score -= (MIN_WINGER_PACE - pace) * 1.2
            if pace < 75: predicted_score -= 8.0

    # --- LOGIC ƯU TIÊN ST ---
    if required_position_key == 'ST':
        if player_pos_str in ['ST', 'CF']: predicted_score += 2.5

    # --- LOGIC CHÂN THUẬN ---
    preferred_foot = str(player_row.get('Preferred foot', '')).strip().lower() 
    weak_foot = float(player_row.get('Weak foot', 3))
    
    if required_position_key in STRICT_RIGHT_SIDED_POSITIONS:
        if preferred_foot == 'right': predicted_score += 2.0 
        elif preferred_foot == 'left': predicted_score -= (5 - weak_foot) * 2.0 
    elif required_position_key in STRICT_LEFT_SIDED_POSITIONS:
        if preferred_foot == 'left': predicted_score += 2.0 
        elif preferred_foot == 'right': predicted_score -= (5 - weak_foot) * 2.0

    # --- LOGIC CƠ BẢN ---
    acceptable_native_positions = POSITION_REQUIREMENTS_DETAILED.get(required_position_key, {}).get('main_positions', [])
    if player_pos_str in acceptable_native_positions:
        predicted_score += 1.5
    elif player_pos_group == required_pos_group:
        predicted_score += 0.5
    else:
        if not (is_defender_origin and is_midfield_target):
            predicted_score *= 0.6

    return predicted_score
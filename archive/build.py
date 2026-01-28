import pandas as pd

# --- PHẦN CẤU HÌNH BAN ĐẦU  ---
FILE_PATH = r'D:\ai002\archive\male_players_final.csv' # Đường dẫn đến file dữ liệu của bạn
ID_COLUMN = 'ID'                      # CẬP NHẬT: Tên cột ID duy nhất
POSITION_COLUMN = 'Position'          # CẬP NHẬT: Tên cột vị trí chính
NAME_COLUMN = 'Name'                  # CẬP NHẬT: Tên cột tên cầu thủ
OVERALL_COLUMN = 'OVR'                # CẬP NHẬT: Tên cột chỉ số tổng thể

# --- NẠP VÀ LÀM SẠCH DỮ LIỆU ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Tải dữ liệu thành công! Dữ liệu có {df.shape[0]} cầu thủ và {df.shape[1]} cột.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại '{FILE_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Làm sạch cột team_color, Nation và cột vị trí
df['team_color'] = df['team_color'].fillna('')
if 'Nation' in df.columns:
    df['Nation'] = df['Nation'].fillna('')
# Chỉ lấy vị trí chính đầu tiên nếu có nhiều vị trí (ví dụ: "RB, RWB" -> "RB")
df[POSITION_COLUMN] = df[POSITION_COLUMN].astype(str).apply(lambda x: x.split(',')[0].strip())

print("Làm sạch dữ liệu cơ bản hoàn tất.")


# --- TỪ ĐIỂN KIẾN THỨC VỀ VỊ TRÍ (VỚI CÁC CHỈ SỐ CHI TIẾT) ---
POSITION_REQUIREMENTS = {
    # Hàng thủ
    'GK': {
        'main_positions': ['GK'], 
        'key_stats': ['GK Diving', 'GK Handling', 'GK Kicking', 'GK Reflexes', 'GK Positioning'],
        'preferred_foot': None,  # GK không cần quan tâm chân thuận
        'weak_foot_bonus': 0  # Weak foot không quan trọng cho GK
    },
    'CB': {
        'main_positions': ['CB','CDM','RB','LB'], 
        'key_stats': ['PAC','Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Heading Accuracy', 'Interceptions', 'Strength', 'Jumping'],
        'preferred_foot': None,  # CB có thể dùng cả 2 chân
        'weak_foot_bonus': 0.5  # Weak foot tốt là lợi thế nhỏ
    },
    'RB': {
        'main_positions': ['CB','LB','RB', 'RWB'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Standing Tackle', 'Sliding Tackle', 'Crossing', 'Dribbling', 'Stamina'],
        'preferred_foot': 'Right',  # RB thường dùng chân phải
        'weak_foot_bonus': 1.0  # Weak foot tốt rất quan trọng cho fullback
    },
    'LB': {
        'main_positions': ['CB','RB','LB', 'LWB'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Standing Tackle', 'Sliding Tackle', 'Crossing', 'Dribbling', 'Stamina'],
        'preferred_foot': 'Left',  # LB thường dùng chân trái
        'weak_foot_bonus': 1.0  # Weak foot tốt rất quan trọng cho fullback
    },
    
    # Hàng tiền vệ
    'CDM': {
        'main_positions': ['CDM','CB','CAM'], 
        'key_stats': ['PAC','Interceptions', 'Standing Tackle', 'Sliding Tackle', 'Short Passing', 'Long Passing', 'Strength', 'Stamina'],
        'preferred_foot': None,  # CDM có thể dùng cả 2 chân
        'weak_foot_bonus': 1.5  # Weak foot tốt rất quan trọng cho CDM
    },
    'CM': {
        'main_positions': ['CM','CDM','CAM','RM','LM'], 
        'key_stats': ['PAC','Short Passing', 'Long Passing', 'Vision', 'Dribbling', 'Ball Control', 'Stamina', 'Reactions'],
        'preferred_foot': None,  # CM có thể dùng cả 2 chân
        'weak_foot_bonus': 2.0  # Weak foot tốt cực kỳ quan trọng cho CM
    },
    'CAM': {
        'main_positions': ['CAM', 'CM','RM','LM','RW','LW','CF'], 
        'key_stats': ['PAC','Vision', 'Short Passing', 'Long Passing', 'Dribbling', 'Ball Control', 'Finishing', 'Long Shots', 'Reactions'],
        'preferred_foot': None,  # CAM có thể dùng cả 2 chân
        'weak_foot_bonus': 2.5  # Weak foot tốt cực kỳ quan trọng cho CAM
    },
    'RAM': {
        'main_positions': ['CAM', 'CM','RM','LM','RW','LW','CF'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Dribbling', 'Crossing', 'Short Passing', 'Finishing', 'Stamina'],
        'preferred_foot': 'Right',  # RAM thường dùng chân phải
        'weak_foot_bonus': 2.0  # Weak foot tốt rất quan trọng
    },
    'LAM': {
        'main_positions': ['CAM', 'CM','RM','LM','RW','LW','CF'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Dribbling', 'Crossing', 'Short Passing', 'Finishing', 'Stamina'],
        'preferred_foot': 'Left',  # LAM thường dùng chân trái
        'weak_foot_bonus': 2.0  # Weak foot tốt rất quan trọng
    },
    'RCM': {
        'main_positions': ['CM','CDM','CAM','RM','LM'], 
        'key_stats': ['Short Passing', 'Long Passing', 'Vision', 'Dribbling', 'Ball Control', 'Stamina', 'Reactions'],
        'preferred_foot': 'Right',  # RCM ưu tiên chân phải
        'weak_foot_bonus': 2.0  # Weak foot tốt cực kỳ quan trọng cho CM
    },
    'LCM': {
        'main_positions': ['CM','CDM','CAM','RM','LM'], 
        'key_stats': ['PAC','Short Passing', 'Long Passing', 'Vision', 'Dribbling', 'Ball Control', 'Stamina', 'Reactions'],
        'preferred_foot': 'Left',  # LCM ưu tiên chân trái
        'weak_foot_bonus': 2.0  # Weak foot tốt cực kỳ quan trọng cho CM
    },
    'RDM': {
        'main_positions': ['CDM','CB','CAM'], 
        'key_stats': ['PAC','Interceptions', 'Standing Tackle', 'Sliding Tackle', 'Short Passing', 'Long Passing', 'Strength', 'Stamina'],
        'preferred_foot': 'Right',  # RDM ưu tiên chân phải
        'weak_foot_bonus': 1.5  # Weak foot tốt rất quan trọng cho CDM
    },
    'LDM': {
        'main_positions': ['CDM','CB','CAM'], 
        'key_stats': ['PAC','Interceptions', 'Standing Tackle', 'Sliding Tackle', 'Short Passing', 'Long Passing', 'Strength', 'Stamina'],
        'preferred_foot': 'Left',  # LDM ưu tiên chân trái
        'weak_foot_bonus': 1.5  # Weak foot tốt rất quan trọng cho CDM
    },
    'RCB': {
        'main_positions': ['CB','CDM','RB','LB'], 
        'key_stats': ['PAC','Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Heading Accuracy', 'Interceptions', 'Strength', 'Jumping'],
        'preferred_foot': 'Right',  # RCB ưu tiên chân phải
        'weak_foot_bonus': 0.5  # Weak foot tốt là lợi thế nhỏ
    },
    'LCB': {
        'main_positions': ['CB','CDM','RB','LB'], 
        'key_stats': ['PAC','Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Heading Accuracy', 'Interceptions', 'Strength', 'Jumping'],
        'preferred_foot': 'Left',  # LCB ưu tiên chân trái
        'weak_foot_bonus': 0.5  # Weak foot tốt là lợi thế nhỏ
    },
    'SW': {
        'main_positions': ['SW', 'CB','CDM'], 
        'key_stats': ['Def Awareness', 'Interceptions', 'Reactions', 'Composure', 'Standing Tackle', 'Sliding Tackle', 'Short Passing', 'Long Passing', 'Vision'],
        'preferred_foot': None,  # SW không cần ưu tiên chân cụ thể
        'weak_foot_bonus': 1.0  # Weak foot tốt là lợi thế cho sweeper (cần phát động tấn công)
    },
    
    # Hàng tiền đạo
    'ST': {
        'main_positions': ['ST', 'CF'], 
        'key_stats': ['PAC','Positioning', 'Finishing', 'Shot Power', 'Volleys', 'Heading Accuracy', 'Reactions', 'Strength', 'Composure','Curve'],
        'preferred_foot': None,  # ST có thể dùng cả 2 chân
        'weak_foot_bonus': 3.0  # Weak foot tốt cực kỳ quan trọng cho ST
    },
    'RW': {
        'main_positions': ['RW', 'RM', 'CAM'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Dribbling', 'Crossing', 'Finishing', 'Shot Power', 'Stamina','Curve'],
        'preferred_foot': 'Left',  # RW thường dùng chân trái để cắt vào trong
        'weak_foot_bonus': 2.5  # Weak foot tốt rất quan trọng
    },
    'LW': {
        'main_positions': ['LW', 'LM'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Dribbling', 'Crossing', 'Finishing', 'Shot Power', 'Stamina','Curve'],
        'preferred_foot': 'Right',  # LW thường dùng chân phải để cắt vào trong
        'weak_foot_bonus': 2.5  # Weak foot tốt rất quan trọng
    },
    'RWB': {
        'main_positions': ['RWB', 'RB'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Standing Tackle', 'Crossing', 'Dribbling', 'Stamina', 'Short Passing'],
        'preferred_foot': 'Right',  # RWB thường dùng chân phải
        'weak_foot_bonus': 1.5  # Weak foot tốt quan trọng
    },
    'LWB': {
        'main_positions': ['LWB', 'LB'], 
        'key_stats': ['Acceleration', 'Sprint Speed', 'Standing Tackle', 'Crossing', 'Dribbling', 'Stamina', 'Short Passing'],
        'preferred_foot': 'Left',  # LWB thường dùng chân trái
        'weak_foot_bonus': 1.5  # Weak foot tốt quan trọng
    },
}

# --- TỪ ĐIỂN KIẾN THỨC VỀ SƠ ĐỒ ---
FORMATION_SLOTS = {
    '4-2-2-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RDM', 'LDM', 'RAM', 'LAM', 'ST', 'ST'],
    '4-3-3': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'RCM', 'LCM', 'RW', 'ST', 'LW'],
    '3-5-2': ['GK', 'RCB', 'CB', 'LCB', 'RWB', 'LWB', 'CDM', 'LCM', 'RCM', 'ST', 'CAM'],
    '4-4-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RM', 'RCM', 'LCM', 'LM', 'ST', 'ST'],
    # Thêm các sơ đồ với vị trí chi tiết hơn
    '4-2-3-1': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RDM', 'LDM', 'RAM', 'CAM', 'LAM', 'ST'],
    '4-1-2-1-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'RCM', 'LCM', 'CAM', 'ST', 'ST'],
    '3-4-3': ['GK', 'RCB', 'CB', 'LCB', 'RWB', 'LWB', 'RCM', 'LCM', 'RW', 'ST', 'LW'],
    # Bạn có thể tự thêm các sơ đồ mới với SW ở đây
    # Ví dụ: 
    # '5-3-2': ['GK', 'RB', 'RCB', 'SW', 'LCB', 'LB', 'RDM', 'CDM', 'LDM', 'ST', 'ST']
}

print("Đã khởi tạo xong 'Bộ não' kiến thức chuyên gia với tên cột chính xác.")


def calculate_suitability_score(player_row, required_position_key):
    """
    Tính điểm phù hợp của một cầu thủ cho một vị trí yêu cầu.
    Bao gồm: OVR, chỉ số chi tiết, Preferred Foot, và Weak Foot.
    """
    requirements = POSITION_REQUIREMENTS[required_position_key]
    
    # Kiểm tra vị trí chính
    if player_row[POSITION_COLUMN] not in requirements['main_positions']:
        return 0
    
    # Lấy điểm OVR
    overall_score = player_row.get(OVERALL_COLUMN, 0)
    if pd.isna(overall_score):
        overall_score = 0
    
    # Tính trung bình các chỉ số chi tiết (key_stats)
    key_stats_total = 0
    num_key_stats = 0
    for stat_name in requirements['key_stats']:
        # Kiểm tra xem cột có tồn tại không để tránh lỗi
        if stat_name in player_row.index:
            stat_value = player_row[stat_name]
            if pd.notna(stat_value):
                try:
                    key_stats_total += float(stat_value)
                    num_key_stats += 1
                except (ValueError, TypeError):
                    pass
    
    if num_key_stats == 0:
        avg_key_stats_score = overall_score  # Nếu không có key_stats, dùng OVR
    else:
        avg_key_stats_score = key_stats_total / num_key_stats
    
    # Tính điểm cơ bản: 50% OVR + 50% chỉ số chi tiết
    base_score = (overall_score * 0.5) + (avg_key_stats_score * 0.5)
    
    # Tính điểm thưởng cho Preferred Foot
    preferred_foot_bonus = 0
    preferred_foot = requirements.get('preferred_foot')
    if preferred_foot is not None:
        player_preferred_foot = str(player_row.get('Preferred foot', '')).strip()
        if player_preferred_foot.lower() == preferred_foot.lower():
            preferred_foot_bonus = 5.0  # Thưởng 5 điểm nếu chân thuận phù hợp (ưu tiên cao)
        else:
            # Trừ điểm nếu chân thuận không phù hợp (nhưng vẫn cho phép nếu không có lựa chọn khác)
            preferred_foot_bonus = -2.0
    
    # Tính điểm thưởng cho Weak Foot
    weak_foot_bonus = 0
    weak_foot_bonus_multiplier = requirements.get('weak_foot_bonus', 0)
    if weak_foot_bonus_multiplier > 0:
        weak_foot_value = player_row.get('Weak foot', 0)
        try:
            weak_foot_value = float(weak_foot_value)
            if pd.notna(weak_foot_value):
                # Weak foot thường là 1-5, nhân với hệ số để tính điểm thưởng
                # Ví dụ: Weak foot = 5, multiplier = 2.0 -> bonus = 10 điểm
                weak_foot_bonus = weak_foot_value * weak_foot_bonus_multiplier
        except (ValueError, TypeError):
            pass
    
    # Điểm cuối cùng = điểm cơ bản + thưởng chân thuận + thưởng chân yếu
    final_score = base_score + preferred_foot_bonus + weak_foot_bonus
    
    return final_score


def build_team(dataframe, filter_name, formation_key, filter_type='team'):
    """
    Hàm chính để xây dựng đội hình.
    
    Parameters:
    - dataframe: DataFrame chứa dữ liệu cầu thủ
    - filter_name: Tên đội bóng hoặc quốc gia để lọc
    - formation_key: Tên sơ đồ (ví dụ: '4-3-3')
    - filter_type: 'team' để lọc theo team_color, 'nation' để lọc theo Nation
    """
    filter_type_name = "đội bóng" if filter_type == 'team' else "quốc gia"
    print(f"\nBắt đầu xây dựng đội hình cho {filter_type_name} '{filter_name}' với sơ đồ {formation_key}...")
    
    # Lọc cầu thủ theo team_color hoặc Nation
    if filter_type == 'nation':
        # Lọc theo quốc gia
        if 'Nation' not in dataframe.columns:
            print("Lỗi: Không tìm thấy cột 'Nation' trong dữ liệu.")
            return None
        potential_players = dataframe[dataframe['Nation'].str.contains(filter_name, case=False, na=False)].copy()
    else:
        # Lọc theo team_color (mặc định)
        if 'team_color' not in dataframe.columns:
            print("Lỗi: Không tìm thấy cột 'team_color' trong dữ liệu.")
            return None
        potential_players = dataframe[dataframe['team_color'].str.contains(filter_name, case=False, na=False)].copy()
    
    if potential_players.empty:
        print(f"Không tìm thấy cầu thủ nào cho {filter_type_name} '{filter_name}'.")
        return None
    
    print(f"Tìm thấy {len(potential_players)} ứng viên tiềm năng.")

    final_team = []
    used_player_ids = set()

    slots_to_fill = FORMATION_SLOTS[formation_key]
    for position_slot in slots_to_fill:
        best_player_for_slot = None
        max_score = -1
        
        for index, current_player in potential_players.iterrows():
            # CẬP NHẬT: Sử dụng ID_COLUMN đã định nghĩa
            if current_player[ID_COLUMN] in used_player_ids:
                continue
            
            score = calculate_suitability_score(current_player, position_slot)
            
            if score > max_score:
                max_score = score
                best_player_for_slot = current_player
        
        if best_player_for_slot is not None:
            final_team.append(best_player_for_slot)
            # CẬP NHẬT: Sử dụng ID_COLUMN đã định nghĩa
            used_player_ids.add(best_player_for_slot[ID_COLUMN])
            
    return final_team


# --- PHẦN THỰC THI CHÍNH ---
if __name__ == "__main__":
    # === BẠN CÓ THỂ THAY ĐỔI CÁC LỰA CHỌN NÀY ===
    FILTER_NAME = "England"           # Tên đội bóng hoặc quốc gia
    FORMATION_TO_USE = "4-3-3"        # Sơ đồ đội hình
    FILTER_TYPE = "nation"            # 'team' để lọc theo team_color, 'nation' để lọc theo Nation
    # ============================================

    dream_team = build_team(df, FILTER_NAME, FORMATION_TO_USE, filter_type=FILTER_TYPE)
    
    if dream_team and len(dream_team) == 11:
        filter_type_name = "đội bóng" if FILTER_TYPE == 'team' else "quốc gia"
        print(f"\n--- ĐỘI HÌNH TRONG MƠ CỦA {filter_type_name.upper()} '{FILTER_NAME}' ({FORMATION_TO_USE}) ---")
        
        slots = FORMATION_SLOTS[FORMATION_TO_USE]
        for i in range(11):
            player = dream_team[i]
            position = slots[i]
            # Lấy thông tin Preferred Foot và Weak Foot
            preferred_foot = str(player.get('Preferred foot', 'N/A')).strip()
            weak_foot = str(player.get('Weak foot', 'N/A')).strip()
            # CẬP NHẬT: Sử dụng các biến tên cột để in kết quả
            print(f"{position:<5}: {player[NAME_COLUMN]:<25} (OVR: {player[OVERALL_COLUMN]}, Pos: {player[POSITION_COLUMN]}, Foot: {preferred_foot}, WF: {weak_foot})")
    else:
        print("\nKhông thể xây dựng đội hình. Có thể do thiếu cầu thủ ở một số vị trí hoặc dữ liệu không đủ.")
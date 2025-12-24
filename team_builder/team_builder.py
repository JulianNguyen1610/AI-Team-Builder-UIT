import pandas as pd
import joblib
import os

# Import cấu hình từ file config.py
from config import FILE_PATH, MODEL_STORAGE_PATH, ID_COLUMN, POSITION_COLUMN, NAME_COLUMN, OVERALL_COLUMN, FEATURES_COLUMNS, POSITION_GROUPS, FORMATION_SLOTS
from config import POSITION_REQUIREMENTS_DETAILED
from genetic_optimizer import GeneticTeamBuilder
from utils import calculate_ml_suitability_score, get_player_group
from visualizer import visualize_team
# Import hàm tiền xử lý từ model_trainer để đảm bảo dữ liệu nhất quán
from model_trainer import preprocess_data

def load_models():
    """Tải tất cả các mô hình đã được huấn luyện từ thư mục 'models'."""
    print("Đang tải các mô hình AI chuyên gia...")
    models = {}
    for group_name in POSITION_GROUPS.keys():
        model_filename = os.path.join(MODEL_STORAGE_PATH, f"model_{group_name.lower()}.joblib")
        try:
            models[group_name] = joblib.load(model_filename)
            print(f"  -> Đã tải thành công mô hình cho: {group_name}")
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file mô hình tại '{model_filename}'.")
    return models

def get_player_group(player_position):
    """Hàm phụ để tìm ra nhóm vị trí của một cầu thủ."""
    for group, positions in POSITION_GROUPS.items():
        if player_position in positions:
            return group
    return None

def calculate_ml_suitability_score(player_row, required_position_key, models_dict):
    """Dự đoán điểm phù hợp và áp dụng điểm phạt vị trí."""
    # ... (phần code lấy model và dự đoán điểm giữ nguyên) ...
    player_pos_group = get_player_group(player_row[POSITION_COLUMN])
    required_pos_group = get_player_group(required_position_key)
    model = models_dict.get(required_pos_group)
    if model is None: return player_row[OVERALL_COLUMN]
    player_features = pd.DataFrame([player_row[FEATURES_COLUMNS]])
    predicted_score = model.predict(player_features)[0]

    # --- PHẦN NÂNG CẤP LOGIC ---
    player_actual_pos = player_row[POSITION_COLUMN]
    
    # Lấy danh sách các vị trí gốc phù hợp cho slot này từ từ điển chi tiết
    acceptable_native_positions = POSITION_REQUIREMENTS_DETAILED.get(required_position_key, {}).get('main_positions', [])
    
    # 1. Nếu cầu thủ không thể chơi ở vị trí yêu cầu -> Loại thẳng
    if not acceptable_native_positions: # Xử lý trường hợp key không có trong từ điển
        return 0

    # 2. Áp dụng điểm phạt dựa trên mức độ phù hợp vị trí
    if player_actual_pos in acceptable_native_positions:
        # Đây là trường hợp lý tưởng, cầu thủ chơi đúng sở trường
        # Thưởng một chút để ưu tiên
        predicted_score += 1.0 
    elif player_pos_group == required_pos_group:
        # Cùng nhóm vị trí (ví dụ xếp 1 CM đá CDM), có thể chấp nhận nhưng bị phạt nhẹ
        predicted_score *= 0.90 # Giảm 10% giá trị
    else:
        # Khác nhóm vị trí (xếp ST đá CDM) -> Phạt cực nặng
        predicted_score *= 0.60 # Giảm 40% giá trị, gần như không thể được chọn

    return predicted_score

# Đảm bảo bạn đã import đủ các thành phần cần thiết ở đầu file team_builder.py
# from config import FORMATION_SLOTS, POSITION_REQUIREMENTS_DETAILED, POSITION_GROUPS, ID_COLUMN, POSITION_COLUMN
# from genetic_optimizer import GeneticTeamBuilder

def build_team(dataframe, filter_name, formation_key, models_dict, filter_type='team', use_genetic_algo=True):
    """
    Hàm chính để xây dựng đội hình sử dụng các mô hình ML.
    """
    filter_type_name = "đội bóng" if filter_type == 'team' else "quốc gia"
    algo_name = "Genetic Algorithm (Tiến hóa)" if use_genetic_algo else "Greedy Algorithm (Tham lam)"
    
    print(f"\nBắt đầu xây dựng đội hình cho {filter_type_name} '{filter_name}'")
    print(f"Sơ đồ: {formation_key} | Thuật toán: {algo_name}")
    
    # 1. Lọc cầu thủ theo Team Color hoặc Quốc gia
    if filter_type == 'nation':
        if 'Nation' not in dataframe.columns:
             print("Lỗi: Dữ liệu thiếu cột 'Nation'.")
             return None
        potential_players = dataframe[dataframe['Nation'].str.contains(filter_name, case=False, na=False)].copy()
    else:
        if 'team_color' not in dataframe.columns:
             print("Lỗi: Dữ liệu thiếu cột 'team_color'.")
             return None
        potential_players = dataframe[dataframe['team_color'].str.contains(filter_name, case=False, na=False)].copy()
    
    if potential_players.empty:
        print(f"Không tìm thấy cầu thủ nào cho {filter_type_name} '{filter_name}'.")
        return None
    
    print(f"Tìm thấy {len(potential_players)} ứng viên tiềm năng.")
    
    slots_to_fill = FORMATION_SLOTS[formation_key]
    
    # --- NHÁNH 1: SỬ DỤNG GENETIC ALGORITHM (Ưu tiên) ---
    if use_genetic_algo:
        # Khởi tạo và chạy thuật toán di truyền
        optimizer = GeneticTeamBuilder(potential_players, slots_to_fill, models_dict)
        final_team = optimizer.run()
        return final_team

    # --- NHÁNH 2: SỬ DỤNG GREEDY ALGORITHM (Dự phòng) ---
    # (Chỉ chạy xuống đây nếu use_genetic_algo = False)
    
    final_team = []
    used_player_ids = set()

    for position_slot in slots_to_fill:
        best_player_for_slot = None
        max_score = -100 # Đặt thấp để tránh lỗi logic

        # --- A. Lọc ứng viên (Candidate Selection) ---
        # 1. Ưu tiên 1: Cầu thủ có vị trí sở trường (Main Position) khớp với slot
        acceptable_native_positions = POSITION_REQUIREMENTS_DETAILED.get(position_slot, {}).get('main_positions', [])
        candidates = potential_players[potential_players[POSITION_COLUMN].isin(acceptable_native_positions)]
        
        # 2. Ưu tiên 2: Nếu không có ai đúng sở trường, mở rộng ra cùng nhóm vị trí (VD: Cần CDM mà hết CDM thì tìm CM)
        if candidates.empty:
            required_pos_group = get_player_group(position_slot)
            if required_pos_group:
                group_positions = POSITION_GROUPS.get(required_pos_group, [])
                candidates = potential_players[potential_players[POSITION_COLUMN].isin(group_positions)]
        
        # 3. Ưu tiên 3: Nếu vẫn rỗng (trường hợp hiếm), xét toàn bộ danh sách để tìm "kẻ đóng thế" (ví dụ CB đá CDM)
        if candidates.empty:
            candidates = potential_players

        # --- B. Chấm điểm và Chọn lựa ---
        for index, current_player in candidates.iterrows():
            if current_player[ID_COLUMN] in used_player_ids:
                continue
            
            # Gọi hàm tính điểm ML (hàm này đã bao gồm logic phạt vị trí và thưởng chơi chân)
            score = calculate_ml_suitability_score(current_player, position_slot, models_dict)
            
            if score > max_score:
                max_score = score
                best_player_for_slot = current_player
        
        # --- C. Thêm vào đội hình ---
        if best_player_for_slot is not None:
            final_team.append(best_player_for_slot)
            used_player_ids.add(best_player_for_slot[ID_COLUMN])
        else:
            print(f"Cảnh báo: Không tìm thấy cầu thủ phù hợp cho vị trí {position_slot}")
            
    return final_team

# --- PHẦN THỰC THI CHÍNH ---
if __name__ == "__main__":
    # Nạp dữ liệu và tiền xử lý
    try:
        main_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{FILE_PATH}'.")
        exit()
        
    # Tiền xử lý dữ liệu (để có các cột số liệu chuẩn)
    processed_df = preprocess_data(main_df)

    # Tải các mô hình đã huấn luyện
    trained_models = load_models()

    if not trained_models:
        print("\nLỗi: Không có mô hình nào được tải. Vui lòng chạy file 'model_trainer.py' trước.")
    else:
        # === BẠN CÓ THỂ THAY ĐỔI CÁC LỰA CHỌN NÀY ĐỂ TƯƠNG TÁC ===
        FILTER_NAME = "Al Ittihad"
        FORMATION_TO_USE = "4-3-3"
        FILTER_TYPE = "team"
        USE_GENETIC = True 
        # ===============

        dream_team = build_team(processed_df, FILTER_NAME, FORMATION_TO_USE, trained_models, 
                                filter_type=FILTER_TYPE, use_genetic_algo=USE_GENETIC)
        
        if dream_team and len(dream_team) == 11:
            print(f"\n--- ĐỘI HÌNH TỐI ƯU HÓA (GENETIC AI) ---")
            slots = FORMATION_SLOTS[FORMATION_TO_USE]
            for i in range(11):
                player = dream_team[i]
                position = slots[i]
                # In thêm Archetype để kiểm tra độ hòa hợp
                archetype = player.get('Archetype', 'N/A')
                print(f"{position:<5}: {player[NAME_COLUMN]:<25} (OVR: {player[OVERALL_COLUMN]}, Role: {archetype:<20})")
            print("\nĐang vẽ sơ đồ chiến thuật...")
            visualize_team(dream_team, FORMATION_TO_USE, FILTER_NAME)
        else:
            print("\nKhông thể xây dựng đội hình đầy đủ 11 người. Có thể do thiếu cầu thủ phù hợp.")
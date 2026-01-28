# config.py

# --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
import os
# Đường dẫn tương đối đến file data.csv (file gốc và file sau khi phân tích)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'male_players_final.csv')
MODEL_STORAGE_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models')
# --- CẤU HÌNH TÊN CỘT ---
ID_COLUMN = 'ID'
POSITION_COLUMN = 'Position'
NAME_COLUMN = 'Name'
OVERALL_COLUMN = 'OVR'

# --- CẤU HÌNH CHO HUẤN LUYỆN ML ---
FEATURES_COLUMNS = [
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Crossing', 'Finishing', 
    'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 
    'FK Accuracy', 'Free Kick Accuracy',  # <--- THÊM DÒNG NÀY (Thêm cả 2 để an toàn)
    'Long Passing', 'Ball Control', 'Acceleration', 'Sprint Speed', 
    'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 
    'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 
    'Vision', 'Penalties', 'Composure', 'Def Awareness', 'Standing Tackle', 
    'Sliding Tackle', 'Weak foot', 'Preferred foot_numeric'
]

# --- TỪ ĐIỂN KIẾN THỨC VỀ VỊ TRÍ ---
POSITION_GROUPS = {
    'Defender': ['CB', 'RB', 'LB', 'RWB', 'LWB', 'SW', 'RCB', 'LCB'],
    'Midfielder': ['CDM', 'CM', 'CAM', 'RM', 'LM', 'RDM', 'LDM', 'RCM', 'LCM'],
    'Attacker': ['ST', 'CF', 'RW', 'LW'],
    'Goalkeeper': ['GK']
}

POSITION_REQUIREMENTS_DETAILED = {
    'GK': {'main_positions': ['GK']},
    'CB': {'main_positions': ['CB', 'RCB', 'LCB', 'SW']},
    'RCB': {'main_positions': ['CB', 'RCB', 'LCB']},
    'LCB': {'main_positions': ['CB', 'RCB', 'LCB']},
    'SW': {'main_positions': ['SW', 'CB']},
    
    'RB': {'main_positions': ['RB', 'RWB']},
    'LB': {'main_positions': ['LB', 'LWB']},
    'RWB': {'main_positions': ['RWB', 'RB']},
    'LWB': {'main_positions': ['LWB', 'LB']},
    
    'CDM': {'main_positions': ['CDM', 'LDM', 'RDM']},
    'RDM': {'main_positions': ['CDM', 'LDM', 'RDM']},
    'LDM': {'main_positions': ['CDM', 'LDM', 'RDM']},
    
    'CM': {'main_positions': ['CM', 'LCM', 'RCM', 'CAM']}, 
    'RCM': {'main_positions': ['CM', 'LCM', 'RCM']},
    'LCM': {'main_positions': ['CM', 'LCM', 'RCM']},
    
    'CAM': {'main_positions': ['CAM', 'RAM', 'LAM', 'CM']}, 
    'RAM': {'main_positions': ['CAM', 'RAM', 'LAM', 'RW', 'RM']},
    'LAM': {'main_positions': ['CAM', 'RAM', 'LAM', 'LW', 'LM']},
    
    'RM': {'main_positions': ['RM', 'RW']},
    'LM': {'main_positions': ['LM', 'LW']},
    'RW': {'main_positions': ['RW', 'RM']},
    'LW': {'main_positions': ['LW', 'LM']},
    
    'ST': {'main_positions': ['ST', 'CF', 'LS', 'RS']},
    'CF': {'main_positions': ['CF', 'ST']},
    'LS': {'main_positions': ['ST', 'CF']},
    'RS': {'main_positions': ['ST', 'CF']}
}

# --- SƠ ĐỒ ĐỘI HÌNH ---
FORMATION_SLOTS = {
    '4-2-2-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RDM', 'LDM', 'RAM', 'LAM', 'ST', 'ST'],
    '4-3-3': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'RCM', 'LCM', 'RW', 'ST', 'LW'],
    '3-5-2': ['GK', 'RCB', 'CB', 'LCB', 'RWB', 'LWB', 'CDM', 'LCM', 'RCM', 'ST', 'ST'],
    '4-4-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RM', 'RCM', 'LCM', 'LM', 'ST', 'ST'],
    '4-2-3-1': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'CDM', 'RAM', 'CAM', 'LAM', 'ST'],
    '4-1-3-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'ST', 'RAM', 'CAM', 'LAM', 'ST'],
    '4-2-2-1-1': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'CDM', 'RM', 'CAM', 'LM', 'ST'],
    '4-1-2-1-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'CDM', 'RCM', 'LCM', 'CAM', 'ST', 'ST'],
    '3-4-3': ['GK', 'CB', 'RCB', 'LCB', 'LM', 'RCM', 'LCM', 'RM', 'RW', 'LW', 'ST'],
    '5-3-2': ['GK', 'CB', 'RCB', 'LCB', 'LWB', 'RWB', 'CM', 'RCM', 'LCM', 'ST', 'ST'],
    '5-4-1': ['GK', 'CB', 'RCB', 'LCB', 'LWB', 'RWB', 'RCM', 'RM', 'LM', 'LCM', 'ST'],
    '3-4-1-2': ['GK', 'CB', 'RCB', 'LCB', 'LWB', 'RWB', 'RCM', 'CAM', 'ST', 'LCM', 'ST'],
    '4-2-4': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RDM', 'LDM', 'RW', 'LW', 'ST', 'ST'],
    '4-2-1-3': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RDM', 'LDM', 'RW', 'LW', 'CAM', 'ST'],
    '4-1-4-1': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RCM', 'LCM', 'RM', 'LM', 'CDM', 'ST'],
    '4-4-1-1': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RCM', 'LCM', 'RM', 'LM', 'CAM', 'ST'],
    '4-3-1-2': ['GK', 'RB', 'RCB', 'LCB', 'LB', 'RCM', 'LCM', 'CM', 'CAM', 'ST', 'ST'],
    '5-2-1-2': ['GK', 'CB', 'RCB', 'LCB', 'LWB', 'RWB', 'RCM', 'CAM', 'LCM', 'ST', 'ST'],
    '5-2-3': ['GK', 'CB', 'RCB', 'LCB', 'LWB', 'RWB', 'RCM', 'RW', 'LW', 'LCM', 'ST']
    
    
}

# --- CẤU HÌNH HÓA HỌC CHIẾN THUẬT (CHEMISTRY) ---
TACTICAL_BONUSES = {
    ('CB', 'CB'): [('Physical Stopper', 'Ball-Playing Defender'), ('Physical Stopper', 'Standard Defender')],
    ('CDM', 'CDM'): [('Box-to-Box', 'Playmaker'), ('Anchor Man', 'Box-to-Box'), ('Anchor Man', 'Playmaker')],
    ('ST', 'ST'): [('Target Man', 'Speedster'), ('Target Man', 'Clinical Finisher'), ('Clinical Finisher', 'Speedster')]
}
SYNERGY_BONUS_SCORE = 3.0

# --- CẤU HÌNH CHO PHÂN CỤM (CLUSTERING) ---
ARCHETYPE_FEATURES = {
    'Midfielder': ['Short Passing', 'Long Passing', 'Vision', 'Dribbling', 'Ball Control', 'Standing Tackle', 'Interceptions', 'Stamina', 'Aggression', 'PAC', 'SHO'],
    'Defender': ['Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Heading Accuracy', 'Jumping', 'Strength', 'Aggression', 'Interceptions', 'PAC', 'Short Passing'],
    'Attacker': ['Finishing', 'Shot Power', 'Long Shots', 'Positioning', 'Volleys', 'Heading Accuracy', 'PAC', 'DRI', 'Strength', 'Composure', 'Vision'],
    'Goalkeeper': ['GK Diving', 'GK Handling', 'GK Kicking', 'GK Reflexes', 'GK Positioning']
}

# --- CẤU HÌNH CHIẾN THUẬT (TACTICAL PRESETS - MỚI) ---
TACTICAL_PROFILES = {
    'Balanced (Cân bằng)': {},
    'Tiki-Taka (Kiểm soát)': {'Short Passing': 0.4, 'Vision': 0.3, 'Ball Control': 0.3, 'Composure': 0.2},
    'Counter Attack (Phản công)': {'Sprint Speed': 0.4, 'Acceleration': 0.4, 'Long Passing': 0.3, 'Positioning': 0.2},
    'Wing Play (Tạt cánh)': {'Crossing': 0.5, 'Curve': 0.3, 'Heading Accuracy': 0.4, 'Jumping': 0.3},
    'High Press (Gegenpressing)': {'Stamina': 0.5, 'Aggression': 0.4, 'Interceptions': 0.3, 'Reactions': 0.3}
}
STRICT_RIGHT_SIDED_POSITIONS = ['RB', 'RWB', 'RM']
STRICT_LEFT_SIDED_POSITIONS = ['LB', 'LWB', 'LM']
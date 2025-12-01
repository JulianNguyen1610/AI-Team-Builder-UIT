# config.py

# --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
FILE_PATH = r'D:\ai002\male_players_with_archetypes.csv' # (Hoặc đường dẫn của bạn)
MODEL_STORAGE_PATH = 'models/'

# --- CẤU HÌNH TÊN CỘT ---
ID_COLUMN = 'ID'
POSITION_COLUMN = 'Position'
NAME_COLUMN = 'Name'
OVERALL_COLUMN = 'OVR'

# --- CẤU HÌNH CHO HUẤN LUYỆN ML ---
FEATURES_COLUMNS = [
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Crossing', 'Finishing', 
    'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 
    'Free Kick Accuracy', 'Long Passing', 'Ball Control', 'Acceleration', 'Sprint Speed', 
    'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 
    'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 
    'Vision', 'Penalties', 'Composure', 'Def Awareness', 'Standing Tackle', 
    'Sliding Tackle', 'Weak foot', 'Preferred foot_numeric'
]

# --- TỪ ĐIỂN NHÓM VỊ TRÍ CHI TIẾT (7 NHÓM - Dùng cho Model Training & Archetype) ---
POSITION_GROUPS = {
    'Goalkeeper': ['GK'],
    'CenterBack': ['CB', 'RCB', 'LCB', 'SW'],
    'FullBack':   ['RB', 'LB', 'RWB', 'LWB'],
    'DefensiveMidfielder': ['CDM', 'RDM', 'LDM'],
    'CentralMidfielder':   ['CM', 'RCM', 'LCM', 'CAM', 'RAM', 'LAM'], 
    'Winger':     ['RW', 'LW', 'RM', 'LM'],
    'Striker':    ['ST', 'CF', 'LS', 'RS']
}

# --- CẤU HÌNH CHỈ SỐ CHO PHÂN CỤM ---
ARCHETYPE_FEATURES = {
    'Goalkeeper': [
        'GK Diving', 'GK Handling', 'GK Kicking', 'GK Reflexes', 'GK Positioning'
    ],
    'CenterBack': [
        'Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Heading Accuracy', 
        'Strength', 'Jumping', 'Interceptions', 'Short Passing', 'Long Passing', 'Aggression'
    ],
    'FullBack': [
        'Acceleration', 'Sprint Speed', 'Crossing', 'Stamina', 'Dribbling', 
        'Standing Tackle', 'Interceptions', 'Short Passing', 'Agility'
    ],
    'DefensiveMidfielder': [
        'Interceptions', 'Standing Tackle', 'Strength', 'Stamina', 'Short Passing', 
        'Long Passing', 'Aggression', 'Def Awareness', 'Ball Control'
    ],
    'CentralMidfielder': [ 
        'Vision', 'Short Passing', 'Long Passing', 'Dribbling', 'Ball Control', 
        'Agility', 'Long Shots', 'Finishing', 'Positioning', 'Reactions'
    ],
    'Winger': [
        'Acceleration', 'Sprint Speed', 'Dribbling', 'Agility', 'Crossing', 
        'Balance', 'Finishing', 'Shot Power', 'Skill moves'
    ],
    'Striker': [
        'Finishing', 'Positioning', 'Shot Power', 'Heading Accuracy', 'Volleys', 
        'Strength', 'Acceleration', 'Sprint Speed', 'Jumping', 'Composure',
        'Short Passing', 'Dribbling', 'Ball Control'
    ]
}

# --- YÊU CẦU VỊ TRÍ CHI TIẾT (Dùng cho logic tính điểm trong utils.py) ---
# ĐÂY LÀ PHẦN BỊ THIẾU GÂY LỖI
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

# --- HÓA HỌC CHIẾN THUẬT ---
TACTICAL_BONUSES = {
    # Cặp trung vệ: Dập + Thòng
    ('CB', 'CB'): [('Stopper', 'Ball-Playing CB'), ('Elite Defender', 'Ball-Playing CB')],
    # Cặp tiền vệ trụ: Máy quét + Nhạc trưởng
    ('CDM', 'CM'): [('Destroyer', 'Advanced Playmaker'), ('Anchor Man', 'Box-to-Box')],
    # Cặp cánh: Hậu vệ bó trong + Cánh bám biên
    ('LB', 'LW'): [('Inverted Fullback', 'Traditional Winger'), ('Attacking Wingback', 'Inside Forward')],
    # Cặp tiền đạo: Làm tường + Chạy nhanh
    ('ST', 'ST'): [('Target Man', 'Poacher'), ('False 9', 'Inside Forward'), ('Target Man', 'Complete Forward')]
}
SYNERGY_BONUS_SCORE = 5
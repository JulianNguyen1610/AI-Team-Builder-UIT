import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import FILE_PATH, POSITION_COLUMN, POSITION_GROUPS, ARCHETYPE_FEATURES

def get_trait_values(stats, traits_dict):
    """Tính điểm trung bình cho từng nhóm đặc tính."""
    results = {}
    for trait_name, columns in traits_dict.items():
        valid_cols = [c for c in columns if c in stats.index]
        if valid_cols:
            results[trait_name] = stats[valid_cols].mean()
        else:
            results[trait_name] = 0
    return results

def auto_name_archetype(group_name, centroid_series):
    stats = centroid_series
    
    # --- 1. CENTER BACK ---
    if group_name == 'CenterBack':
        traits = {
            'BallPlaying': ['Short Passing', 'Long Passing', 'Ball Control'],
            'Physical':    ['Strength', 'Aggression', 'Jumping'],
            'Defensive':   ['Standing Tackle', 'Def Awareness', 'Interceptions'],
            'Pace':        ['Acceleration', 'Sprint Speed']
        }
        vals = get_trait_values(stats, traits)
        
        # Logic phân loại sâu
        if vals['BallPlaying'] > 60 and vals['BallPlaying'] > vals['Physical'] - 15:
            return "Ball-Playing CB"
        
        if vals['Pace'] > 70 and vals['Defensive'] > 75:
            return "Fast Defender" # Trung vệ thòng tốc độ
            
        if vals['Physical'] > vals['Defensive'] + 2:
            return "Stopper" # Dập
        
        if vals['Defensive'] > vals['Physical']:
            return "Cover Defender" # Thòng/Bọc lót
            
        return "No-Nonsense CB"

    # --- 2. FULL BACK ---
    elif group_name == 'FullBack':
        traits = {
            'Attack': ['Crossing', 'Dribbling', 'Curve'],
            'Inverted': ['Short Passing', 'Vision', 'Ball Control'],
            'Defense': ['Standing Tackle', 'Interceptions', 'Def Awareness'],
            'Speed':   ['Acceleration', 'Sprint Speed']
        }
        vals = get_trait_values(stats, traits)
        
        if vals['Inverted'] > 68 and vals['Inverted'] > vals['Attack']:
            return "Inverted Fullback"
        
        if vals['Attack'] > 70:
            if vals['Speed'] > 85: return "Complete Wingback"
            return "Attacking Fullback"
            
        if vals['Defense'] > 72:
            return "Defensive Fullback"
            
        return "Balanced Fullback"

    # --- 3. DEFENSIVE MIDFIELDER ---
    elif group_name == 'DefensiveMidfielder':
        traits = {
            'Playmaking': ['Short Passing', 'Long Passing', 'Vision'],
            'Physical':   ['Strength', 'Aggression', 'Standing Tackle'],
            'Defense':    ['Interceptions', 'Def Awareness', 'Positioning']
        }
        vals = get_trait_values(stats, traits)
        
        if vals['Playmaking'] > 75: return "Deep-Lying Playmaker"
        if vals['Physical'] > vals['Defense'] + 3: return "Destroyer" # Chuyên húc ủi
        if vals['Defense'] > 75: return "Anchor Man" # Chuyên cắt bóng
        return "Holding Midfielder"

    # --- 4. CENTRAL MIDFIELDER ---
    elif group_name == 'CentralMidfielder':
        traits = {
            'Create': ['Vision', 'Short Passing', 'Long Passing'],
            'Score':  ['Finishing', 'Long Shots', 'Positioning'],
            'Dribble':['Dribbling', 'Agility', 'Ball Control'],
            'Work':   ['Stamina', 'Reactions', 'Interceptions']
        }
        vals = get_trait_values(stats, traits)
        
        if vals['Create'] > 78: return "Advanced Playmaker"
        if vals['Score'] > 70 and vals['Score'] > vals['Create'] - 5: return "Shadow Striker"
        if vals['Dribble'] > 78: return "Technical Midfielder"
        if vals['Work'] > 70: return "Box-to-Box"
        return "Central Midfielder"

    # --- 5. WINGER ---
    elif group_name == 'Winger':
        traits = {
            'Score': ['Finishing', 'Shot Power', 'Positioning'],
            'Cross': ['Crossing', 'Curve', 'Long Passing'],
            'Tech':  ['Dribbling', 'Agility', 'Ball Control'],
            'Speed': ['Acceleration', 'Sprint Speed']
        }
        vals = get_trait_values(stats, traits)
        
        # Hầu hết winger đều nhanh, nên phải xét chỉ số phụ
        if vals['Score'] > 70 and vals['Score'] > vals['Cross']: return "Inside Forward"
        if vals['Cross'] > 72: return "Traditional Winger"
        if vals['Tech'] > 80: return "Wide Playmaker"
        if vals['Speed'] > 85: return "Speedster"
        return "Standard Winger"

    # --- 6. STRIKER ---
    elif group_name == 'Striker':
        traits = {
            'Target':  ['Heading Accuracy', 'Strength', 'Jumping'],
            'Poach':   ['Finishing', 'Positioning', 'Reactions', 'Composure'], # Thêm Composure
            'Linkup':  ['Short Passing', 'Vision', 'Dribbling'],
            'Speed':   ['Acceleration', 'Sprint Speed', 'Agility'] # Thêm Agility
        }
        vals = get_trait_values(stats, traits)
        
        # --- LOGIC MỚI: ƯU TIÊN TỐC ĐỘ VÀ DỨT ĐIỂM TRƯỚC ---
        
        # 1. Nếu cực nhanh -> Speedster
        if vals['Speed'] > 85: return "Speedster"
        
        # 2. Nếu dứt điểm cực bén -> Poacher
        if vals['Poach'] > 80: return "Poacher"
        
        # 3. Nếu toàn diện (Dứt điểm tốt, Tốc độ ổn, Chuyền ổn) -> Complete Forward
        if vals['Poach'] > 75 and vals['Speed'] > 75 and vals['Linkup'] > 70:
            return "Complete Forward"
            
        # 4. Nếu chuyền tốt -> False 9
        if vals['Linkup'] > 72 and vals['Linkup'] > vals['Target']: return "False 9"
        
        # 5. Cuối cùng mới xét Target Man (chỉ dành cho cầu thủ chậm mà khỏe)
        if vals['Target'] > 75: return "Target Man"
        
        return "Advanced Forward"

    # --- 7. GK ---
    elif group_name == 'Goalkeeper':
        kicking = stats.get('GK Kicking', 50)
        reflexes = stats.get('GK Reflexes', 50)
        
        if kicking > 70 and kicking > reflexes - 5: return "Sweeper Keeper"
        if reflexes > 75: return "Shot Stopper"
        return "Standard GK"

    return "Generic"

def analyze_and_cluster():
    print("--- BẮT ĐẦU PHÂN TÍCH HÌNH MẪU CẦU THỦ (V6 - MULTI DIMENSIONAL) ---")
    
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Đã tải dữ liệu: {len(df)} cầu thủ.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    df[POSITION_COLUMN] = df[POSITION_COLUMN].astype(str).apply(lambda x: x.split(',')[0].strip())
    df['Archetype'] = 'N/A'
    
    # Số lượng cụm
    N_CLUSTERS = 5

    for group_name, features in ARCHETYPE_FEATURES.items():
        print(f"\n>> Đang phân tích nhóm: {group_name}...")
        
        if group_name not in POSITION_GROUPS: continue
        positions = POSITION_GROUPS[group_name]
        
        group_df = df[df[POSITION_COLUMN].isin(positions)].copy()
        
        valid_features = [f for f in features if f in group_df.columns]
        if group_df.empty or len(valid_features) < 3: 
            print("   Không đủ dữ liệu.")
            continue

        X = group_df[valid_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        group_df['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        # Tự động đặt tên
        X['cluster_id'] = group_df['cluster_id']
        centroids = X.groupby('cluster_id').mean()
        
        cluster_names = {}
        
        # Sắp xếp các cluster theo mức độ 'xịn' (ví dụ tổng điểm trung bình) để dễ phân biệt Tier
        centroids['mean_score'] = centroids.mean(axis=1)
        sorted_indices = centroids.sort_values('mean_score', ascending=False).index

        for c_id in sorted_indices:
            name = auto_name_archetype(group_name, centroids.loc[c_id])
            
            # Xử lý trùng tên bằng cách thêm Tier (Elite, Solid...)
            if name in cluster_names.values():
                # Nếu tên đã tồn tại, kiểm tra xem cái hiện tại xịn hơn hay dở hơn
                # Vì ta đã sort từ cao xuống thấp, nên cái đầu tiên gặp là Elite
                # Cái sau sẽ đổi tên khác
                original_name = name
                count = 1
                # Đổi tên các cái bị trùng phía sau thành 'Solid', 'Backup'
                suffixes = ["", " (Solid)", " (Backup)", " (Young)", " (Low Tier)"]
                
                # Logic đếm số lần xuất hiện của tên này
                existing_count = list(cluster_names.values()).count(name) + list(cluster_names.values()).count(name + " (Solid)") # Simplification
                
                # Cách đơn giản: Thêm hậu tố dựa trên thứ tự sort
                # (Cái đầu tiên là xịn nhất -> giữ nguyên tên)
                # (Cái thứ 2 trùng -> thêm Solid)
                count = 0
                for existing_name in cluster_names.values():
                    if existing_name.startswith(name):
                        count += 1
                
                if count > 0 and count < len(suffixes):
                    name = f"{name}{suffixes[count]}"
                elif count >= len(suffixes):
                    name = f"{name} ({count})"

            cluster_names[c_id] = name
            print(f"   Cluster {c_id}: {name:<25} (Avg Stat: {centroids.loc[c_id, 'mean_score']:.1f})")

        for idx, row in group_df.iterrows():
            c_id = row['cluster_id']
            df.at[idx, 'Archetype'] = cluster_names[c_id]

    OUTPUT_FILE = 'male_players_with_archetypes.csv'
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n--- HOÀN TẤT ---")
    print(f"File mới đã được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_and_cluster()
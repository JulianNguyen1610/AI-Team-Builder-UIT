import random
import pandas as pd
import copy
import numpy as np
from config import ID_COLUMN, POSITION_COLUMN, NAME_COLUMN, OVERALL_COLUMN, TACTICAL_BONUSES, SYNERGY_BONUS_SCORE, FEATURES_COLUMNS
from utils import calculate_ml_suitability_score, get_player_group

class GeneticTeamBuilder:
    def __init__(self, dataframe, formation_slots, models_dict):
        self.df = dataframe
        self.formation_slots = formation_slots
        self.models_dict = models_dict
        
        # --- CẤU HÌNH ---
        self.POPULATION_SIZE = 100      
        self.GENERATIONS = 50           
        self.CANDIDATES_PER_SLOT = 60   
        self.MUTATION_RATE = 0.25       

        self.candidates_pool = [] 
        
    def precompute_candidates(self):
        print(f"   -> Đang sàng lọc {self.CANDIDATES_PER_SLOT} ứng viên tốt nhất mỗi vị trí...")
        self.candidates_pool = []
        
        # --- TÍNH OVR TRUNG BÌNH CỦA ĐỘI ---
        # Tính ngưỡng sàn chất lượng
        avg_team_ovr = self.df[OVERALL_COLUMN].mean()
        if pd.isna(avg_team_ovr): avg_team_ovr = 75 # Giá trị mặc định nếu lỗi
        min_ovr_threshold = avg_team_ovr - 5
        
        for slot_idx, position_req in enumerate(self.formation_slots):
            req_group = get_player_group(position_req)
            
            # Lấy top 600 cầu thủ để lọc sơ bộ
            candidates = self.df.sort_values(by=OVERALL_COLUMN, ascending=False).head(600).copy()
            
            # --- ÁP DỤNG SÀN CHẤT LƯỢNG ---
            # Chỉ giữ lại cầu thủ có OVR >= ngưỡng sàn
            high_quality = candidates[candidates[OVERALL_COLUMN] >= min_ovr_threshold]
            
            # Chỉ áp dụng lọc nếu số lượng cầu thủ còn lại đủ nhiều
            if len(high_quality) >= self.CANDIDATES_PER_SLOT:
                candidates = high_quality
            
            # --- LOGIC HẬU VỆ ĐÁ TIỀN VỆ ---
            if req_group == 'Midfielder':
                # Lấy thêm hậu vệ giỏi
                ball_playing_cbs = self.df[
                    (self.df[POSITION_COLUMN].isin(['CB', 'RCB', 'LCB'])) & 
                    (self.df['Short Passing'] > 78)
                ].copy()
                inverted_fullbacks = self.df[
                    (self.df[POSITION_COLUMN].isin(['RB', 'LB', 'RWB', 'LWB'])) & 
                    (self.df['Vision'] > 74) & 
                    ( (self.df['Long Passing'] > 74) | (self.df['Short Passing'] > 78) )
                ].copy()
                
                # Cũng áp dụng sàn chất lượng cho nhóm bổ sung này
                ball_playing_cbs = ball_playing_cbs[ball_playing_cbs[OVERALL_COLUMN] >= min_ovr_threshold]
                inverted_fullbacks = inverted_fullbacks[inverted_fullbacks[OVERALL_COLUMN] >= min_ovr_threshold]
                
                candidates = pd.concat([candidates, ball_playing_cbs, inverted_fullbacks]).drop_duplicates(subset=[ID_COLUMN])
            
            # --- BATCH PREDICTION (TÍNH ĐIỂM HÀNG LOẠT) ---
            model = self.models_dict.get(req_group)
            if model:
                # Tạo DataFrame features để tránh warning
                X_candidates = candidates[FEATURES_COLUMNS]
                base_scores = model.predict(X_candidates)
                candidates['base_ml_score'] = base_scores
            else:
                candidates['base_ml_score'] = candidates[OVERALL_COLUMN]

            # Tính điểm cuối cùng (đã bao gồm các logic phạt/thưởng vị trí)
            candidates['ml_score'] = candidates.apply(
                lambda row: calculate_ml_suitability_score(row, position_req, self.models_dict), axis=1
            )
            
            # Lấy top ứng viên tốt nhất
            top_candidates = candidates.sort_values(by='ml_score', ascending=False).head(self.CANDIDATES_PER_SLOT)
            
            # Chuyển thành dict và lưu vào pool
            pool_for_slot = top_candidates.to_dict('records')
            
            # --- KIỂM TRA AN TOÀN ---
            if not pool_for_slot:
                print(f"CẢNH BÁO: Không tìm thấy ứng viên nào cho vị trí {position_req}!")
                # Fallback: Lấy đại top 10 cầu thủ OVR cao nhất bất kể vị trí để tránh crash
                fallback = self.df.sort_values(by=OVERALL_COLUMN, ascending=False).head(10).to_dict('records')
                # Gán điểm ảo để code chạy tiếp
                for p in fallback: p['ml_score'] = 0 
                pool_for_slot = fallback

            # --- DÒNG QUAN TRỌNG NHẤT: APPEND VÀO POOL ---
            self.candidates_pool.append(pool_for_slot)
            
    def create_individual(self, smart_seed=False):
        team = []
        used_ids = set()
        
        for slot_idx in range(len(self.formation_slots)):
            pool = self.candidates_pool[slot_idx]
            
            if smart_seed:
                # Lấy ngẫu nhiên trong Top 5 người giỏi nhất
                top_k = min(len(pool), 5)
                candidate = random.choice(pool[:top_k])
                
                retries = 0
                while candidate[ID_COLUMN] in used_ids and retries < 10:
                    candidate = random.choice(pool[:top_k])
                    retries += 1
            else:
                # Random có trọng số
                pool_size = len(pool)
                weights = [pool_size - i for i in range(pool_size)]
                candidate = random.choices(pool, weights=weights, k=1)[0]
                retries = 0
                while candidate[ID_COLUMN] in used_ids and retries < 15:
                    candidate = random.choices(pool, weights=weights, k=1)[0]
                    retries += 1
            
            team.append(candidate)
            used_ids.add(candidate[ID_COLUMN])
            
        return team

    def calculate_fitness(self, team):
        total_score = 0
        def clean_arch(name):
            if not isinstance(name, str): return "Generic"
            return name.split(' (')[0]

        archetypes = [clean_arch(p.get('Archetype', 'Generic')) for p in team]
        
        # 1. Điểm kỹ năng
        for player in team:
            total_score += player['ml_score']
            
        # 2. Điểm Chemistry
        import itertools
        for (pos1, pos2), valid_pairs in TACTICAL_BONUSES.items():
            indices_1 = [i for i, pos in enumerate(self.formation_slots) if pos == pos1]
            indices_2 = [i for i, pos in enumerate(self.formation_slots) if pos == pos2]
            
            if pos1 == pos2: 
                for i1, i2 in itertools.combinations(indices_1, 2):
                    arch1, arch2 = archetypes[i1], archetypes[i2]
                    if (arch1, arch2) in valid_pairs or (arch2, arch1) in valid_pairs:
                        total_score += SYNERGY_BONUS_SCORE
            else: 
                for i1 in indices_1:
                    for i2 in indices_2:
                         arch1, arch2 = archetypes[i1], archetypes[i2]
                         if (arch1, arch2) in valid_pairs or (arch2, arch1) in valid_pairs:
                            total_score += SYNERGY_BONUS_SCORE
                            
        # 3. Phạt trùng lặp
        ids = [p[ID_COLUMN] for p in team]
        if len(ids) != len(set(ids)):
            total_score -= 100 

        return total_score

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(self.formation_slots) - 1)
        child1 = copy.deepcopy(parent1[:crossover_point] + parent2[crossover_point:])
        child2 = copy.deepcopy(parent2[:crossover_point] + parent1[crossover_point:])
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.MUTATION_RATE:
            slot_idx = random.randint(0, len(self.formation_slots) - 1)
            pool = self.candidates_pool[slot_idx]
            # Đột biến trong top 30
            top_pool = pool[:30] 
            individual[slot_idx] = random.choice(top_pool)
        return individual

    def repair_team(self, team):
        used_ids = set()
        new_team = []
        for slot_idx, player in enumerate(team):
            if player[ID_COLUMN] in used_ids:
                pool = self.candidates_pool[slot_idx]
                replacement = player
                for cand in pool:
                    if cand[ID_COLUMN] not in used_ids:
                        replacement = cand
                        break
                new_team.append(replacement)
                used_ids.add(replacement[ID_COLUMN])
            else:
                new_team.append(player)
                used_ids.add(player[ID_COLUMN])
        return new_team

    def run(self):
        self.precompute_candidates()
        
        # --- KHỞI TẠO ---
        population = []
        for _ in range(5):
            population.append(self.create_individual(smart_seed=True))
        for _ in range(self.POPULATION_SIZE - 5):
            population.append(self.create_individual(smart_seed=False))
        
        print(f"\n   -> Bắt đầu tiến hóa qua {self.GENERATIONS} thế hệ...")
        print(f"   ------------------------------------------------")
        
        global_best_team = None
        global_best_score = -float('inf')

        for gen in range(self.GENERATIONS):
            scored_population = [(team, self.calculate_fitness(team)) for team in population]
            
            current_gen_best_team, current_gen_best_score = max(scored_population, key=lambda x: x[1])
            
            if current_gen_best_score > global_best_score:
                global_best_score = current_gen_best_score
                global_best_team = copy.deepcopy(current_gen_best_team)
            
            # In ra thông tin
            print(f"      Gen {gen + 1:02d}: Best = {current_gen_best_score:.2f} | Global Best = {global_best_score:.2f}")
            
            # --- EVOLUTION ---
            next_generation = [copy.deepcopy(global_best_team)]
            
            scored_population.sort(key=lambda x: x[1], reverse=True)
            top_parents = [team for team, score in scored_population[:int(self.POPULATION_SIZE * 0.4)]]
            
            # Elitism
            num_elites = int(self.POPULATION_SIZE * 0.1)
            for i in range(num_elites):
                if len(next_generation) < self.POPULATION_SIZE:
                    next_generation.append(copy.deepcopy(top_parents[i]))
            
            while len(next_generation) < self.POPULATION_SIZE:
                parent1 = random.choice(top_parents)
                parent2 = random.choice(top_parents)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.repair_team(self.mutate(child1))
                child2 = self.repair_team(self.mutate(child2))
                
                if len(next_generation) < self.POPULATION_SIZE: next_generation.append(child1)
                if len(next_generation) < self.POPULATION_SIZE: next_generation.append(child2)
            
            population = next_generation
            
        return global_best_team
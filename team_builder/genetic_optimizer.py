import random
import pandas as pd
from config import ID_COLUMN, POSITION_COLUMN, NAME_COLUMN, OVERALL_COLUMN, TACTICAL_BONUSES, SYNERGY_BONUS_SCORE
from utils import calculate_ml_suitability_score, get_player_group

class GeneticTeamBuilder:
    def __init__(self, dataframe, formation_slots, models_dict, tactic_name='Balanced (Cân bằng)'):
        self.df = dataframe
        self.formation_slots = formation_slots
        self.models_dict = models_dict
        self.tactic_name = tactic_name
        
        self.POPULATION_SIZE = 40       
        self.GENERATIONS = 20           
        self.CANDIDATES_PER_SLOT = 15   
        self.MUTATION_RATE = 0.1        
        self.candidates_pool = [] 
        
    def precompute_candidates(self):
        print(f"   -> Đang sàng lọc ứng viên cho chiến thuật: {self.tactic_name}...")
        self.candidates_pool = []
        
        for slot_idx, position_req in enumerate(self.formation_slots):
            req_group = get_player_group(position_req)
            
            # Lọc ứng viên cơ bản (Top OVR)
            candidates = self.df.sort_values(by=OVERALL_COLUMN, ascending=False).head(300).copy()
            
            # Lấy thêm hậu vệ giỏi chuyền nếu cần tìm tiền vệ
            if req_group == 'Midfielder':
                ball_playing_cbs = self.df[
                    (self.df[POSITION_COLUMN].isin(['CB', 'RCB', 'LCB'])) & 
                    (self.df['Short Passing'] > 78)
                ].copy()
                
                inverted_fullbacks = self.df[
                    (self.df[POSITION_COLUMN].isin(['RB', 'LB', 'RWB', 'LWB'])) & 
                    (self.df['Vision'] > 74) & 
                    ( (self.df['Long Passing'] > 74) | (self.df['Short Passing'] > 78) )
                ].copy()
                
                candidates = pd.concat([candidates, ball_playing_cbs, inverted_fullbacks]).drop_duplicates(subset=[ID_COLUMN])

            # Tính điểm ML (có kèm Tactic Name)
            candidates['ml_score'] = candidates.apply(
                lambda row: calculate_ml_suitability_score(row, position_req, self.models_dict, self.tactic_name), 
                axis=1
            )
            
            top_candidates = candidates.sort_values(by='ml_score', ascending=False).head(self.CANDIDATES_PER_SLOT)
            pool_for_slot = top_candidates.to_dict('records')
            self.candidates_pool.append(pool_for_slot)
            
    def create_individual(self):
        team = []
        used_ids = set()
        for slot_idx in range(len(self.formation_slots)):
            pool = self.candidates_pool[slot_idx]
            candidate = random.choice(pool)
            retries = 0
            while candidate[ID_COLUMN] in used_ids and retries < 10:
                candidate = random.choice(pool)
                retries += 1
            team.append(candidate)
            used_ids.add(candidate[ID_COLUMN])
        return team

    def calculate_fitness(self, team):
        total_score = 0
        archetypes = [p.get('Archetype', 'Generic') for p in team]
        
        for player in team:
            total_score += player['ml_score']
            
        for (pos1, pos2), valid_pairs in TACTICAL_BONUSES.items():
            indices_1 = [i for i, pos in enumerate(self.formation_slots) if pos == pos1]
            indices_2 = [i for i, pos in enumerate(self.formation_slots) if pos == pos2]
            
            if pos1 == pos2:
                import itertools
                for i1, i2 in itertools.combinations(indices_1, 2):
                    arch1 = archetypes[i1]
                    arch2 = archetypes[i2]
                    if (arch1, arch2) in valid_pairs or (arch2, arch1) in valid_pairs:
                        total_score += SYNERGY_BONUS_SCORE
            else:
                for i1 in indices_1:
                    for i2 in indices_2:
                         arch1 = archetypes[i1]
                         arch2 = archetypes[i2]
                         if (arch1, arch2) in valid_pairs or (arch2, arch1) in valid_pairs:
                            total_score += SYNERGY_BONUS_SCORE
        return total_score

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(self.formation_slots) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.MUTATION_RATE:
            slot_idx = random.randint(0, len(self.formation_slots) - 1)
            pool = self.candidates_pool[slot_idx]
            individual[slot_idx] = random.choice(pool)
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
        population = [self.create_individual() for _ in range(self.POPULATION_SIZE)]
        print(f"\n   -> Bắt đầu tiến hóa qua {self.GENERATIONS} thế hệ...")
        
        for gen in range(self.GENERATIONS):
            scored_population = [(team, self.calculate_fitness(team)) for team in population]
            scored_population.sort(key=lambda x: x[1], reverse=True)
            current_best_score = scored_population[0][1]
            print(f"      Gen {gen + 1:02d}: Best Fitness = {current_best_score:.2f}")
            
            top_parents = [team for team, score in scored_population[:self.POPULATION_SIZE // 2]]
            next_generation = top_parents[:] 
            
            while len(next_generation) < self.POPULATION_SIZE:
                parent1 = random.choice(top_parents)
                parent2 = random.choice(top_parents)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.repair_team(self.mutate(child1))
                child2 = self.repair_team(self.mutate(child2))
                next_generation.extend([child1, child2])
            
            population = next_generation[:self.POPULATION_SIZE]
            
        return population[0]
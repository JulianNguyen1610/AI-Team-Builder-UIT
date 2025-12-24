import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

def draw_vertical_pitch(ax):
    """Vẽ sân bóng dọc (Vertical Pitch) với tỉ lệ chuẩn."""
    width = 68
    length = 105
    
    # 1. VẼ NỀN SÂN
    base_color = '#387B44' 
    stripe_color = '#469152'
    
    border = 5
    background = Rectangle((-border, -border), width + 2*border, length + 2*border, 
                           facecolor=base_color, edgecolor='none', zorder=0)
    ax.add_patch(background)
    
    # Sọc cỏ
    num_stripes = 18
    stripe_height = length / num_stripes
    for i in range(0, num_stripes, 2):
        stripe = Rectangle((0, i * stripe_height), width, stripe_height, 
                           facecolor=stripe_color, edgecolor='none', zorder=0)
        ax.add_patch(stripe)

    # --- ĐƯỜNG KẺ ---
    line_color = 'white'
    line_width = 2
    z_lines = 1

    # Biên và giữa sân
    plt.plot([0, width, width, 0, 0], [0, 0, length, length, 0], color=line_color, linewidth=line_width, zorder=z_lines)
    plt.plot([0, width], [length/2, length/2], color=line_color, linewidth=line_width, zorder=z_lines)
    centre_circle = Circle((width/2, length/2), 9.15, color=line_color, fill=False, linewidth=line_width, zorder=z_lines)
    ax.add_patch(centre_circle)
    centre_spot = Circle((width/2, length/2), 0.6, color=line_color, zorder=z_lines)
    ax.add_patch(centre_spot)
    
    # Vòng cấm (Dưới & Trên)
    for y_base, direction in [(0, 1), (length, -1)]:
        # Box lớn
        plt.plot([13.85, 13.85, 54.15, 54.15], 
                 [y_base, y_base + 16.5*direction, y_base + 16.5*direction, y_base], 
                 color=line_color, linewidth=line_width, zorder=z_lines)
        # Box nhỏ
        plt.plot([24.85, 24.85, 43.15, 43.15], 
                 [y_base, y_base + 5.5*direction, y_base + 5.5*direction, y_base], 
                 color=line_color, linewidth=line_width, zorder=z_lines)
    
    # Cung tròn và chấm phạt đền
    penalty_arc_bottom = Arc((width/2, 11), 18.3, 18.3, theta1=53, theta2=127, color=line_color, linewidth=line_width, zorder=z_lines)
    ax.add_patch(penalty_arc_bottom)
    penalty_spot_bottom = Circle((width/2, 11), 0.6, color=line_color, zorder=z_lines)
    ax.add_patch(penalty_spot_bottom)

    penalty_arc_top = Arc((width/2, length-11), 18.3, 18.3, theta1=233, theta2=307, color=line_color, linewidth=line_width, zorder=z_lines)
    ax.add_patch(penalty_arc_top)
    penalty_spot_top = Circle((width/2, length-11), 0.6, color=line_color, zorder=z_lines)
    ax.add_patch(penalty_spot_top)

    # Khung thành
    plt.plot([30.34, 37.66], [-0.5, -0.5], color=line_color, linewidth=4, zorder=z_lines)
    plt.plot([30.34, 37.66], [length+0.5, length+0.5], color=line_color, linewidth=4, zorder=z_lines)

def visualize_team(team_list, formation_name, team_name):
    """Vẽ đội hình thoáng hơn, phân tán đều sân."""
    plt.ioff()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(9, 13)) 
    fig.patch.set_facecolor('#387B44') 
    ax.set_aspect('equal')
    
    draw_vertical_pitch(ax)
    
    center_x = 34 
    
    # --- CẬP NHẬT TỌA ĐỘ MỚI (KÉO GIÃN TRỤC Y) ---
    coordinates = {
        # GK: Giữ thấp nhưng không quá sát đáy
        'GK': (center_x, 8), 
        'SW': (center_x, 16),
        
        # HẬU VỆ: Đẩy lên Y=24-26 (Thoát khỏi vòng cấm 16m50)
        'LB': (8, 26), 'LWB': (8, 36),
        'LCB': (22, 24), 
        'CB': (center_x, 24), 
        'RCB': (46, 24), 
        'RB': (60, 26), 'RWB': (60, 36),
        
        # TIỀN VỆ TRỤ: Đẩy lên Y=42-45 (Giữa sân nhà và vòng tròn trung tâm)
        'LDM': (24, 42), 'CDM': (center_x, 42), 'RDM': (44, 42),
        
        # TIỀN VỆ TRUNG TÂM: Đẩy lên Y=60 (Qua vạch giữa sân)
        'LM': (6, 62), 
        'LCM': (22, 60), 'CM': (center_x, 60), 'RCM': (46, 60),
        'RM': (62, 62),
        
        # TIỀN VỆ CÔNG: Đẩy lên Y=78 (Sát vòng cấm đối phương)
        'LAM': (22, 78), 'CAM': (center_x, 78), 'RAM': (46, 78),
        
        # TIỀN ĐẠO: Đẩy lên Y=92-96 (Trong vòng cấm đối phương)
        'LW': (10, 92), 'RW': (58, 92),
        'ST': (center_x, 96), 'CF': (center_x, 88),
        
        # CẶP TIỀN ĐẠO (2 ST): Tách rộng ra (Y=94)
        'LS': (24, 94), 'RS': (44, 94) 
    }

    from config import FORMATION_SLOTS
    slots = FORMATION_SLOTS.get(formation_name, [])
    
    if not slots:
        print("Chưa định nghĩa slots.")
        return

    # Logic đếm số lượng vị trí
    position_counts = {pos: slots.count(pos) for pos in set(slots)}
    current_counts = {pos: 0 for pos in set(slots)}

    for i, player in enumerate(team_list):
        pos_label = slots[i]
        
        current_counts[pos_label] += 1
        count_so_far = current_counts[pos_label]
        total_in_pos = position_counts[pos_label]
        
        # Lấy tọa độ mặc định
        coord = coordinates.get(pos_label, (center_x, 50))
        
        # --- XỬ LÝ TÁCH VỊ TRÍ (Cập nhật cho thoáng) ---
        
        # 1. CẶP TIỀN ĐẠO (QUAN TRỌNG)
        if pos_label in ['ST', 'CF'] and total_in_pos == 2:
            if count_so_far == 1: coord = coordinates['LS'] # Trái
            if count_so_far == 2: coord = coordinates['RS'] # Phải
            
        # 2. CẶP TRUNG VỆ
        elif pos_label == 'CB':
            if total_in_pos == 2:
                if count_so_far == 1: coord = coordinates['LCB']
                if count_so_far == 2: coord = coordinates['RCB']
            elif total_in_pos == 3:
                # 3 trung vệ thì giãn đều ra
                if count_so_far == 1: coord = (20, 24) # Lệch trái
                if count_so_far == 2: coord = (center_x, 24) # Giữa
                if count_so_far == 3: coord = (48, 24) # Lệch phải

        # 3. CẶP CDM
        elif pos_label == 'CDM' and total_in_pos == 2:
            if count_so_far == 1: coord = coordinates['LDM']
            if count_so_far == 2: coord = coordinates['RDM']

        # 4. CẶP CM
        elif pos_label == 'CM' and total_in_pos == 2:
            if count_so_far == 1: coord = coordinates['LCM']
            if count_so_far == 2: coord = coordinates['RCM']
            
        # 5. CẶP CAM
        elif pos_label == 'CAM' and total_in_pos == 2:
            if count_so_far == 1: coord = coordinates['LAM']
            if count_so_far == 2: coord = coordinates['RAM']

        # --- VẼ CẦU THỦ ---
        # Vòng tròn
        player_circle = Circle(coord, 3.0, facecolor='#222222', edgecolor='#FFD700', linewidth=2, zorder=10)
        ax.add_patch(player_circle)
        
        # Tên
        short_name = player.get('Name', 'Unknown').split()[-1]
        if len(short_name) > 9: short_name = short_name[:7] + ".."
            
        ax.text(coord[0], coord[1]-5.5, short_name, ha='center', va='top', 
                fontsize=10, color='white', fontweight='bold', zorder=11,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Vị trí
        ax.text(coord[0], coord[1]+0.8, pos_label, ha='center', va='center', 
                fontsize=7, color='#FFD700', fontweight='bold', zorder=11)
        
        # OVR
        ax.text(coord[0], coord[1]-1.2, str(player.get('OVR', '')), ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold', zorder=11)

    # Tiêu đề
    plt.title(f"{team_name.upper()} ({formation_name})", color='white', fontsize=18, fontweight='bold', pad=20)
    
    plt.xlim(-5, 73)
    plt.ylim(-5, 110)
    plt.axis('off') 
    
    print("Đang hiển thị sơ đồ... (Đóng cửa sổ để kết thúc)")
    plt.show(block=True)
import streamlit as st

# --- QUAN TRỌNG: KHÓA MATPLOTLIB NGAY ĐẦU FILE ---
import matplotlib
matplotlib.use('Agg') # Chế độ không hiển thị cửa sổ để tránh treo
import matplotlib.pyplot as plt
# -------------------------------------------------

import pandas as pd

# Import các hàm từ dự án
from team_builder import build_team, load_models
from model_trainer import preprocess_data
from config import FILE_PATH, FORMATION_SLOTS, ID_COLUMN, NAME_COLUMN, OVERALL_COLUMN, POSITION_COLUMN, TACTICAL_PROFILES
from visualizer import draw_pitch 

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Football Team Builder",
    page_icon="⚽",
    layout="wide"
)

# --- 1. HÀM LOAD DỮ LIỆU & MODEL (ĐÃ TẮT CACHE ĐỂ TRÁNH TREO) ---
# Lưu ý: Tôi đã comment lại dòng @st.cache để debug
# @st.cache_resource 
def load_resources_safe():
    print("DEBUG: Đang load Models...")
    models = load_models()
    print("DEBUG: Load Models xong!")
    return models

# @st.cache_data
def load_dataset_safe():
    print("DEBUG: Đang load CSV...")
    try:
        df = pd.read_csv(FILE_PATH)
        processed_df = preprocess_data(df)
        print("DEBUG: Load CSV xong!")
        return processed_df
    except FileNotFoundError:
        return None

# --- 2. GIAO DIỆN CHÍNH ---
def main():
    st.title("⚽ AI Football Team Optimization System")
    st.markdown("Hệ thống xây dựng đội hình tối ưu sử dụng **Machine Learning** và **Genetic Algorithm**.")

    st.sidebar.header("🛠 Cấu hình Đội hình")
    
    # Load tài nguyên trực tiếp (có in log ra terminal để bạn theo dõi)
    with st.spinner("Đang khởi động hệ thống AI... (Vui lòng chờ)"):
        models = load_resources_safe()
        df = load_dataset_safe()

    if df is None or not models:
        st.error("Lỗi: Không tìm thấy dữ liệu hoặc mô hình. Hãy kiểm tra Terminal để xem chi tiết.")
        return

    # --- PHẦN NHẬP LIỆU ---
    filter_type = st.sidebar.radio("Chọn chế độ lọc:", ("Theo CLB (Team Color)", "Theo Quốc gia (Nation)"))
    
    if filter_type == "Theo Quốc gia (Nation)":
        filter_mode = 'nation'
        all_nations = sorted(df['Nation'].unique().astype(str))
        default_index = all_nations.index("England") if "England" in all_nations else 0
        filter_name = st.sidebar.selectbox("Chọn Quốc gia:", all_nations, index=default_index)
    else:
        filter_mode = 'team'
        filter_name = st.sidebar.text_input("Nhập tên CLB (VD: Real Madrid, Arsenal):", "Real Madrid")

    formation_options = list(FORMATION_SLOTS.keys())
    formation = st.sidebar.selectbox("Chọn Sơ đồ chiến thuật:", formation_options)

    tactic_options = list(TACTICAL_PROFILES.keys())
    selected_tactic = st.sidebar.selectbox("Chọn Lối đá (Tactical Style):", tactic_options)

    use_genetic = st.sidebar.checkbox("Sử dụng Genetic Algorithm (Khuyên dùng)", value=True)
    
    # --- NÚT CHẠY ---
    if st.sidebar.button("🚀 Xây dựng Đội hình", type="primary"):
        if not filter_name:
            st.warning("Vui lòng nhập tên Đội bóng hoặc Quốc gia.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 AI đang phân tích dữ liệu và tối ưu hóa...")
            progress_bar.progress(30)
            
            try:
                best_team = build_team(
                    df, filter_name, formation, models, 
                    filter_type=filter_mode, 
                    use_genetic_algo=use_genetic,
                    tactic_name=selected_tactic
                )
                
                progress_bar.progress(100)
                status_text.empty()

                if best_team:
                    st.success(f"Đã tìm thấy đội hình tối ưu cho: **{filter_name}** ({formation})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    avg_ovr = sum([p[OVERALL_COLUMN] for p in best_team]) / 11
                    with col1: st.metric("Điểm OVR Trung bình", f"{avg_ovr:.1f}")
                    with col2: st.metric("Số lượng cầu thủ", "11")
                    with col3: st.metric("Thuật toán", "Genetic AI" if use_genetic else "Greedy")
                    with col4: st.metric("Lối đá", selected_tactic)

                    st.subheader("📋 Sơ đồ Chiến thuật")
                    # Gọi hàm vẽ
                    pitch_fig = draw_pitch(best_team, formation, filter_name)
                    st.pyplot(pitch_fig)
                    # Đóng figure để giải phóng bộ nhớ
                    plt.close(pitch_fig) 

                    st.subheader("📊 Chi tiết Cầu thủ")
                    display_data = []
                    slots = FORMATION_SLOTS[formation]
                    for i, player in enumerate(best_team):
                        display_data.append({
                            "Vị trí": slots[i],
                            "Tên cầu thủ": player[NAME_COLUMN],
                            "OVR": player[OVERALL_COLUMN],
                            "Vai trò (Role)": player.get('Archetype', 'N/A'),
                            "Tuổi": player.get('Age', 'N/A'),
                            "Chân thuận": player.get('Preferred foot', 'N/A'),
                            "Weak Foot": player.get('Weak foot', 'N/A')
                        })
                    
                    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
                    
                else:
                    st.error("Không tìm thấy đội hình phù hợp. Hãy thử đổi tên đội hoặc sơ đồ.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi chạy thuật toán: {e}")
                print(e)

if __name__ == "__main__":
    main()
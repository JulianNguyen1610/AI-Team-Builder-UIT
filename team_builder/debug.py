import streamlit as st
import time

st.set_page_config(page_title="Debug", layout="wide")
st.title("🕵️‍♂️ Chế độ Dò Lỗi")

def log(msg):
    st.write(f"✅ {msg}")
    print(msg) # In cả ra terminal
    time.sleep(0.1) # Nghỉ xíu để kịp in

log("1. Bắt đầu Streamlit...")

# --- KIỂM TRA MATPLOTLIB ---
try:
    import matplotlib
    matplotlib.use('Agg') # Bắt buộc phải có dòng này
    import matplotlib.pyplot as plt
    log("2. Matplotlib OK (Backend: Agg)")
except Exception as e:
    st.error(f"Lỗi Matplotlib: {e}")

# --- KIỂM TRA CONFIG ---
try:
    import config
    log("3. Config.py OK")
except Exception as e:
    st.error(f"Lỗi Config: {e}")

# --- KIỂM TRA UTILS ---
try:
    import utils
    log("4. Utils.py OK")
except Exception as e:
    st.error(f"Lỗi Utils: {e}")

# --- KIỂM TRA VISUALIZER (NGHI PHẠM SỐ 1) ---
try:
    import visualizer
    log("5. Visualizer.py OK")
except Exception as e:
    st.error(f"Lỗi Visualizer: {e}")

# --- KIỂM TRA DATA LOADING (NGHI PHẠM SỐ 2) ---
try:
    import pandas as pd
    from config import FILE_PATH
    log(f"6. Chuẩn bị đọc file: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)
    log(f"7. Đọc CSV thành công! ({len(df)} dòng)")
except Exception as e:
    st.error(f"Lỗi Đọc Data: {e}")

# --- KIỂM TRA MODEL LOADING (NGHI PHẠM SỐ 3) ---
try:
    import joblib
    import os
    from config import MODEL_STORAGE_PATH, POSITION_GROUPS
    log("8. Chuẩn bị load Model...")
    
    count = 0
    for group in POSITION_GROUPS.keys():
        path = os.path.join(MODEL_STORAGE_PATH, f"model_{group.lower()}.joblib")
        if os.path.exists(path):
            _ = joblib.load(path)
            count += 1
    log(f"9. Load Model thành công! ({count} models)")
except Exception as e:
    st.error(f"Lỗi Load Model: {e}")

# --- KIỂM TRA CÁC MODULE KHÁC ---
try:
    import genetic_optimizer
    log("10. Genetic Optimizer OK")
    import team_builder
    log("11. Team Builder OK")
except Exception as e:
    st.error(f"Lỗi Import Logic: {e}")

st.success("🎉 NẾU BẠN THẤY DÒNG NÀY THÌ MỌI THỨ ĐỀU ỔN!")
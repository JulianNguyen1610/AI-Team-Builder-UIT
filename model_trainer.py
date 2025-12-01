import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # Thư viện để lưu và tải mô hình
import os
import numpy as np 

# Import các biến cấu hình từ file config.py
from config import FILE_PATH, MODEL_STORAGE_PATH, POSITION_COLUMN, OVERALL_COLUMN, FEATURES_COLUMNS, POSITION_GROUPS

def clean_height(height_value):
    """
    Chuyển đổi chiều cao dạng '191cm / 6'3"' sang số nguyên 191.
    """
    if pd.isna(height_value):
        return 175 # Giá trị mặc định nếu dữ liệu trống
        
    str_val = str(height_value).strip()
    
    try:
        # TRƯỜNG HỢP 1: Dữ liệu chứa 'cm' (Ưu tiên nhất)
        # Ví dụ: "191cm / 6'3"" hoặc "191cm"
        if 'cm' in str_val:
            # Lấy phần trước chữ 'cm', loại bỏ khoảng trắng dư thừa
            # "191cm / 6'3"" -> lấy "191"
            clean_str = str_val.split('cm')[0].strip()
            return int(clean_str)
            
        # TRƯỜNG HỢP 2: Dữ liệu chỉ có hệ Feet (Ví dụ: "6'3"")
        elif "'" in str_val:
            # Xử lý nếu chuỗi là " / 6'3"" (trường hợp hiếm nếu mất phần cm)
            if '/' in str_val:
                str_val = str_val.split('/')[-1].strip()
            
            str_val = str_val.replace('"', '') # Bỏ dấu "
            parts = str_val.split("'")
            feet = int(parts[0])
            inches = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            # Công thức đổi: 1 foot = 30.48cm, 1 inch = 2.54cm
            return int(feet * 30.48 + inches * 2.54)
            
        # TRƯỜNG HỢP 3: Chỉ là số (Ví dụ: 191)
        else:
            return int(float(str_val))
            
    except (ValueError, IndexError):
        # Nếu lỗi format quá lạ, trả về mặc định
        return 175

def preprocess_data(df):
    """Hàm tiền xử lý dữ liệu chung."""
    print("Bắt đầu tiền xử lý dữ liệu...")
    df['Preferred foot_numeric'] = df['Preferred foot'].apply(lambda x: 1 if x == 'Right' else 0)
    
    # Dọn dẹp các giá trị thiếu trong các cột features
    for col in FEATURES_COLUMNS:
        if col not in df.columns:
            print(f"Cảnh báo: Cột '{col}' không tồn tại. Bỏ qua...")
            FEATURES_COLUMNS.remove(col)
        else:
            # Chuyển thành số và điền giá trị thiếu bằng trung bình của cột
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
            
    # Xử lý cột vị trí
    df[POSITION_COLUMN] = df[POSITION_COLUMN].astype(str).apply(lambda x: x.split(',')[0].strip())
    if 'Height' in df.columns:
        print(" -> Đang chuẩn hóa cột Height...")
        df['Height_clean'] = df['Height'].apply(clean_height)
    else:
        print("Cảnh báo: Không tìm thấy cột 'Height'. Dùng giá trị mặc định 180cm.")
        df['Height_clean'] = 180
    print("Tiền xử lý hoàn tất.")
    return df

def train_and_save_models():
    """Hàm chính để huấn luyện và lưu tất cả các mô hình."""
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Tải dữ liệu thành công từ '{FILE_PATH}'.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{FILE_PATH}'.")
        return

    df = preprocess_data(df)

    # Tạo thư mục 'models' nếu chưa tồn tại
    if not os.path.exists(MODEL_STORAGE_PATH):
        os.makedirs(MODEL_STORAGE_PATH)
        print(f"Đã tạo thư mục '{MODEL_STORAGE_PATH}' để lưu các mô hình.")

    print("\n--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")
    
    for group_name, positions_in_group in POSITION_GROUPS.items():
        print(f"\nĐang huấn luyện cho nhóm: {group_name}")
        
        position_df = df[df[POSITION_COLUMN].isin(positions_in_group)]
        
        if len(position_df) < 50:
            print(f"  -> Dữ liệu không đủ cho nhóm {group_name}. Bỏ qua.")
            continue
            
        X = position_df[FEATURES_COLUMNS]
        y = position_df[OVERALL_COLUMN]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Sử dụng RandomForestRegressor - một mô hình mạnh mẽ và linh hoạt
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=7)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions) # Tính MSE
        rmse = np.sqrt(mse)                           # Lấy căn bậc hai để ra RMSE
        print(f"  -> Huấn luyện xong. Độ lỗi trung bình (RMSE) trên tập test: {rmse:.2f}")
        
        # Lưu mô hình vào file
        model_filename = os.path.join(MODEL_STORAGE_PATH, f"model_{group_name.lower()}.joblib")
        joblib.dump(model, model_filename)
        print(f"  -> Đã lưu mô hình tại '{model_filename}'")

    print("\n--- HOÀN TẤT HUẤN LUYỆN VÀ LƯU TẤT CẢ CÁC MÔ HÌNH ---")

# Chạy hàm chính khi file này được thực thi
if __name__ == "__main__":
    train_and_save_models()
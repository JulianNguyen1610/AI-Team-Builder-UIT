# ⚽ AI Football Team Builder

Dự án môn học: Tư duy Trí tuệ Nhân tạo.
Hệ thống tự động xây dựng đội hình bóng đá tối ưu dựa trên dữ liệu cầu thủ FIFA/SoFIFA.

## 🚀 Tính năng nổi bật
- **Phân tích hình mẫu (Clustering):** Tự động nhận diện vai trò cầu thủ (Playmaker, Target Man, Anchor Man...).
- **Hệ chuyên gia (Machine Learning):** Sử dụng Random Forest để đánh giá mức độ phù hợp của cầu thủ cho từng vị trí.
- **Tối ưu hóa (Genetic Algorithm):** Sử dụng thuật toán di truyền để tìm ra đội hình 11 người có sự kết nối (Chemistry) tốt nhất.
- **Chiến thuật linh hoạt:** Hỗ trợ "Inverted Fullback" (Hậu vệ bó trong) và "Ball-Playing Defender".

## 🛠️ Cài đặt
1. Clone dự án:
   `git clone https://github.com/JulianNguyen1610/AI-Team-Builder-UIT`
2. Chuẩn bị dữ liệu:
   - Đặt file dữ liệu gốc (CSV) vào thư mục `team_builder/` với tên `data.csv`
   - File này sẽ được sử dụng làm file dữ liệu chính cho toàn bộ hệ thống
3. Cài đặt thư viện:
   `pip install pandas scikit-learn joblib matplotlib numpy`
4. Chạy chương trình (theo thứ tự):
   - **Bước 1:** `python team_builder/archetype_analyzer.py` 
     - Phân tích và phân loại cầu thủ theo hình mẫu (archetype)
     - Ghi kết quả trực tiếp vào `data.csv`
   - **Bước 2:** `python team_builder/model_trainer.py` 
     - Huấn luyện các mô hình AI cho từng nhóm vị trí
     - Sử dụng `data.csv` đã được phân tích ở bước 1
   - **Bước 3:** `python team_builder/team_builder.py` 
     - Xây dựng đội hình tối ưu sử dụng các mô hình đã huấn luyện
     - Sử dụng `data.csv` làm nguồn dữ liệu

## 📊 Kết quả Demo
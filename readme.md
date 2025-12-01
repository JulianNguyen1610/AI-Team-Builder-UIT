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
   `git clone https://github.com/JulianNguyen1610/FIFA-26-AI-Team-Builder`
2. Cài đặt thư viện:
   `pip install pandas scikit-learn`
3. Chạy chương trình:
   - Bước 1: `python archetype_analyzer.py` (Phân tích)
   - Bước 2: `python model_trainer.py` (Huấn luyện AI)
   - Bước 3: `python team_builder.py` (Xây đội hình)

## 📊 Kết quả Demo
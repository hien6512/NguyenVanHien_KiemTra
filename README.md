# 🩺 Dự đoán nguy cơ bệnh tiểu đường bằng Machine Learning

Bài kiểm tra môn **Trí tuệ nhân tạo**: xây dựng phần mềm AI dự đoán nguy cơ bệnh tiểu đường từ các thông số sức khỏe (tuổi, giới tính và 11 chỉ số xét nghiệm máu), có giao diện nhập thông tin người bệnh, sinh cảnh báo và giải thích kết quả.

**Sinh viên:** Nguyễn Văn Hiến – MSSV 26ACN020 – Lớp CNTT4

🔗 **Dùng thử trực tuyến:** https://nguyenvanhien-dudoantieuduong.streamlit.app

---

## Tính năng của giao diện

- Nhập thông tin người bệnh theo nhóm: thông tin chung, đường huyết, mỡ máu, chức năng thận, chức năng gan. Chỉ số chưa xét nghiệm có thể để trống.
- Nút điền nhanh dữ liệu mẫu: *Người khỏe mạnh*, *Nguy cơ cao*, *Xóa trắng*.
- **Cảnh báo 3 mức** theo xác suất mô hình dự đoán: thấp (< 30%), trung bình (30–70%), cao (≥ 70%).
- **Cảnh báo theo tiêu chuẩn y khoa** khi Glucose ≥ 7,0 mmol/L hoặc HbA1C ≥ 6,5%, độc lập với mô hình.
- **Giải thích kết quả**:
  - Chỉ số nào đang làm tăng nguy cơ, và tăng bao nhiêu điểm %.
  - Bảng đối chiếu từng chỉ số với ngưỡng tham chiếu (có phân biệt nam/nữ).
  - Khuyến nghị theo từng mức nguy cơ.

## Kết quả mô hình (tập Test – 707 người)

| Mô hình | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Mốc so sánh (đoán lớp đa số) | 0.7666 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.9901 | 0.9817 | 0.9758 | 0.9787 | 0.9992 |
| **Random Forest** (được chọn) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| SVM | 0.9760 | 0.9458 | 0.9515 | 0.9486 | 0.9976 |

Kết quả rất cao vì nhãn trong dữ liệu được xác định theo tiêu chuẩn chẩn đoán (Glucose lúc đói ≥ 7,0 mmol/L hoặc HbA1C ≥ 6,5%), nên mô hình chủ yếu học lại tiêu chuẩn này. Khi bỏ Glucose và HbA1C, F1-score chỉ còn khoảng 0,28. Phần phân tích chi tiết nằm trong file Word giải thích kèm theo bài nộp.

## Các bước thực hiện

1. **Xác định bài toán:** phân loại nhị phân (có / không có nguy cơ), ưu tiên Recall và F1-score.
2. **Thu thập dữ liệu:** `CSDL_TieuDuong.csv` gồm 4.707 người và 14 cột.
3. **Tiền xử lý:**
   - Điền dữ liệu thiếu bằng trung vị.
   - Đồng bộ cách ghi giới tính và nhãn.
   - Sửa lỗi nhập sai dấu thập phân (750 → 7.5).
   - Coi giá trị HbA1C điền sẵn là "chưa xét nghiệm".
   - Biến đổi log cho các chỉ số lệch phải để giảm ảnh hưởng của ngoại lai.
   - Mã hóa One-Hot, chuẩn hóa StandardScaler.
   - Chia dữ liệu 70% Train – 15% Validation – 15% Test.
4. **Chọn mô hình:** Logistic Regression, Random Forest, SVM, so với mốc đoán lớp đa số.
5. **Huấn luyện:** toàn bộ tiền xử lý và mô hình nằm trong một `Pipeline` của scikit-learn, chỉ học từ tập Train.
6. **Đánh giá:** chọn mô hình theo F1-score trên tập Validation, báo cáo trên tập Test. Có thêm ma trận nhầm lẫn và mức độ quan trọng của từng chỉ số.
7. **Giao diện:** Streamlit, có cảnh báo và giải thích kết quả.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `NguyenVanHien_KiemTra.py` | Bước 1 → 6: tiền xử lý, huấn luyện, đánh giá, lưu mô hình |
| `App_DuDoanTieuDuong.py` | Bước 7: giao diện Streamlit |
| `CSDL_TieuDuong.csv` | Dữ liệu gốc |
| `requirements.txt` | Danh sách thư viện |

Khi chạy, chương trình tự sinh thêm các file sau: `CSDL_TieuDuong_DaLamSach.csv` (dữ liệu sạch), `mo_hinh_tieu_duong.joblib` (mô hình) và thư mục `hinh_anh/` (biểu đồ).

## Chạy trên máy cá nhân

Cần Python 3.10 trở lên. Mở Terminal **tại thư mục chứa các file trên** rồi chạy:

```bash
python -m pip install -r requirements.txt
python NguyenVanHien_KiemTra.py        # Bước 1–6: in kết quả từng bước, lưu mô hình và biểu đồ
streamlit run App_DuDoanTieuDuong.py   # Bước 7: mở giao diện trên trình duyệt
```

Giao diện mở tại `http://localhost:8501`, hoặc `http://localhost:8501/NguyenVanHien` nếu có file cấu hình `.streamlit/config.toml`. Nếu chưa chạy file huấn luyện, giao diện tự huấn luyện mô hình ở lần mở đầu tiên.

## Công nghệ

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit

---

⚠️ *Sản phẩm phục vụ mục đích học tập. Kết quả dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.*

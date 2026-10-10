# -*- coding: utf-8 -*-
"""
BÀI KIỂM TRA MÔN TRÍ TUỆ NHÂN TẠO
Đề: Xây dựng phần mềm AI (Machine Learning) dự đoán nguy cơ bệnh tiểu đường
    từ thông số sức khỏe.
Sinh viên: Nguyễn Văn Hiến - 26ACN020 - CNTT4

File này thực hiện Bước 1 -> Bước 6 (xử lý dữ liệu, huấn luyện, đánh giá) và lưu
mô hình tốt nhất để giao diện App_DuDoanTieuDuong.py (Bước 7) sử dụng.

Cài thư viện (chạy trong Terminal):  python -m pip install -r requirements.txt
Chạy:                                 python NguyenVanHien_KiemTra.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # chỉ lưu hình ra file, không mở cửa sổ (để App gọi lại được)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Tránh lỗi UnicodeEncodeError khi in tiếng Việt ra file/pipe trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================

# Đường dẫn tính theo vị trí file code, chạy từ thư mục nào cũng đúng
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CSDL_TieuDuong.csv"
CLEAN_PATH = BASE_DIR / "CSDL_TieuDuong_DaLamSach.csv"
MODEL_PATH = BASE_DIR / "mo_hinh_tieu_duong.joblib"
HINH_DIR = BASE_DIR / "hinh_anh"

RANDOM_STATE = 42

COT_NHAN = "NguyCoTieuDuong"
COT_PHAN_LOAI = ["GioiTinh"]
COT_SO = [
    "Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL",
    "Triglycerid", "Creatinine", "Ure", "AcidUric", "AST", "ALT",
]

DON_VI = {
    "Tuoi": "tuổi", "Glucose": "mmol/L", "HbA1C": "%", "Cholesterol": "mmol/L",
    "HDL": "mmol/L", "LDL": "mmol/L", "Triglycerid": "mmol/L",
    "Creatinine": "µmol/L", "Ure": "mmol/L", "AcidUric": "µmol/L",
    "AST": "U/L", "ALT": "U/L",
}

# Đồng bộ nhãn: dữ liệu ghi lẫn TRUE/FALSE, 1/0, "Có nguy cơ"/"Không có nguy cơ"
MAP_NHAN = {
    "TRUE": 1, "True": 1, "1": 1, "Có nguy cơ": 1,
    "FALSE": 0, "False": 0, "0": 0, "Không có nguy cơ": 0,
}
TEN_NHAN = ["Không có nguy cơ", "Có nguy cơ"]

# Đồng bộ giới tính: dữ liệu ghi lẫn tiếng Việt và tiếng Anh
MAP_GIOI_TINH = {"Male": "Nam", "male": "Nam", "Female": "Nữ", "female": "Nữ", "NU": "Nữ", "Nu": "Nữ"}

# Khoảng giá trị CÓ THỂ XẢY RA về mặt sinh lý. Ngoài khoảng này là dữ liệu LỖI
# (nhập sai), khác với NGOẠI LAI (hiếm nhưng có thật, ví dụ bệnh nhân suy thận nặng).
KHOANG_HOP_LE = {
    "Tuoi": (1, 110),
    "Glucose": (1, 50),
    "HbA1C": (3, 20),
    "Cholesterol": (1, 20),
    "HDL": (0.2, 5),
    "LDL": (0.5, 10),
    "Triglycerid": (0.1, 100),
    "Creatinine": (1, 2500),
    "Ure": (0.5, 100),
    "AcidUric": (50, 1500),
    "AST": (1, 5000),
    "ALT": (1, 5000),
}

# 96% số dòng có HbA1C = 8.025 (kể cả người không có nguy cơ, trong khi HbA1C >= 6.5%
# đã là ngưỡng chẩn đoán tiểu đường) => đây là giá trị điền sẵn cho ô bị thiếu, không
# phải kết quả xét nghiệm thật. Ta đổi lại thành "thiếu" và để mô hình biết điều đó.
GIA_TRI_DIEN_SAN_HBA1C = 8.025

# Các chỉ số lệch phải mạnh (có ngoại lai rất lớn) -> biến đổi log trước khi chuẩn hóa
COT_LOG = ["Glucose", "Triglycerid", "Creatinine", "Ure", "AST", "ALT"]


def tieu_de(noi_dung):
    print("\n" + "=" * 72)
    print(noi_dung)
    print("=" * 72)


def luu_hinh(fig, ten_file):
    HINH_DIR.mkdir(exist_ok=True)
    fig.savefig(HINH_DIR / ten_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Đã lưu hình: hinh_anh/{ten_file}")


# ============================================================
# BƯỚC 1. XÁC ĐỊNH BÀI TOÁN
# ============================================================

def buoc_1_xac_dinh_bai_toan():
    tieu_de("BƯỚC 1. XÁC ĐỊNH BÀI TOÁN")
    print("- Loại bài toán: Phân loại nhị phân (Supervised Learning - Classification)")
    print("- Đầu vào (X) : Tuổi, giới tính và 11 chỉ số xét nghiệm máu")
    print("                (Glucose, HbA1C, mỡ máu, chức năng thận, chức năng gan)")
    print("- Đầu ra  (y) : NguyCoTieuDuong -> 1 = Có nguy cơ, 0 = Không có nguy cơ")
    print("- Mục tiêu    : Phát hiện được càng nhiều người có nguy cơ càng tốt")
    print("                => ưu tiên Recall và F1-score của lớp 'Có nguy cơ', không chỉ Accuracy")


# ============================================================
# BƯỚC 2. THU THẬP DỮ LIỆU
# ============================================================

def buoc_2_doc_du_lieu():
    tieu_de("BƯỚC 2. THU THẬP DỮ LIỆU")
    df = pd.read_csv(DATA_PATH)
    print(f"Đọc file {DATA_PATH.name}: {df.shape[0]} dòng, {df.shape[1]} cột")
    print("\n5 dòng đầu:")
    print(df.head().to_string())
    print("\nKiểu dữ liệu của các cột:")
    print(df.dtypes.to_string())
    print("\nThống kê mô tả:")
    print(df.describe().T.round(2).to_string())
    return df


# ============================================================
# BƯỚC 3.1. LÀM SẠCH DỮ LIỆU
# ============================================================

def xu_ly_du_lieu_thieu(df):
    print("\n--- a) Kiểm tra và xử lý dữ liệu thiếu ---")
    so_thieu = df.isna().sum()
    print(so_thieu[so_thieu > 0].to_string())
    print(f"Tổng: {int(so_thieu.sum())} ô thiếu trên {len(df)} dòng")
    print("Cách xử lý:")
    print("  - Dòng thiếu NHÃN: xóa (không dùng để học được)")
    print("  - Cột số thiếu: điền bằng TRUNG VỊ (ít bị ảnh hưởng bởi ngoại lai hơn trung bình)")
    print("  - Cột phân loại thiếu: điền bằng giá trị XUẤT HIỆN NHIỀU NHẤT")
    print("  - Việc điền được đặt trong Pipeline (Bước 3.2) và chỉ học từ tập Train,")
    print("    để thông tin của tập Test không bị lộ vào quá trình huấn luyện")
    return df.dropna(subset=[COT_NHAN])


def dong_bo_du_lieu(df):
    print("\n--- b) Đồng bộ dữ liệu ---")
    print("Giá trị GioiTinh trước khi đồng bộ:", df["GioiTinh"].value_counts(dropna=False).to_dict())
    print("Giá trị nhãn trước khi đồng bộ   :", df[COT_NHAN].value_counts(dropna=False).to_dict())

    df["GioiTinh"] = df["GioiTinh"].str.strip().replace(MAP_GIOI_TINH)
    df[COT_NHAN] = df[COT_NHAN].astype(str).str.strip().map(MAP_NHAN)

    so_nhan_la = df[COT_NHAN].isna().sum()
    if so_nhan_la:
        print(f"Xóa {so_nhan_la} dòng có nhãn không nhận dạng được")
        df = df.dropna(subset=[COT_NHAN])
    df[COT_NHAN] = df[COT_NHAN].astype(int)

    print("Giá trị GioiTinh sau khi đồng bộ :", df["GioiTinh"].value_counts(dropna=False).to_dict())
    print("Giá trị nhãn sau khi đồng bộ     :", df[COT_NHAN].value_counts().to_dict(), "(1 = Có nguy cơ)")
    return df


def xu_ly_du_lieu_loi(df):
    print("\n--- c) Kiểm tra và xử lý dữ liệu lỗi ---")
    print("Giá trị nằm ngoài khoảng sinh lý có thể xảy ra được coi là lỗi nhập liệu.")
    print("Lỗi hay gặp nhất là gõ thiếu dấu thập phân (8.025 -> 8025), nên ta thử chia 10")
    print("(tối đa 3 lần) cho tới khi giá trị hợp lý; nếu vẫn không hợp lý thì coi là thiếu.")

    for cot, (nho_nhat, lon_nhat) in KHOANG_HOP_LE.items():
        goc = df[cot]
        moi = goc.copy()
        for _ in range(3):
            moi = moi.where(moi <= lon_nhat, moi / 10)
        moi = moi.where(moi.between(nho_nhat, lon_nhat))

        thay_doi = goc.notna() & (moi != goc)
        for chi_so in df.index[thay_doi]:
            nhan = TEN_NHAN[df.at[chi_so, COT_NHAN]]
            print(f"  {cot:12s}: {goc[chi_so]:>8g} -> {moi[chi_so]:<6g} (nhãn: {nhan})")
        df[cot] = moi

    print("Kiểm chứng với nhãn: Glucose sau khi sửa >= 7.0 thì nhãn là 'Có nguy cơ', < 7.0 thì")
    print("'Không có nguy cơ', khớp với tiêu chuẩn chẩn đoán => cách sửa hợp lý.")

    dien_san = np.isclose(df["HbA1C"], GIA_TRI_DIEN_SAN_HBA1C)
    print(f"\nHbA1C = {GIA_TRI_DIEN_SAN_HBA1C} xuất hiện ở {dien_san.mean():.1%} số dòng, kể cả người KHÔNG có nguy cơ,")
    print("trong khi HbA1C >= 6.5% đã là ngưỡng chẩn đoán tiểu đường => đây là giá trị điền sẵn")
    print("cho ô thiếu. Đổi thành 'thiếu' để mô hình biết người đó chưa có kết quả HbA1C thật.")
    df.loc[dien_san, "HbA1C"] = np.nan

    print("\nGhi chú: các cột Cholesterol, HDL, LDL, Triglycerid, AcidUric cũng có 77-99% dòng")
    print("cùng một giá trị nên mang rất ít thông tin, nhưng không mâu thuẫn y khoa nên giữ nguyên.")
    return df


def kiem_tra_ngoai_lai(df):
    print("\n--- d) Kiểm tra và xử lý dữ liệu ngoại lai (phương pháp IQR) ---")
    for cot in COT_SO:
        cot_du_lieu = df[cot].dropna()
        q1, q3 = cot_du_lieu.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            gia_tri_chinh = cot_du_lieu.mode()[0]
            ty_le = (cot_du_lieu == gia_tri_chinh).mean()
            print(f"  {cot:12s}: IQR = 0 vì {ty_le:.1%} dòng cùng giá trị {gia_tri_chinh:g} -> IQR không áp dụng được")
            continue
        duoi, tren = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        so_ngoai_lai = ((cot_du_lieu < duoi) | (cot_du_lieu > tren)).sum()
        print(f"  {cot:12s}: {so_ngoai_lai:4d} ngoại lai ngoài [{duoi:.2f}, {tren:.2f}], lớn nhất = {cot_du_lieu.max():g}")

    print("Cách xử lý:")
    print("  - KHÔNG xóa: trong y tế, giá trị cực đoan thường là bệnh nhân nặng thật")
    print("    (ví dụ Glucose 40 mmol/L, Creatinine 1500 µmol/L ở người suy thận).")
    print("  - KHÔNG cắt theo phân vị: cột HbA1C có phân vị 1% = 6.8%, cắt sẽ biến người")
    print("    bình thường (HbA1C 5%) thành mức tiểu đường -> sai lệch nghiêm trọng.")
    print(f"  - Biến đổi log cho các chỉ số lệch phải {COT_LOG}:")
    print("    giữ nguyên thứ tự giá trị nhưng kéo các giá trị cực lớn lại gần,")
    print("    giảm ảnh hưởng của ngoại lai lên Logistic Regression và SVM.")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, cot in zip(axes.flat, COT_SO):
        sns.boxplot(y=df[cot], ax=ax, color="#7fb3d5")
        ax.set_title(f"{cot} ({DON_VI[cot]})")
        ax.set_ylabel("")
    fig.suptitle("Phân bố và ngoại lai của các chỉ số", fontsize=14)
    luu_hinh(fig, "1_ngoai_lai_boxplot.png")


def xu_ly_trung_lap(df):
    print("\n--- e) Kiểm tra dữ liệu trùng lặp ---")
    so_trung = df.duplicated().sum()
    print(f"Số dòng trùng hoàn toàn: {so_trung}")
    return df.drop_duplicates().reset_index(drop=True)


def buoc_3_1_lam_sach(df):
    tieu_de("BƯỚC 3.1. LÀM SẠCH DỮ LIỆU")
    df = df.copy()
    df = xu_ly_du_lieu_thieu(df)
    df = dong_bo_du_lieu(df)
    df = xu_ly_du_lieu_loi(df)
    kiem_tra_ngoai_lai(df)
    df = xu_ly_trung_lap(df)

    print(f"\nKích thước sau làm sạch: {df.shape[0]} dòng, {df.shape[1]} cột")
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã lưu dữ liệu sạch: {CLEAN_PATH.name}")

    ty_le = df[COT_NHAN].value_counts(normalize=True)
    print(f"\nPhân bố nhãn: Không có nguy cơ {ty_le[0]:.1%} - Có nguy cơ {ty_le[1]:.1%}")
    print("=> Dữ liệu lệch lớp: đoán bừa 'Không có nguy cơ' cho tất cả đã đạt "
          f"Accuracy {ty_le.max():.1%}, nên phải xem thêm Recall/F1.")

    ten_nhan = df[COT_NHAN].map(dict(enumerate(TEN_NHAN)))
    mau = {"Không có nguy cơ": "#58d68d", "Có nguy cơ": "#ec7063"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=ten_nhan, hue=ten_nhan, order=TEN_NHAN, palette=mau, legend=False, ax=axes[0])
    axes[0].set_title("Phân bố nhãn")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Số người")
    sns.boxplot(x=ten_nhan, y=df["Glucose"], hue=ten_nhan, order=TEN_NHAN, palette=mau, legend=False, ax=axes[1])
    axes[1].axhline(7.0, color="red", linestyle="--", label="Ngưỡng chẩn đoán 7.0 mmol/L")
    axes[1].set_yscale("log")
    axes[1].set_title("Glucose theo nhãn")
    axes[1].set_xlabel("")
    axes[1].legend()
    luu_hinh(fig, "2_phan_bo_nhan_va_glucose.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[COT_SO + [COT_NHAN]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Ma trận tương quan")
    luu_hinh(fig, "3_tuong_quan.png")
    return df


# ============================================================
# BƯỚC 3.2. BIẾN ĐỔI DỮ LIỆU (MÃ HÓA + CHUẨN HÓA)
# ============================================================

def tao_bo_tien_xu_ly(cac_cot):
    """Tạo bộ tiền xử lý cho danh sách cột đầu vào.

    Mỗi nhóm cột có một Pipeline riêng; tất cả chỉ học (fit) trên tập Train.
    """
    cot_log = [c for c in COT_LOG if c in cac_cot]
    cot_hba1c = [c for c in ["HbA1C"] if c in cac_cot]
    cot_so = [c for c in COT_SO if c in cac_cot and c not in cot_log + cot_hba1c]
    cot_phan_loai = [c for c in COT_PHAN_LOAI if c in cac_cot]

    cac_nhom = [
        # Chỉ số lệch phải: điền thiếu -> log(1 + x) -> chuẩn hóa
        ("so_lech_phai", Pipeline([
            ("dien_thieu", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("chuan_hoa", StandardScaler()),
        ]), cot_log),
        # HbA1C: thêm cột đánh dấu "có bị thiếu không" vì phần lớn chưa xét nghiệm
        ("hba1c", Pipeline([
            ("dien_thieu", SimpleImputer(strategy="median", add_indicator=True)),
            ("chuan_hoa", StandardScaler()),
        ]), cot_hba1c),
        # Các chỉ số còn lại: điền thiếu -> chuẩn hóa (trung bình 0, độ lệch chuẩn 1)
        ("so", Pipeline([
            ("dien_thieu", SimpleImputer(strategy="median")),
            ("chuan_hoa", StandardScaler()),
        ]), cot_so),
        # Giới tính: điền thiếu -> mã hóa One-Hot (Nam/Nữ -> 1 cột 0/1)
        ("phan_loai", Pipeline([
            ("dien_thieu", SimpleImputer(strategy="most_frequent")),
            ("ma_hoa", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), cot_phan_loai),
    ]
    return ColumnTransformer([nhom for nhom in cac_nhom if nhom[2]])


def buoc_3_2_bien_doi():
    tieu_de("BƯỚC 3.2. BIẾN ĐỔI DỮ LIỆU")
    print("- Mã hóa nhãn       : 'Có nguy cơ' = 1, 'Không có nguy cơ' = 0 (đã làm ở bước đồng bộ)")
    print("- Mã hóa GioiTinh   : One-Hot Encoding (Nam/Nữ -> 1 cột 0/1)")
    print(f"- Biến đổi log      : {COT_LOG}")
    print("- HbA1C             : thêm cột đánh dấu 'chưa có kết quả xét nghiệm'")
    print("- Chuẩn hóa         : StandardScaler (đưa các chỉ số về cùng thang đo,")
    print("                      cần cho Logistic Regression và SVM vì đơn vị rất khác nhau,")
    print("                      ví dụ Glucose ~ 5 mmol/L nhưng AcidUric ~ 400 µmol/L)")


# ============================================================
# BƯỚC 3.3. PHÂN CHIA DỮ LIỆU
# ============================================================

def buoc_3_3_chia_du_lieu(df):
    tieu_de("BƯỚC 3.3. PHÂN CHIA DỮ LIỆU (70% Train - 15% Validation - 15% Test)")
    X = df[COT_SO + COT_PHAN_LOAI]
    y = df[COT_NHAN]

    # stratify giữ nguyên tỷ lệ nhãn ở cả 3 tập
    X_train, X_tam, y_train, y_tam = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tam, y_tam, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tam
    )
    for ten, y_tap in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        print(f"  {ten:10s}: {len(y_tap):5d} dòng ({len(y_tap) / len(y):.0%}) - tỷ lệ có nguy cơ {y_tap.mean():.1%}")
    print("Train dùng để học, Validation để so sánh và chọn mô hình, Test để đánh giá cuối cùng.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# BƯỚC 4 + 5. CHỌN VÀ HUẤN LUYỆN MÔ HÌNH
# ============================================================

def tao_cac_mo_hinh():
    return {
        "Mốc so sánh (đoán lớp đa số)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        # SVC không tự cho ra xác suất; bọc trong CalibratedClassifierCV để có xác suất
        # (cách thay cho SVC(probability=True) đã lỗi thời từ scikit-learn 1.9)
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False),
    }


def buoc_4_5_huan_luyen(X_train, y_train):
    tieu_de("BƯỚC 4 + 5. CHỌN VÀ HUẤN LUYỆN MÔ HÌNH")
    print("- Logistic Regression: mô hình tuyến tính, dễ giải thích, cho ra xác suất")
    print("- Random Forest      : nhiều cây quyết định bỏ phiếu, học được ngưỡng phi tuyến")
    print("- SVM (kernel RBF)   : tìm đường biên phân tách tối ưu, mạnh với dữ liệu đã chuẩn hóa")
    print("- Mốc so sánh        : luôn đoán lớp đa số; mô hình tốt phải vượt xa mốc này")

    cac_pipeline = {}
    for ten, mo_hinh in tao_cac_mo_hinh().items():
        pipeline = Pipeline([
            ("tien_xu_ly", tao_bo_tien_xu_ly(X_train.columns)),
            ("mo_hinh", mo_hinh),
        ])
        pipeline.fit(X_train, y_train)
        cac_pipeline[ten] = pipeline
        print(f"  Đã huấn luyện: {ten}")
    return cac_pipeline


# ============================================================
# BƯỚC 6. ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def tinh_chi_so(pipeline, X, y):
    du_doan = pipeline.predict(X)
    xac_suat = pipeline.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, du_doan),
        "Precision": precision_score(y, du_doan, zero_division=0),
        "Recall": recall_score(y, du_doan, zero_division=0),
        "F1-score": f1_score(y, du_doan, zero_division=0),
        "ROC-AUC": roc_auc_score(y, xac_suat),
    }


def buoc_6_danh_gia(cac_pipeline, X_train, X_val, X_test, y_train, y_val, y_test):
    tieu_de("BƯỚC 6. ĐÁNH GIÁ MÔ HÌNH")

    print("6.1. So sánh trên tập VALIDATION (dùng để chọn mô hình):")
    bang_val = pd.DataFrame({ten: tinh_chi_so(p, X_val, y_val) for ten, p in cac_pipeline.items()}).T
    print(bang_val.round(4).to_string())

    # Chọn mô hình theo F1-score (cân bằng Precision và Recall), hòa thì xét Recall
    ung_vien = bang_val.drop(index="Mốc so sánh (đoán lớp đa số)")
    ten_tot_nhat = ung_vien.sort_values(["F1-score", "Recall", "ROC-AUC"], ascending=False).index[0]
    mo_hinh_tot_nhat = cac_pipeline[ten_tot_nhat]
    print(f"\n=> Chọn mô hình: {ten_tot_nhat} (F1-score cao nhất trên Validation)")

    fig, ax = plt.subplots(figsize=(11, 5))
    bang_val.plot(kind="bar", ax=ax, rot=0, colormap="Set2")
    ax.set_ylim(0, 1.05)
    ax.set_title("So sánh các mô hình trên tập Validation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    luu_hinh(fig, "4_so_sanh_mo_hinh.png")

    print("\n6.2. Kết quả trên tập TEST (dữ liệu mô hình chưa từng thấy):")
    bang_test = pd.DataFrame({ten: tinh_chi_so(p, X_test, y_test) for ten, p in cac_pipeline.items()}).T
    print(bang_test.round(4).to_string())

    du_doan_test = mo_hinh_tot_nhat.predict(X_test)
    print(f"\nBáo cáo chi tiết của {ten_tot_nhat} trên tập Test:")
    print(classification_report(y_test, du_doan_test, target_names=TEN_NHAN, digits=4))

    cm = confusion_matrix(y_test, du_doan_test)
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print(pd.DataFrame(cm, index=TEN_NHAN, columns=TEN_NHAN).to_string())
    print(f"  - Bỏ sót người có nguy cơ (False Negative): {cm[1, 0]}")
    print(f"  - Báo động nhầm (False Positive)           : {cm[0, 1]}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TEN_NHAN, yticklabels=TEN_NHAN, ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn - {ten_tot_nhat} (tập Test)")
    luu_hinh(fig, "5_ma_tran_nham_lan.png")

    print("\n6.3. Mức độ quan trọng của từng chỉ số (Permutation Importance trên Validation):")
    print("     Xáo trộn ngẫu nhiên một cột rồi đo F1-score giảm bao nhiêu; giảm càng nhiều")
    print("     thì mô hình càng dựa vào chỉ số đó.")
    quan_trong = permutation_importance(
        mo_hinh_tot_nhat, X_val, y_val, scoring="f1", n_repeats=10, random_state=RANDOM_STATE
    )
    bang_quan_trong = pd.Series(quan_trong.importances_mean, index=X_val.columns).sort_values(ascending=False)
    print(bang_quan_trong.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    bang_quan_trong.sort_values().plot(kind="barh", ax=ax, color="#5dade2")
    ax.set_xlabel("Mức giảm F1-score khi xáo trộn cột")
    ax.set_title(f"Mức độ quan trọng của các chỉ số - {ten_tot_nhat}")
    luu_hinh(fig, "6_muc_do_quan_trong.png")

    print("\n6.4. Thí nghiệm bổ sung: bỏ Glucose và HbA1C, chỉ dùng các chỉ số còn lại")
    cot_con_lai = [c for c in X_train.columns if c not in ("Glucose", "HbA1C")]
    pipeline_phu = Pipeline([
        ("tien_xu_ly", tao_bo_tien_xu_ly(cot_con_lai)),
        ("mo_hinh", tao_cac_mo_hinh()[ten_tot_nhat]),
    ]).fit(X_train[cot_con_lai], y_train)
    chi_so_phu = tinh_chi_so(pipeline_phu, X_test[cot_con_lai], y_test)
    print("     " + ", ".join(f"{k} = {v:.4f}" for k, v in chi_so_phu.items()))
    print("     => Không có Glucose/HbA1C thì mô hình gần như chỉ ngang mốc đoán bừa.")
    print("        Nhãn trong dữ liệu được xác định theo tiêu chuẩn chẩn đoán y khoa")
    print("        (Glucose lúc đói >= 7.0 mmol/L hoặc HbA1C >= 6.5%), nên mô hình chủ yếu")
    print("        học lại tiêu chuẩn này; các chỉ số mỡ máu, gan, thận chỉ hỗ trợ thêm.")

    return ten_tot_nhat, mo_hinh_tot_nhat, bang_val, bang_test, bang_quan_trong


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    buoc_1_xac_dinh_bai_toan()
    df = buoc_2_doc_du_lieu()
    df = buoc_3_1_lam_sach(df)
    buoc_3_2_bien_doi()
    X_train, X_val, X_test, y_train, y_val, y_test = buoc_3_3_chia_du_lieu(df)
    cac_pipeline = buoc_4_5_huan_luyen(X_train, y_train)
    ten_tot_nhat, mo_hinh_tot_nhat, bang_val, bang_test, bang_quan_trong = buoc_6_danh_gia(
        cac_pipeline, X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Lưu mô hình cùng các thông tin mà giao diện cần hiển thị
    goi_mo_hinh = {
        "mo_hinh": mo_hinh_tot_nhat,
        "ten_mo_hinh": ten_tot_nhat,
        "cac_cot": list(X_train.columns),
        "chi_so_test": bang_test.loc[ten_tot_nhat].to_dict(),
        "moc_lop_da_so": float(y_test.value_counts(normalize=True).max()),
        "kich_thuoc": {"Train": len(y_train), "Validation": len(y_val), "Test": len(y_test)},
        "muc_do_quan_trong": bang_quan_trong.to_dict(),
    }
    joblib.dump(goi_mo_hinh, MODEL_PATH)
    tieu_de("HOÀN THÀNH")
    print(f"Đã lưu mô hình: {MODEL_PATH.name}")
    print("Chạy giao diện (Bước 7): streamlit run App_DuDoanTieuDuong.py")
    print("Địa chỉ giao diện      : http://localhost:8501/NguyenVanHien")
    return goi_mo_hinh


if __name__ == "__main__":
    main()

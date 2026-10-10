# -*- coding: utf-8 -*-
"""
BƯỚC 7. GIAO DIỆN NHẬP THÔNG TIN NGƯỜI BỆNH - CẢNH BÁO VÀ GIẢI THÍCH KẾT QUẢ
Sinh viên: Nguyễn Văn Hiến - 26ACN020 - CNTT4

Chạy (trong Terminal, tại thư mục chứa file này):
    streamlit run App_DuDoanTieuDuong.py
Địa chỉ giao diện: http://localhost:8501/NguyenVanHien (cấu hình trong .streamlit/config.toml)

Nếu chưa có file mô hình (mo_hinh_tieu_duong.joblib), giao diện sẽ tự gọi
NguyenVanHien_KiemTra.py để huấn luyện ở lần chạy đầu tiên.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from NguyenVanHien_KiemTra import COT_SO, DON_VI, KHOANG_HOP_LE, MODEL_PATH
from NguyenVanHien_KiemTra import main as huan_luyen

# ============================================================
# CẤU HÌNH TRANG
# ============================================================

st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="centered")

# Tên hiển thị và bước nhảy của từng chỉ số
TEN_HIEN_THI = {
    "Glucose": "Glucose (đường huyết lúc đói)",
    "HbA1C": "HbA1C (đường huyết trung bình 3 tháng)",
    "Cholesterol": "Cholesterol toàn phần",
    "HDL": "HDL-Cholesterol (mỡ tốt)",
    "LDL": "LDL-Cholesterol (mỡ xấu)",
    "Triglycerid": "Triglycerid",
    "Creatinine": "Creatinine (chức năng thận)",
    "Ure": "Ure (chức năng thận)",
    "AcidUric": "Acid Uric",
    "AST": "AST / GOT (men gan)",
    "ALT": "ALT / GPT (men gan)",
}
BUOC_NHAY = {
    "Glucose": 0.1, "HbA1C": 0.1, "Cholesterol": 0.1, "HDL": 0.1, "LDL": 0.1,
    "Triglycerid": 0.1, "Creatinine": 1.0, "Ure": 0.1, "AcidUric": 1.0, "AST": 1.0, "ALT": 1.0,
}
NHOM_CHI_SO = {
    "🩸 Đường huyết": ["Glucose", "HbA1C"],
    "🫀 Mỡ máu": ["Cholesterol", "HDL", "LDL", "Triglycerid"],
    "🫘 Chức năng thận": ["Creatinine", "Ure", "AcidUric"],
    "🧪 Chức năng gan": ["AST", "ALT"],
}
CHI_SO_XET_NGHIEM = [cot for nhom in NHOM_CHI_SO.values() for cot in nhom]

# Khoảng tham chiếu bình thường ở người trưởng thành: (thấp nhất, cao nhất).
# None = không xét chiều đó. Một số chỉ số khác nhau giữa nam và nữ.
KHOANG_BINH_THUONG = {
    "Glucose": (3.9, 5.6),
    "HbA1C": (None, 5.7),
    "Cholesterol": (None, 5.2),
    "HDL": {"Nam": (1.0, None), "Nữ": (1.3, None)},
    "LDL": (None, 3.4),
    "Triglycerid": (None, 1.7),
    "Creatinine": {"Nam": (62, 106), "Nữ": (44, 80)},
    "Ure": (2.5, 7.5),
    "AcidUric": {"Nam": (210, 420), "Nữ": (150, 360)},
    "AST": (None, 40),
    "ALT": (None, 40),
}

# Giá trị điển hình của người khỏe mạnh, dùng để hỏi mô hình:
# "nếu chỉ số này ở mức bình thường thì xác suất thay đổi thế nào?"
GIA_TRI_BINH_THUONG = {
    "Glucose": 5.0, "HbA1C": 5.3, "Cholesterol": 4.5,
    "HDL": {"Nam": 1.3, "Nữ": 1.5}, "LDL": 2.6, "Triglycerid": 1.2,
    "Creatinine": {"Nam": 85, "Nữ": 65}, "Ure": 5.0,
    "AcidUric": {"Nam": 320, "Nữ": 260}, "AST": 25, "ALT": 25,
}

# Dữ liệu mẫu để thử nhanh giao diện
MAU_KHOE_MANH = {
    "Tuoi": 35, "GioiTinh": "Nam", "Glucose": 5.1, "HbA1C": None, "Cholesterol": 4.8,
    "HDL": 1.4, "LDL": 2.9, "Triglycerid": 1.3, "Creatinine": 80.0, "Ure": 5.0,
    "AcidUric": 320.0, "AST": 22.0, "ALT": 20.0,
}
MAU_NGUY_CO_CAO = {
    "Tuoi": 62, "GioiTinh": "Nữ", "Glucose": 11.9, "HbA1C": 8.1, "Cholesterol": 6.2,
    "HDL": 1.0, "LDL": 3.9, "Triglycerid": 2.8, "Creatinine": 95.0, "Ure": 7.7,
    "AcidUric": 400.0, "AST": 45.0, "ALT": 40.0,
}
MAU_TRONG = {"Tuoi": 45, "GioiTinh": "Nam", **{cot: None for cot in CHI_SO_XET_NGHIEM}}

NGUONG_CAO = 0.70
NGUONG_THAP = 0.30


def theo_gioi_tinh(gia_tri, gioi_tinh):
    """Lấy giá trị theo giới tính nếu chỉ số đó phân biệt nam/nữ."""
    return gia_tri[gioi_tinh] if isinstance(gia_tri, dict) else gia_tri


def danh_gia_chi_so(cot, gia_tri, gioi_tinh):
    """Đối chiếu một chỉ số với ngưỡng y khoa, trả về (mức độ, nhận xét)."""
    if cot == "Glucose":
        if gia_tri >= 7.0:
            return "🔴", "Đạt ngưỡng chẩn đoán tiểu đường (≥ 7.0 mmol/L)"
        if gia_tri >= 5.6:
            return "🟡", "Mức tiền tiểu đường (5.6 – 6.9 mmol/L)"
        if gia_tri < 3.9:
            return "🟡", "Thấp – hạ đường huyết (< 3.9 mmol/L)"
        return "🟢", "Bình thường (3.9 – 5.5 mmol/L)"
    if cot == "HbA1C":
        if gia_tri >= 6.5:
            return "🔴", "Đạt ngưỡng chẩn đoán tiểu đường (≥ 6.5%)"
        if gia_tri >= 5.7:
            return "🟡", "Mức tiền tiểu đường (5.7 – 6.4%)"
        return "🟢", "Bình thường (< 5.7%)"

    thap, cao = theo_gioi_tinh(KHOANG_BINH_THUONG[cot], gioi_tinh)
    don_vi = DON_VI[cot]
    if thap is not None and gia_tri < thap:
        return "🟡", f"Thấp hơn bình thường (< {thap:g} {don_vi})"
    if cao is not None and gia_tri > cao:
        return "🟡", f"Cao hơn bình thường (> {cao:g} {don_vi})"
    if thap is None:
        return "🟢", f"Bình thường (≤ {cao:g} {don_vi})"
    if cao is None:
        return "🟢", f"Bình thường (≥ {thap:g} {don_vi})"
    return "🟢", f"Bình thường ({thap:g} – {cao:g} {don_vi})"


def tao_dataframe(thong_tin, cac_cot):
    """Chuyển thông tin nhập vào thành DataFrame đúng cột mô hình cần; ô trống -> NaN."""
    X = pd.DataFrame([thong_tin], columns=cac_cot)
    X[COT_SO] = X[COT_SO].astype(float)
    return X


def giai_thich_mo_hinh(mo_hinh, thong_tin, cac_cot, xac_suat_goc):
    """Với mỗi chỉ số đã nhập: thay bằng giá trị bình thường rồi dự đoán lại.

    Xác suất giảm nhiều khi đưa chỉ số về bình thường => chỉ số đó đang làm tăng nguy cơ.
    """
    gioi_tinh = thong_tin["GioiTinh"]
    cac_dong = []
    for cot in CHI_SO_XET_NGHIEM:
        if thong_tin[cot] is None:
            continue
        gia_tri_tot = theo_gioi_tinh(GIA_TRI_BINH_THUONG[cot], gioi_tinh)
        thong_tin_moi = {**thong_tin, cot: gia_tri_tot}
        xac_suat_moi = mo_hinh.predict_proba(tao_dataframe(thong_tin_moi, cac_cot))[0, 1]
        cac_dong.append({
            "Chỉ số": cot,
            "Giá trị nhập": f"{thong_tin[cot]:g} {DON_VI[cot]}",
            "Nếu ở mức bình thường": f"{gia_tri_tot:g} {DON_VI[cot]}",
            "Ảnh hưởng (điểm %)": (xac_suat_goc - xac_suat_moi) * 100,
        })
    bang = pd.DataFrame(cac_dong)
    if not bang.empty:
        bang = bang.reindex(bang["Ảnh hưởng (điểm %)"].abs().sort_values(ascending=False).index)
    return bang


# ============================================================
# TẢI MÔ HÌNH (huấn luyện ở lần đầu nếu chưa có)
# ============================================================

@st.cache_resource(show_spinner="Đang tải mô hình (lần đầu sẽ huấn luyện, mất khoảng 1 phút)...")
def tai_mo_hinh():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return huan_luyen()


try:
    goi = tai_mo_hinh()
except FileNotFoundError as loi:
    st.error(f"Không tìm thấy file dữ liệu: {loi.filename}. Hãy đặt CSDL_TieuDuong.csv cùng thư mục với chương trình.")
    st.stop()

mo_hinh = goi["mo_hinh"]
cac_cot = goi["cac_cot"]

# ============================================================
# THANH BÊN: THÔNG TIN MÔ HÌNH
# ============================================================

with st.sidebar:
    st.header("Thông tin mô hình")
    st.write(f"**Mô hình được chọn:** {goi['ten_mo_hinh']}")
    st.write("**Dữ liệu:** " + ", ".join(f"{k} {v}" for k, v in goi["kich_thuoc"].items()))
    st.write("**Kết quả trên tập Test:**")
    st.dataframe(
        pd.Series(goi["chi_so_test"], name="Giá trị").map(lambda v: f"{v:.2%}"),
        width="stretch",
    )
    st.caption(
        f"Mốc so sánh (luôn đoán 'Không có nguy cơ'): Accuracy {goi['moc_lop_da_so']:.1%}. "
        "Kết quả rất cao vì nhãn trong dữ liệu được xác định theo tiêu chuẩn chẩn đoán "
        "(Glucose ≥ 7.0 mmol/L hoặc HbA1C ≥ 6.5%), nên mô hình chủ yếu dựa vào hai chỉ số này."
    )

# ============================================================
# NHẬP THÔNG TIN NGƯỜI BỆNH
# ============================================================

st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.write(
    "Nhập kết quả xét nghiệm của người bệnh. Chỉ số nào **chưa xét nghiệm thì để trống**, "
    "mô hình sẽ tự xử lý như dữ liệu thiếu."
)

for khoa, gia_tri in MAU_TRONG.items():
    st.session_state.setdefault(khoa, gia_tri)


def dien_mau(mau):
    for khoa, gia_tri in mau.items():
        st.session_state[khoa] = gia_tri


st.write("Thử nhanh với dữ liệu mẫu:")
cot_mau = st.columns(3)
cot_mau[0].button("🟢 Người khỏe mạnh", on_click=dien_mau, args=(MAU_KHOE_MANH,), width="stretch")
cot_mau[1].button("🔴 Nguy cơ cao", on_click=dien_mau, args=(MAU_NGUY_CO_CAO,), width="stretch")
cot_mau[2].button("🧹 Xóa trắng", on_click=dien_mau, args=(MAU_TRONG,), width="stretch")

with st.form("form_nguoi_benh"):
    st.subheader("👤 Thông tin chung")
    cot_trai, cot_phai = st.columns(2)
    tuoi_min, tuoi_max = KHOANG_HOP_LE["Tuoi"]
    cot_trai.number_input("Tuổi", min_value=tuoi_min, max_value=tuoi_max, step=1, key="Tuoi")
    cot_phai.selectbox("Giới tính", ["Nam", "Nữ"], key="GioiTinh")

    for ten_nhom, cac_chi_so in NHOM_CHI_SO.items():
        st.subheader(ten_nhom)
        cac_o = st.columns(2)
        for vi_tri, cot in enumerate(cac_chi_so):
            nho_nhat, lon_nhat = KHOANG_HOP_LE[cot]
            cac_o[vi_tri % 2].number_input(
                f"{TEN_HIEN_THI[cot]} ({DON_VI[cot]})",
                min_value=float(nho_nhat),
                max_value=float(lon_nhat),
                step=BUOC_NHAY[cot],
                format="%.2f" if BUOC_NHAY[cot] < 1 else "%.0f",
                placeholder="Chưa xét nghiệm",
                key=cot,
            )

    du_doan = st.form_submit_button("🔍 Dự đoán nguy cơ", type="primary", width="stretch")

# ============================================================
# DỰ ĐOÁN, CẢNH BÁO VÀ GIẢI THÍCH
# ============================================================

if du_doan:
    thong_tin = {cot: st.session_state[cot] for cot in cac_cot}
    gioi_tinh = thong_tin["GioiTinh"]
    xac_suat = mo_hinh.predict_proba(tao_dataframe(thong_tin, cac_cot))[0, 1]

    st.divider()
    st.header("Kết quả dự đoán")

    if thong_tin["Glucose"] is None and thong_tin["HbA1C"] is None:
        st.warning(
            "⚠️ Chưa có Glucose và HbA1C – hai chỉ số quan trọng nhất để đánh giá tiểu đường. "
            "Kết quả dưới đây có độ tin cậy thấp, nên làm thêm hai xét nghiệm này."
        )

    if xac_suat >= NGUONG_CAO:
        st.error(f"🔴 **NGUY CƠ CAO** mắc bệnh tiểu đường (xác suất {xac_suat:.1%})")
    elif xac_suat >= NGUONG_THAP:
        st.warning(f"🟡 **NGUY CƠ TRUNG BÌNH** – cần theo dõi thêm (xác suất {xac_suat:.1%})")
    else:
        st.success(f"🟢 **NGUY CƠ THẤP** (xác suất {xac_suat:.1%})")

    # Cảnh báo an toàn theo tiêu chuẩn chẩn đoán, độc lập với mô hình: dữ liệu huấn luyện
    # có rất ít người có kết quả HbA1C thật nên mô hình có thể chưa chắc chắn ở các ca này.
    vuot_nguong = []
    if thong_tin["Glucose"] is not None and thong_tin["Glucose"] >= 7.0:
        vuot_nguong.append(f"Glucose lúc đói = {thong_tin['Glucose']:g} mmol/L (≥ 7.0)")
    if thong_tin["HbA1C"] is not None and thong_tin["HbA1C"] >= 6.5:
        vuot_nguong.append(f"HbA1C = {thong_tin['HbA1C']:g}% (≥ 6.5)")
    if vuot_nguong and xac_suat < NGUONG_CAO:
        st.error(
            "🚨 **Cảnh báo theo tiêu chuẩn y khoa:** " + "; ".join(vuot_nguong) + " đã đạt ngưỡng "
            "chẩn đoán tiểu đường. Dù mô hình cho xác suất chưa cao, người bệnh vẫn cần đi khám để xác nhận."
        )

    cot_1, cot_2 = st.columns(2)
    cot_1.metric("Xác suất có nguy cơ", f"{xac_suat:.1%}")
    cot_2.metric("Xác suất không có nguy cơ", f"{1 - xac_suat:.1%}")
    st.progress(float(xac_suat))
    st.caption(
        f"Mức cảnh báo: thấp < {NGUONG_THAP:.0%} ≤ trung bình < {NGUONG_CAO:.0%} ≤ cao. "
        f"Mô hình sử dụng: {goi['ten_mo_hinh']}."
    )

    # --- Giải thích 1: chỉ số nào làm thay đổi dự đoán của mô hình ---
    st.subheader("🔎 Vì sao mô hình đưa ra kết quả này?")
    st.write(
        "Với từng chỉ số đã nhập, ta thay nó bằng giá trị của người khỏe mạnh rồi cho mô hình "
        "dự đoán lại. Ảnh hưởng **dương** nghĩa là chỉ số đó đang **làm tăng** nguy cơ."
    )
    bang_anh_huong = giai_thich_mo_hinh(mo_hinh, thong_tin, cac_cot, xac_suat)
    if bang_anh_huong.empty:
        st.info("Chưa nhập chỉ số xét nghiệm nào nên không thể giải thích theo từng chỉ số.")
    else:
        dang_ke = bang_anh_huong[bang_anh_huong["Ảnh hưởng (điểm %)"].abs() >= 1]
        tang = dang_ke[dang_ke["Ảnh hưởng (điểm %)"] > 0]
        if not tang.empty:
            mo_ta = ", ".join(
                f"**{dong['Chỉ số']}** ({dong['Giá trị nhập']}, +{dong['Ảnh hưởng (điểm %)']:.0f} điểm %)"
                for _, dong in tang.iterrows()
            )
            st.write(f"Yếu tố làm tăng nguy cơ: {mo_ta}.")
        elif xac_suat < NGUONG_THAP:
            st.write("Không có chỉ số nào làm tăng nguy cơ đáng kể.")
        else:
            st.write("Không có một chỉ số riêng lẻ nào quyết định kết quả; nguy cơ đến từ tổng hợp nhiều chỉ số.")

        st.dataframe(
            bang_anh_huong.style.format({"Ảnh hưởng (điểm %)": "{:+.1f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Khi nhiều chỉ số cùng bất thường, ảnh hưởng của từng chỉ số riêng lẻ có thể nhỏ hơn thực tế, "
            "vì các chỉ số còn lại vẫn báo nguy cơ."
        )

    # --- Giải thích 2: đối chiếu với ngưỡng y khoa ---
    st.subheader("📋 Đối chiếu với ngưỡng tham chiếu y khoa")
    cac_dong = []
    for cot in CHI_SO_XET_NGHIEM:
        if thong_tin[cot] is None:
            cac_dong.append({"": "⚪", "Chỉ số": TEN_HIEN_THI[cot], "Giá trị": "—", "Nhận xét": "Chưa xét nghiệm"})
            continue
        muc, nhan_xet = danh_gia_chi_so(cot, thong_tin[cot], gioi_tinh)
        cac_dong.append({
            "": muc,
            "Chỉ số": TEN_HIEN_THI[cot],
            "Giá trị": f"{thong_tin[cot]:g} {DON_VI[cot]}",
            "Nhận xét": nhan_xet,
        })
    st.dataframe(pd.DataFrame(cac_dong), width="stretch", hide_index=True)
    st.caption("Ngưỡng Glucose áp dụng cho mẫu máu lấy lúc đói (nhịn ăn ít nhất 8 giờ).")

    # --- Khuyến nghị ---
    st.subheader("💡 Khuyến nghị")
    if xac_suat >= NGUONG_CAO:
        st.write(
            "- Nên đến khám chuyên khoa **Nội tiết – Đái tháo đường** sớm.\n"
            "- Làm lại xét nghiệm Glucose lúc đói và HbA1C để khẳng định chẩn đoán.\n"
            "- Hạn chế đồ ngọt, tinh bột; tăng vận động; theo dõi cân nặng và huyết áp."
        )
    elif xac_suat >= NGUONG_THAP:
        st.write(
            "- Nên tái khám và xét nghiệm lại sau 3 – 6 tháng.\n"
            "- Điều chỉnh chế độ ăn, tập thể dục ít nhất 150 phút/tuần."
        )
    else:
        st.write(
            "- Duy trì lối sống lành mạnh.\n"
            "- Kiểm tra sức khỏe định kỳ mỗi năm, đặc biệt nếu trên 45 tuổi hoặc thừa cân."
        )

    st.caption(
        "⚠️ Kết quả chỉ mang tính tham khảo, được tạo bởi mô hình học máy cho mục đích học tập; "
        "không thay thế chẩn đoán của bác sĩ."
    )

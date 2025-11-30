import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import zipfile
import pandas as pd

# ------------------------------------
# إعداد صفحة التطبيق
# ------------------------------------
st.set_page_config(page_title="Green Area Detection", layout="wide")
st.title("🌱 تطبيق الكشف عن المناطق الخضراء باستخدام U-Net")

# ------------------------------------
# تحميل الموديل
# ------------------------------------
@st.cache_resource
def load_unet():
    model = tf.keras.models.load_model("simple_unet_model.h5", compile=False)
    return model

model = load_unet()

# ------------------------------------
# المعالجة المسبقة للصورة (بدون OpenCV)
# ------------------------------------
def preprocess_image(upload):
    img = Image.open(upload).convert("RGB")
    img = img.resize((256, 256))
    array = np.array(img) / 255.0
    return array, img

# ------------------------------------
# التنبؤ بالماسك
# ------------------------------------
def predict_mask(model, img_array):
    inp = np.expand_dims(img_array, axis=0)
    pred = model.predict(inp)[0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    return mask

# ------------------------------------
# الواجهة
# ------------------------------------
uploaded = st.file_uploader("📤 ارفعي صورة", type=["jpg", "png", "jpeg"])

if uploaded:
    st.subheader("الصورة الأصلية")
    arr, original_img = preprocess_image(uploaded)
    st.image(original_img, width=400)

    st.subheader("نتيجة الموديل")
    mask = predict_mask(model, arr)
    st.image(mask, width=400, caption="الماسك المتوقع")

    # تحميل الماسك
    mask_pil = Image.fromarray(mask.squeeze())
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    st.download_button("📥 تحميل الماسك", buf.getvalue(), "mask.png")

# ------------------------------
# التنبؤ بالصور داخل ملف ZIP
# ------------------------------
st.subheader("📦 التنبؤ لمجموعة صور (ZIP)")

zip_file = st.file_uploader("ارفعي ملف ZIP", type="zip")

if zip_file:
    with zipfile.ZipFile(zip_file, "r") as z:
        file_list = z.namelist()
        results = []

        for file_name in file_list:
            if file_name.lower().endswith((".jpg",".png",".jpeg")):
                img_data = z.read(file_name)
                arr, _ = preprocess_image(BytesIO(img_data))
                mask = predict_mask(model, arr)

                # تجهيز الماسك لحفظه في ZIP
                mask_pil = Image.fromarray(mask.squeeze())
                buf = BytesIO()
                mask_pil.save(buf, format="PNG")

                results.append((file_name, buf.getvalue()))

        # إنشاء ZIP للمخرجات
        out_zip = BytesIO()
        with zipfile.ZipFile(out_zip, "w") as z_out:
            for name, content in results:
                z_out.writestr(name.replace(".", "_mask."), content)

        st.download_button("📥 تحميل ملف المخرجات",
                           data=out_zip.getvalue(),
                           file_name="predicted_masks.zip")

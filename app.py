import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io

# -----------------------------

# Load Model Architecture

# -----------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True), # Corrected typo from nn.ReRLou to nn.ReLU
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(32, 64)

        self.up2   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)

        self.up1   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        b = self.bridge(p2)

        u2 = self.up2(b)
        m2 = torch.cat([u2, c2], dim=1)
        c4 = self.conv2(m2)

        u1 = self.up1(c4)
        m1 = torch.cat([u1, c1], dim=1)
        c5 = self.conv1(m1)

        out = torch.sigmoid(self.final(c5))
        return out

# -----------------------------

# Utility functions

# -----------------------------

IMAGE_SIZE = (128, 128)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm

def compute_gvi(img, pred_mask_bin):
    veg_pixels = img[pred_mask_bin[:,:,0] == 1]
    if veg_pixels.size == 0:
        return 0.0
    return float(veg_pixels[:, 1].mean())

def irrigation_advice(prop_agri, gvi):
    if prop_agri < 0.05:
        return "🟤 المنطقة معظمها صحراء."
    if prop_agri > 0.70:
        return "🟢 الغطاء النباتي كثيف."
    if gvi > 0.55:
        return "🟢 النبات صحي – قللي الري."
    elif gvi > 0.35:
        return "🟡 حالة متوسطة – ري معتدل."
    else:
        return "🔴 النبات مجهد – زوّدي الري."

# -----------------------------

# Load Model Weights

# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)

model.load_state_dict(torch.load("desert_agri_unet_weights.pth", map_location=device))
model.eval()

# -----------------------------

# Streamlit UI

# -----------------------------

st.title("🌱 Green Area Detection")
uploaded = st.file_uploader("Upload aerial image", type=["jpg","jpeg","png"])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Original Image", use_column_width=True)

    # Preprocess
    img_prep = preprocess_image(img)
    img_t = torch.tensor(img_prep.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)[0].cpu().numpy()

    pred = np.transpose(pred, (1,2,0))
    pred_mask_bin = (pred > 0.5).astype(np.uint8)

    # Compute vegetation ratio
    prop_agri = np.sum(pred_mask_bin) / (pred_mask_bin.shape[0] * pred_mask_bin.shape[1])
    gvi = compute_gvi(img_prep, pred_mask_bin)
    advice = irrigation_advice(prop_agri, gvi)

    # Create overlay
    overlay = (img_prep*255).astype(np.uint8).copy()
    overlay[pred_mask_bin[:,:,0] == 1] = (
        overlay[pred_mask_bin[:,:,0] == 1] * 0.4 +
        np.array([0,255,0]) * 0.6
    ).astype(np.uint8)

    st.subheader("🟩 Vegetation Mask")
    st.image(pred_mask_bin[:,:,0]*255, caption="Predicted Mask", use_column_width=True)

    st.subheader("🌿 Overlay Output")
    st.image(overlay, caption="Overlay", use_column_width=True)

    st.subheader("📊 Results")
    st.write(f"**Green Area Ratio:** {prop_agri*100:.2f}%")
    st.write(f"**GVI (Vegetation Index):** {gvi:.3f}")
    st.write(f"**Advice:** {advice}")

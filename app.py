# app.py
# Streamlit web app for real-time emotion detection

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .confidence-text {
        font-size: 0.85rem;
        color: #aaaaaa;
    }
    .metric-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888888;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── EMOTION COLORS (hex for Streamlit) ──────────────────────
EMOTION_COLORS = {
    'angry'   : '#ff4444',
    'disgust' : '#ff8c00',
    'fear'    : '#9b59b6',
    'happy'   : '#00d4aa',
    'neutral' : '#3498db',
    'sad'     : '#5b9bd5',
    'surprise': '#f1c40f',
}

EMOTION_EMOJIS = {
    'angry'   : '😠',
    'disgust' : '🤢',
    'fear'    : '😨',
    'happy'   : '😄',
    'neutral' : '😐',
    'sad'     : '😢',
    'surprise': '😮',
}

# ── LOAD MODEL (cached so it only loads once) ────────────────
@st.cache_resource
def load_emotion_model():
    model = load_model('model/emotion_model.h5')
    classes = np.load('model/classes.npy', allow_pickle=True)
    classes = [str(c) for c in classes]
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml'
    )
    return model, classes, face_cascade

# ── PREDICT EMOTION FROM FRAME ───────────────────────────────
def predict_emotion(frame, model, classes, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    results = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        predictions = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(predictions[0])
        emotion_label = classes[emotion_index]
        confidence = float(predictions[0][emotion_index]) * 100
        all_probs = {
            classes[i]: float(predictions[0][i]) * 100
            for i in range(len(classes))
        }

        # Draw on frame
        color_bgr = {
            'angry'   : (0,   0,   255),
            'disgust' : (0,   140, 255),
            'fear'    : (128, 0,   128),
            'happy'   : (0,   212, 170),
            'neutral' : (255, 150, 0  ),
            'sad'     : (210, 150, 90 ),
            'surprise': (0,   220, 255),
        }
        c = color_bgr.get(emotion_label, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)

        label = f"{EMOTION_EMOJIS.get(emotion_label,'')} {emotion_label}: {confidence:.1f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-label_size[1]-15), (x+label_size[0]+10, y), c, -1)
        cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        results.append({
            'emotion': emotion_label,
            'confidence': confidence,
            'all_probs': all_probs,
            'bbox': (x, y, w, h)
        })

    return frame, results

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<p class="title">😊 Real-Time Emotion Detection</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN trained on FER-2013 · 65.55% accuracy · 7 emotions</p>',
            unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────
with st.spinner("Loading emotion detection model..."):
    model, classes, face_cascade = load_emotion_model()

st.success("Model loaded successfully!")

# ── LAYOUT: TWO COLUMNS ──────────────────────────────────────
col_cam, col_stats = st.columns([2, 1])

with col_cam:
    st.markdown("### 📹 Live Camera Feed")
    run = st.toggle("Start Camera", value=False)
    FRAME_WINDOW = st.image([])
    status_text = st.empty()

with col_stats:
    st.markdown("### 📊 Emotion Analysis")

    # Metric placeholders
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        detected_metric = st.empty()
    with metric_col2:
        confidence_metric = st.empty()

    st.markdown("#### Confidence Bars")
    bar_placeholders = {}
    for emotion in classes:
        bar_placeholders[emotion] = st.empty()

    st.markdown("---")
    st.markdown("#### About This Model")
    st.markdown("""
    - **Dataset:** FER-2013
    - **Training samples:** 28,709
    - **Test accuracy:** 65.55%
    - **Architecture:** CNN (3 conv blocks)
    - **Framework:** TensorFlow / Keras
    """)

# ── WEBCAM LOOP ──────────────────────────────────────────────
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Could not open webcam!")
    else:
        frame_count = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Process every 3rd frame for speed
            if frame_count % 3 == 0:
                processed_frame, results = predict_emotion(
                    frame.copy(), model, classes, face_cascade
                )
            else:
                processed_frame = frame.copy()
                results = []

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame, width=640)

            # Update stats panel
            if results:
                top = results[0]
                emotion = top['emotion']
                confidence = top['confidence']
                emoji = EMOTION_EMOJIS.get(emotion, '')

                # Update metrics
                detected_metric.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{emoji}</div>
                    <div class="metric-label">{emotion.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

                confidence_metric.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{confidence:.0f}%</div>
                    <div class="metric-label">CONFIDENCE</div>
                </div>
                """, unsafe_allow_html=True)

                status_text.success(f"Face detected: {emoji} {emotion} ({confidence:.1f}%)")

                # Update confidence bars
                for emotion_name, placeholder in bar_placeholders.items():
                    prob = top['all_probs'].get(emotion_name, 0)
                    color = EMOTION_COLORS.get(emotion_name, '#ffffff')
                    em_emoji = EMOTION_EMOJIS.get(emotion_name, '')
                    is_top = emotion_name == top['emotion']
                    border = f"border: 2px solid {color};" if is_top else ""

                    placeholder.markdown(f"""
                    <div class="emotion-card" style="{border}">
                        <div class="emotion-label">
                            {em_emoji} {emotion_name.capitalize()}
                            <span class="confidence-text" style="float:right">
                                {prob:.1f}%
                            </span>
                        </div>
                        <div style="background:#2d3250;border-radius:6px;height:10px;">
                            <div style="
                                width:{prob}%;
                                background:{color};
                                height:10px;
                                border-radius:6px;
                                transition: width 0.3s ease;
                                max-width:100%;
                            "></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                status_text.warning("No face detected — move closer to camera")
                detected_metric.markdown("""
                <div class="metric-box">
                    <div class="metric-value">—</div>
                    <div class="metric-label">DETECTED</div>
                </div>
                """, unsafe_allow_html=True)

                confidence_metric.markdown("""
                <div class="metric-box">
                    <div class="metric-value">—</div>
                    <div class="metric-label">CONFIDENCE</div>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()

else:
    # Show placeholder when camera is off
    FRAME_WINDOW.markdown("""
    <div style="
        background:#1e2130;
        border-radius:12px;
        height:400px;
        display:flex;
        align-items:center;
        justify-content:center;
        flex-direction:column;
        color:#888888;
    ">
        <div style="font-size:4rem">📷</div>
        <div style="font-size:1.2rem;margin-top:1rem">
            Toggle "Start Camera" to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show empty bars
    for emotion_name, placeholder in bar_placeholders.items():
        color = EMOTION_COLORS.get(emotion_name, '#ffffff')
        em_emoji = EMOTION_EMOJIS.get(emotion_name, '')
        placeholder.markdown(f"""
        <div class="emotion-card">
            <div class="emotion-label">
                {em_emoji} {emotion_name.capitalize()}
                <span class="confidence-text" style="float:right">0.0%</span>
            </div>
            <div style="background:#2d3250;border-radius:6px;height:10px;">
                <div style="width:0%;background:{color};height:10px;border-radius:6px;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555555;font-size:0.85rem">
    Built by Himanshu Nath Tripathi · VIT Bhopal · B.Tech CSE (24BSA10323) 2024-2028
    · GitHub: himanshunath007
</div>
""", unsafe_allow_html=True)
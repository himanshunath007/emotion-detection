import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('model/emotion_model.h5')
classes = np.load('model/classes.npy', allow_pickle=True)
classes = [str(c) for c in classes]
print(f"Model loaded! Emotions: {classes}")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

COLORS = {
    'angry'   : (0,   0,   255),
    'disgust' : (0,   140, 255),
    'fear'    : (128, 0,   128),
    'happy'   : (0,   255, 0  ),
    'neutral' : (255, 255, 0  ),
    'sad'     : (255, 0,   0  ),
    'surprise': (0,   255, 255),
}

print("Opening webcam...")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam opened! Starting detection...")
print("A window should appear - click on it then press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to read frame - retrying...")
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        predictions = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(predictions[0])
        emotion_label = classes[emotion_index]
        confidence = predictions[0][emotion_index] * 100

        color = COLORS.get(emotion_label, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label_text = f"{emotion_label}: {confidence:.1f}%"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 15), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        bar_x = 10
        bar_y = 10
        bar_h = 20
        bar_gap = 28
        max_width = 200

        for i, (emotion, prob) in enumerate(zip(classes, predictions[0])):
            bar_width = int(prob * max_width)
            bar_color = COLORS.get(emotion, (200, 200, 200))
            y_pos = bar_y + i * bar_gap

            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + max_width, y_pos + bar_h), (50, 50, 50), -1)

            if bar_width > 0:
                cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_h), bar_color, -1)

            bar_text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(frame, bar_text, (bar_x + max_width + 8, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected - move closer", (50, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, "Press Q to quit", (frame.shape[1] - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from transformers import pipeline
import time

# Load the trained model and labels
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    exit()

# Initialize Mediapipe holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
text_model = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion')

# Function to predict emotion from text
def predict_text_emotion(text):
    try:
        predictions = text_model(text)
        emotion_label = predictions[0]['label']
        confidence = predictions[0]['score']
        return emotion_label, confidence
    except Exception as e:
        print(f"Error during text emotion prediction: {e}")
        return "Error", 0.0

# Initialize webcam
cap = cv2.VideoCapture(0)
classifier = pipeline("sentiment-analysis")
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    lst = []
    _, frm = cap.read()
    if not _:
        break

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        # Predict the gesture
        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    # Draw landmarks on frame
    
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:  # Exit on ESC key
        cv2.destroyAllWindows()
        cap.release()
        break


# Text input emotion detection
text = input("Enter text for emotion detection: ")
text_emotion, text_confidence = predict_text_emotion(text)
print(f"Text Emotion: {text_emotion} ({text_confidence:.2f})")
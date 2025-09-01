from modules import *
from MP_holistic_styled_landmarks import mp_holistic, draw_styled_landmarks
from mediapipe_detection import mediapipe_detection
from keypoints_extraction import extract_keypoints
import keras
from folder_setup import *
from visualization import prob_viz, colors
import numpy as np
import cv2

# Define the actions your model was trained on
actions = np.array(['hello', 'thanks', 'iloveyou'])

sequence = []
sentence = []
threshold = 0.5   # lower threshold for easier debugging

# Use forward slashes for cross-platform compatibility
model = keras.models.load_model("../Model/lstm_model.h5")



cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mediapipe detection
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]   # keep last 30 frames

        # Debug: print sequence length
        print("Sequence length:", len(sequence))

        # Make prediction once we have 30 frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            # Debug: show raw probabilities and prediction
            print("Probabilities:", res)
            print("Prediction:", actions[np.argmax(res)], 
                  "Confidence:", res[np.argmax(res)])

            # If confidence is good enough, update sentence
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            # Keep last 5 predictions only
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualization
            image = prob_viz(res, actions, image, colors)

        # Display result bar with recognized sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 
                    cv2.LINE_AA)

        # Show window
        cv2.imshow('Action_Recognition', image)

        # Quit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

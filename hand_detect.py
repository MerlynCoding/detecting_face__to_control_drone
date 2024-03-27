import mediapipe as mp
from function import *

def detect_hand(frame,holistic):
    img, results = mediapipe_detection(frame, holistic)

    # Draw landmarks
    draw_styled_landmarks(img, results)

    # 2. Prediction logic
    keypoints = extract_keypoints(results)
        #sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
    sequence.append(keypoints)
    sequence = sequence[:30]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        b=actions[np.argmax(res)]
        return b,img
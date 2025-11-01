import cv2
import mediapipe as mp
import math

print("Gesture Recognition Starting...")
print("Press 'q' to quit")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
camera = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    """Calculate distance between two landmarks"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_finger_extended(landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is extended by comparing tip and middle joint positions"""
    tip = landmarks[finger_tip_id]
    pip = landmarks[finger_pip_id]
    
    # Finger is extended if tip is above (lower y value) the middle joint
    return tip.y < pip.y

def recognize_gesture(hand_landmarks):
    """Recognize gesture from hand landmarks"""
    landmarks = hand_landmarks.landmark
    
    # Check each finger (except thumb which needs different logic)
    index_extended = is_finger_extended(landmarks, 8, 6)
    middle_extended = is_finger_extended(landmarks, 12, 10)
    ring_extended = is_finger_extended(landmarks, 16, 14)
    pinky_extended = is_finger_extended(landmarks, 20, 18)
    
    # Thumb logic (different because it moves sideways)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_extended = thumb_tip.x < thumb_ip.x  # For right hand
    
    # Count extended fingers
    fingers_up = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    count = sum(fingers_up)
    
    # Classify gestures based on finger patterns
    if count == 0:
        return "FIST"
    elif count == 5:
        return "OPEN_HAND"
    elif fingers_up == [True, False, False, False, False]:
        return "THUMBS_UP"
    elif fingers_up == [False, True, True, False, False]:
        return "PEACE_SIGN"
    elif fingers_up == [False, True, False, False, False]:
        return "POINTING"
    else:
        return f"UNKNOWN ({count} fingers)"

while True:
    success, frame = camera.read()
    
    if success:
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(frame_rgb)
        
        # If hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)
                
                # Display gesture on screen
                cv2.putText(frame, gesture, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                print(f"Detected: {gesture}")
        
        # Show frame
        cv2.imshow('Gesture Recognition', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
hands.close()
print("Gesture recognition closed!")
import cv2
import mediapipe as mp
import math

print("Gesture Command System Starting...")
print("This maps gestures to robot commands")
print("Press 'q' to quit")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
camera = cv2.VideoCapture(0)

def is_finger_extended(landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is extended"""
    tip = landmarks[finger_tip_id]
    pip = landmarks[finger_pip_id]
    return tip.y < pip.y

def recognize_gesture(hand_landmarks):
    """Recognize gesture from hand landmarks"""
    landmarks = hand_landmarks.landmark
    
    # Check each finger
    index_extended = is_finger_extended(landmarks, 8, 6)
    middle_extended = is_finger_extended(landmarks, 12, 10)
    ring_extended = is_finger_extended(landmarks, 16, 14)
    pinky_extended = is_finger_extended(landmarks, 20, 18)
    
    # Thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_extended = thumb_tip.x < thumb_ip.x
    
    fingers_up = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    count = sum(fingers_up)
    
    # Classify gestures
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
        return "UNKNOWN"

def gesture_to_command(gesture):
    """Map gestures to robot commands"""
    command_map = {
        "FIST": "STOP",
        "OPEN_HAND": "MOVE_FORWARD",
        "THUMBS_UP": "INCREASE_SPEED",
        "PEACE_SIGN": "MOVE_BACKWARD",
        "POINTING": "TURN"
    }
    
    return command_map.get(gesture, "NO_COMMAND")

# Store last command to avoid spam
last_command = None

while True:
    success, frame = camera.read()
    
    if success:
        # Flip frame
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
                
                # Map to command
                command = gesture_to_command(gesture)
                
                # Display gesture and command
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Command: {command}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Print only when command changes
                if command != last_command:
                    print(f"Gesture: {gesture} -> Robot Command: {command}")
                    last_command = command
        else:
            last_command = None
        
        # Show frame
        cv2.imshow('Gesture Commands', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
hands.close()
print("Gesture command system closed!")
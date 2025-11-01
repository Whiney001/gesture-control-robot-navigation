import cv2
import mediapipe as mp

print("MediaPipe Hand Detection Starting...")
print("Press 'q' to quit")

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    
    if success:
        # Convert to RGB (MediaPipe needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(frame_rgb)
        
        # If hands detected, draw them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                print("Hand detected with 21 landmarks!")
        
        # Show the frame
        cv2.imshow('MediaPipe Hand Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
hands.close()
print("Hand detection closed!")
import cv2
import numpy as np

print("Simple Hand Detection Starting...")
print("This will detect skin color (works best in good lighting)")
print("Press 'q' to quit")

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, frame = camera.read()
    
    if success:
        # Convert to HSV color space (better for color detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (these values detect skin tones)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask to find skin-colored areas
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours (outlines) of skin-colored areas
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we found any contours
        if contours:
            # Get the largest contour (probably your hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only draw if it's big enough (not just noise)
            if cv2.contourArea(largest_contour) > 5000:
                # Draw a rectangle around the hand
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw the contour outline
                cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)
                
                # Add text
                cv2.putText(frame, "Hand Detected!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                print("Hand detected!")
        
        # Show the original frame with detection
        cv2.imshow('Hand Detection', frame)
        
        # Show the mask (for debugging - you can see what it detects)
        cv2.imshow('Skin Mask', mask)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("Hand detection closed!")
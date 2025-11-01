import cv2

print("Opening camera... Press 'q' to quit")

# Open your webcam
camera = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, frame = camera.read()
    
    if success:
        # Show the frame
        cv2.imshow('My Webcam Test', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("Camera closed!")
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('best.pt')

# Initialize video capture for two cameras
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(2)

# Set desired width and height for both cameras
width = 1920  # Desired width
height = 1080  # Desired height

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if ret1:
        # Detect objects and track on frame1
        results1 = model.track(frame1, persist=True)
        frame1_ = results1[0].plot()
        
        # Visualize frame1
        cv2.imshow('Camera 1', frame1_)

    if ret2:
        # Detect objects and track on frame2
        
        results2 = model.track(frame2, persist=True)
        frame2_ = results2[0].plot()
        frame2_ = frame2_[200:800, 600:1200]

        # Visualize frame2
        cv2.imshow('Camera 2', frame2_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()

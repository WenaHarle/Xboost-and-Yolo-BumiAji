import cv2

# Initialize the first webcam
cap1 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize the second webcam
cap2 = cv2.VideoCapture(4)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Define the codec and create VideoWriter object for the first webcam
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for mp4 format
out1 = cv2.VideoWriter('last.mp4', fourcc, 30.0, (1920, 1080))

# Define the codec and create VideoWriter object for the second webcam
out2 = cv2.VideoWriter('last.mp4', fourcc, 30.0, (1920, 1080))

while(cap1.isOpened() and cap2.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # Write the frames to the output video files
        out1.write(frame1)
        out2.write(frame2)

        # Display the frames
        cv2.imshow('frame1',frame1)
        cv2.imshow('frame2',frame2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

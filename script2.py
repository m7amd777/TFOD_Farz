import cv2
import numpy as np
import ultralytics

# Load the YOLO model
model = ultralytics.YOLO('best.pt')  # pretrained YOLOv8n model

# Function to capture image from webcam
def capture_image():
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('snapshot.jpg', frame)
    cap.release()

# Run inference on the captured image
def run_inference():
    results = model(['snapshot.jpg'], stream=True)  # return a generator of Results objects
    for result in results:
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk

# Main loop
while True:
        input("press enter")
        print("collecting images")
        capture_image()  # Capture image from webcam
        print ("predicting")
        run_inference()  # Run inference on the captured image

cv2.destroyAllWindows()

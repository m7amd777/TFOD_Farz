import numpy as np
from PIL import Image
import tensorflow as tf
import time
import cv2

model_path = 'best_float32.tflite'

def capture_image():
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('snapshot.jpg', frame)
    cap.release()

capture_image()

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Obtain the height and width of the corresponding image from the input tensor
image_height = input_details[0]['shape'][1] # 640
image_width = input_details[0]['shape'][2] # 640

# Image Preparation
image_name = 'snapshot.jpg'
image = Image.open(image_name)
image_resized = image.resize((image_width, image_height)) # Resize the image to the corresponding size of the input tensor and store it in a new variable

image_np = np.array(image_resized)
image_np = np.true_divide(image_np, 255, dtype=np.float32)
image_np = image_np[np.newaxis, :]

# inference
interpreter.set_tensor(input_details[0]['index'], image_np)

start = time.time()
interpreter.invoke()
print(f'run time：{time.time() - start:.2f}s')

# Obtaining output results
output = interpreter.get_tensor(output_details[0]['index'])
output = output[0]
output = output.T

boxes_xywh = output[..., :4] #Get coordinates of bounding box, first 4 columns of output tensor
scores = np.max(output[..., 4:], axis=1) #Get score value, 5th column of output tensor
classes = np.argmax(output[..., 4:], axis=1) # Get the class value, get the 6th and subsequent columns of the output tensor, and store the largest value in the output tensor.

# Threshold Setting
threshold = 0.3

# List to store predicted labels
predicted_labels = []

for box, score, cls in zip(boxes_xywh, scores, classes):
    if score >= threshold:
        predicted_labels.append(cls)

classes = {
    0: "BIODEGRADABLE",
    1: "CARDBOARD",
    2: "GLASS",
    3: "METAL",
    4: "PAPER",
    5: "PLASTIC"
}

prediction_label = classes[list(set(predicted_labels))[0]]
print("Predicted Label : ", prediction_label)
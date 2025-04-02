import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# load model
model = load_model("best_asl_model.h5")

#classes
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

#webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # region of interest
    height, width, _ = frame.shape
    x1, y1, x2, y2 = width//3, height//4, 2*width//3, 3*height//4
    roi = frame[y1:y2, x1:x2]
    
    # roi
    img = cv2.resize(roi, (64, 64))  # size for model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # dimensions for model input
    
    # Make prediction
    predictions = model.predict(img)
    confidence = np.max(predictions) * 100  # confidence score
    
    # Debugging output
    print(f"Model output shape: {predictions.shape}")  
    print(f"Model raw predictions: {predictions}")

    # Ensure predictions match class labels
    if predictions.shape[1] == len(class_labels):  
        label = class_labels[np.argmax(predictions)]
    else:
        label = "Unknown"  # Handle mismatch 

    print(f"Predicted label: {label}")
    
    # Draw bounding box and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("ASL Recognition", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop
cap.release()
cv2.destroyAllWindows()

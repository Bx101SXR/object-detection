import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model CNN
model = tf.keras.models.load_model('object_classifier.h5')

# Load label dari file
class_labels = {}
with open("class_labels.txt", "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        class_labels[int(idx)] = name

# Load deteksi objek (SSD)
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'mobilenet_iter_73000.caffemodel'
)

# Load deteksi wajah (Haar)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ukuran input klasifikasi
image_size = (100, 100)
THRESHOLD = 0.9

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    obj_detected = False

    # Deteksi wajah terlebih dahulu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Deteksi objek (SSD)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            obj_detected = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            roi_box = [x1, y1, x2, y2]

            # Cek apakah ROI ini bertabrakan dengan wajah
            is_face_region = False
            for (fx, fy, fw, fh) in faces:
                face_box = [fx, fy, fx + fw, fy + fh]
                # Periksa overlap antara ROI dan face_box
                ix1 = max(x1, face_box[0])
                iy1 = max(y1, face_box[1])
                ix2 = min(x2, face_box[2])
                iy2 = min(y2, face_box[3])
                if ix1 < ix2 and iy1 < iy2:
                    is_face_region = True
                    break

            # Jika overlaping dengan wajah, skip
            if is_face_region:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, image_size)
            img_array = image.img_to_array(roi_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(prediction)
            prob = np.max(prediction)

            if prob >= THRESHOLD:
                label = f"{class_labels[predicted_index]} ({prob:.2f})"
                color = (0, 255, 0)
            else:
                label = "Objek Tidak Dikenali"
                color = (0, 0, 255)

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not obj_detected:
        cv2.putText(frame, "Tidak Ada Objek Terdeteksi", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Deteksi dan Klasifikasi Objek", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

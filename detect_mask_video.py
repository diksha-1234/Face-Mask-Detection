from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import mysql.connector

# Function to connect to the database
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Replace with your DB username
            password="02062006@Manish",  # Replace with your DB password
            database="face_mask_detection"  # Replace with your DB name
        )
        print("[INFO] Database connected successfully")
        return connection
    except mysql.connector.Error as err:
        print(f"[ERROR] Database connection failed: {err}")
        return None

# Function to store user mask status along with their name, timestamp, and user ID in the database
def store_user_data(user_id, user_name, mask_status):
    try:
        connection = connect_to_db()
        if connection is None:
            return
        cursor = connection.cursor()

        # Get the current timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Check if the user already exists with the same mask status
        check_query = """
        SELECT * FROM users WHERE user_id = %s AND mask_status = %s
        """
        cursor.execute(check_query, (user_id, mask_status))
        result = cursor.fetchone()

        if result is None:
            # Insert the new record along with timestamp and user_id
            insert_query = """
            INSERT INTO users (user_id, name, mask_status, timestamp) 
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, user_name, mask_status, timestamp))
            connection.commit()
            print(f"[INFO] Data inserted for {user_id}: {user_name} - {mask_status} at {timestamp}")
        else:
            print(f"[INFO] No update needed for {user_id} - {mask_status}")

        cursor.close()
        connection.close()

    except mysql.connector.Error as err:
        print(f"[ERROR] Database operation failed: {err}")

# Function to detect faces and predict mask status
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the region of interest (ROI) for the face
            face = frame[startY:endY, startX:endX]
            if face is not None and face.size > 0:
                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
                except Exception as e:
                    print(f"[ERROR] Error processing face: {e}")

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Initialize the video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow camera sensor to warm up

stored_users = {}  # Dictionary to track already-stored users

# Load the serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.h5")

# Loop over the frames from the video stream
frame_processed = False
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces in the frame and determine mask status
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    if not frame_processed:
        # Process each detected face once
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display label and bounding box around the detected face region
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Generate a unique user ID based on coordinates or other logic
            user_id = f"user_{startX}_{startY}"

            # Display the detected user's label on the frame
            cv2.putText(frame, f"User: {user_id}", (startX, endY + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Show the output frame and capture user name input
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Ask for user name and store data for each user
        if key == ord("s"):
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                user_id = f"user_{startX}_{startY}"

                # Prompt for user's name
                user_name = input(f"Enter name for {user_id}: ")

                # Only store if the user status has changed or is not already stored
                if user_id not in stored_users or stored_users[user_id] != label:
                    store_user_data(user_id, user_name, label)
                    stored_users[user_id] = label

            # Automatically exit after storing the data
            print("[INFO] Data stored successfully. Exiting program.")
            frame_processed = True
            break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

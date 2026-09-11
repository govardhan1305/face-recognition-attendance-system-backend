from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

STUDENT_FOLDER = "students"
ATTENDANCE_FOLDER = "attendance"

os.makedirs(STUDENT_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)


# -----------------------------
# HOME / TEST
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "Face Recognition Attendance Backend is running!"
    })


# -----------------------------
# REGISTER STUDENT
# -----------------------------
@app.route("/api/register", methods=["POST"])
def register_student():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Image is required"
        }), 400

    name = request.form.get("name")
    student_id = request.form.get("student_id")

    if not name or not student_id:
        return jsonify({
            "success": False,
            "message": "Name and Student ID are required"
        }), 400

    image_file = request.files["image"]

    filename = f"{student_id}.jpg"
    filepath = os.path.join(STUDENT_FOLDER, filename)

    image_file.save(filepath)

    # Load image
    image = face_recognition.load_image_file(filepath)

    # Find face
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        os.remove(filepath)

        return jsonify({
            "success": False,
            "message": "No face detected"
        }), 400

    if len(face_locations) > 1:
        os.remove(filepath)

        return jsonify({
            "success": False,
            "message": "Multiple faces detected. Use only one face."
        }), 400

    # Create face encoding
    encoding = face_recognition.face_encodings(
        image,
        face_locations
    )[0]

    # Save encoding
    encoding_file = os.path.join(
        STUDENT_FOLDER,
        f"{student_id}.npy"
    )

    np.save(encoding_file, encoding)

    # Save student information
    info_file = os.path.join(
        STUDENT_FOLDER,
        f"{student_id}.txt"
    )

    with open(info_file, "w") as file:
        file.write(name)

    return jsonify({
        "success": True,
        "message": "Student registered successfully",
        "name": name,
        "student_id": student_id
    })


# -----------------------------
# RECOGNIZE FACE
# -----------------------------
@app.route("/api/recognize", methods=["POST"])
def recognize_face():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Image is required"
        }), 400

    image_file = request.files["image"]

    # Read uploaded image
    file_bytes = np.frombuffer(
        image_file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "success": False,
            "message": "Invalid image"
        }), 400

    # Convert BGR → RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Find faces
    face_locations = face_recognition.face_locations(
        rgb_image
    )

    if len(face_locations) == 0:
        return jsonify({
            "success": False,
            "message": "No face detected"
        })

    face_encodings = face_recognition.face_encodings(
        rgb_image,
        face_locations
    )

    known_encodings = []
    known_names = []
    known_ids = []

    # Load registered students
    for filename in os.listdir(STUDENT_FOLDER):

        if not filename.endswith(".npy"):
            continue

        student_id = filename.replace(".npy", "")

        encoding_path = os.path.join(
            STUDENT_FOLDER,
            filename
        )

        encoding = np.load(encoding_path)

        info_path = os.path.join(
            STUDENT_FOLDER,
            f"{student_id}.txt"
        )

        if os.path.exists(info_path):

            with open(info_path, "r") as file:
                name = file.read().strip()

            known_encodings.append(encoding)
            known_names.append(name)
            known_ids.append(student_id)

    if len(known_encodings) == 0:
        return jsonify({
            "success": False,
            "message": "No students registered yet"
        })

    # Compare face
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=0.50
        )

        distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

        best_match = np.argmin(distances)

        if matches[best_match]:

            name = known_names[best_match]
            student_id = known_ids[best_match]

            # Mark attendance
            mark_attendance(
                student_id,
                name
            )

            return jsonify({
                "success": True,
                "recognized": True,
                "name": name,
                "student_id": student_id,
                "message": "Attendance marked successfully"
            })

    return jsonify({
        "success": False,
        "recognized": False,
        "message": "Face not recognized"
    })


# -----------------------------
# MARK ATTENDANCE
# -----------------------------
def mark_attendance(student_id, name):

    today = datetime.now().strftime("%Y-%m-%d")

    time_now = datetime.now().strftime("%H:%M:%S")

    attendance_file = os.path.join(
        ATTENDANCE_FOLDER,
        f"{today}.csv"
    )

    # Prevent duplicate attendance
    if os.path.exists(attendance_file):

        with open(attendance_file, "r") as file:

            lines = file.readlines()

            for line in lines:

                if line.startswith(student_id + ","):
                    return

    # Create file if it doesn't exist
    file_exists = os.path.exists(attendance_file)

    with open(
        attendance_file,
        "a"
    ) as file:

        if not file_exists:
            file.write(
                "Student ID,Name,Date,Time,Status\n"
            )

        file.write(
            f"{student_id},{name},{today},{time_now},Present\n"
        )


# -----------------------------
# GET TODAY'S ATTENDANCE
# -----------------------------
@app.route("/api/attendance", methods=["GET"])
def get_attendance():

    today = datetime.now().strftime("%Y-%m-%d")

    attendance_file = os.path.join(
        ATTENDANCE_FOLDER,
        f"{today}.csv"
    )

    records = []

    if not os.path.exists(attendance_file):

        return jsonify({
            "success": True,
            "attendance": []
        })

    with open(attendance_file, "r") as file:

        lines = file.readlines()

        for line in lines[1:]:

            data = line.strip().split(",")

            if len(data) >= 5:

                records.append({
                    "student_id": data[0],
                    "name": data[1],
                    "date": data[2],
                    "time": data[3],
                    "status": data[4]
                })

    return jsonify({
        "success": True,
        "attendance": records
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

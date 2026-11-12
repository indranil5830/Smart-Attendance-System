import cv2
import json
import sqlite3
import csv
from pathlib import Path
from datetime import datetime, date

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "trainer.yml"
LABELS_PATH = DATA_DIR / "labels.json"
DB_PATH = DATA_DIR / "attendance.db"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            attendance_time TEXT NOT NULL,
            UNIQUE(student_id, attendance_date),
            FOREIGN KEY(student_id) REFERENCES students(student_id)
        )
    """)
    conn.commit()
    return conn

def get_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV face detector.")
    return detector

def register_student():
    student_id = input("Student ID (example: CSE001): ").strip()
    name = input("Student full name: ").strip()
    if not student_id or not name:
        print("ID and name are required.")
        return

    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO students(student_id, name, created_at) VALUES (?, ?, ?)",
            (student_id, name, datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        print("That student ID already exists. Use a unique ID.")
        conn.close()
        return
    conn.close()

    student_dir = FACES_DIR / student_id
    student_dir.mkdir(exist_ok=True)
    detector = get_detector()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera could not be opened. Check camera permissions.")
        return

    print("\nLook at the camera. Capture 30 face samples.")
    print("Press SPACE to save a face sample; press Q to stop.")
    saved = 0
    while saved < 30:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from camera.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {saved}/30 | SPACE=save Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Register student - consent required", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and len(faces) == 1:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(str(student_dir / f"{saved:03d}.jpg"), face)
            saved += 1
        elif key == ord(' ') and len(faces) != 1:
            print("Please ensure exactly one face is visible before saving.")
    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples for {name}.")
    if saved < 10:
        print("At least 10 samples are recommended. Register again if needed.")

def train_model():
    if not hasattr(cv2, "face"):
        print("OpenCV face module missing. Install opencv-contrib-python.")
        return
    faces, labels = [], []
    label_map = {}
    label_number = 0

    for student_dir in sorted(FACES_DIR.iterdir()):
        if not student_dir.is_dir():
            continue
        image_files = sorted(student_dir.glob("*.jpg"))
        if not image_files:
            continue
        label_map[str(label_number)] = student_dir.name
        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(cv2.resize(image, (200, 200)))
                labels.append(label_number)
        label_number += 1

    if not faces:
        print("No face samples found. Register at least one student first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, __import__("numpy").array(labels))
    recognizer.save(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"Model trained with {len(faces)} face samples from {len(label_map)} student(s).")

def mark_attendance():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Train the model first (option 2).")
        return
    if not hasattr(cv2, "face"):
        print("OpenCV face module missing. Install opencv-contrib-python.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    label_map = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    detector = get_detector()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera could not be opened. Check camera permissions.")
        return

    print("\nAttendance camera started. Press Q to quit.")
    print("Recognition threshold: 65 (lower confidence distance is a closer match).")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face)
            if str(label) in label_map and confidence < 65:
                student_id = label_map[str(label)]
                conn = db_connect()
                student = conn.execute(
                    "SELECT name FROM students WHERE student_id = ?", (student_id,)
                ).fetchone()
                if student:
                    now = datetime.now()
                    try:
                        conn.execute(
                            "INSERT INTO attendance(student_id, attendance_date, attendance_time) VALUES (?, ?, ?)",
                            (student_id, now.date().isoformat(), now.strftime("%H:%M:%S"))
                        )
                        conn.commit()
                        status = f"Marked: {student[0]}"
                        print(status)
                    except sqlite3.IntegrityError:
                        status = f"Already marked: {student[0]}"
                    conn.close()
                    color = (0, 200, 0)
                else:
                    status, color = "Unknown student", (0, 0, 255)
            else:
                status, color = "Unknown - try again", (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Smart Attendance - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

def show_report():
    conn = db_connect()
    rows = conn.execute("""
        SELECT s.student_id, s.name, a.attendance_date, a.attendance_time
        FROM attendance a JOIN students s ON s.student_id = a.student_id
        ORDER BY a.attendance_date DESC, a.attendance_time DESC
    """).fetchall()
    conn.close()
    if not rows:
        print("No attendance records yet.")
        return
    print("\nSTUDENT ID | NAME | DATE | TIME")
    print("-" * 65)
    for row in rows:
        print(" | ".join(row))

def export_csv():
    conn = db_connect()
    rows = conn.execute("""
        SELECT s.student_id, s.name, a.attendance_date, a.attendance_time
        FROM attendance a JOIN students s ON s.student_id = a.student_id
        ORDER BY a.attendance_date DESC, a.attendance_time DESC
    """).fetchall()
    conn.close()
    output = APP_DIR / "attendance_report.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Student ID", "Name", "Date", "Time"])
        writer.writerows(rows)
    print(f"CSV exported to: {output}")

def main():
    db_connect().close()
    while True:
        print("\n========== SMART ATTENDANCE SYSTEM ==========")
        print("1. Register student and capture face")
        print("2. Train / update face recognition model")
        print("3. Start attendance camera")
        print("4. View attendance report")
        print("5. Export attendance to CSV")
        print("6. Exit")
        choice = input("Choose an option (1-6): ").strip()
        if choice == "1":
            register_student()
        elif choice == "2":
            train_model()
        elif choice == "3":
            mark_attendance()
        elif choice == "4":
            show_report()
        elif choice == "5":
            export_csv()
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()

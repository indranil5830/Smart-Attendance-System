# Smart Attendance System (Beginner Hackathon MVP)

A local, consent-based face-recognition attendance prototype using Python, OpenCV LBPH, SQLite, and CSV export.

## Features
- Register a student and capture up to 30 face samples.
- Train/update an LBPH face-recognition model.
- Mark each recognized student once per day.
- View attendance and export it to CSV.
- Stores data locally on the computer.

## Requirements
- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Working webcam
- VS Code

## Setup (Windows PowerShell)
1. Install Python 3.11 and select "Add Python to PATH".
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   `py -3.11 -m venv .venv`
5. Activate it:
   `.venv\Scripts\Activate.ps1`
   If PowerShell blocks activation, run:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   then activate again.
6. Install packages:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
7. Run:
   `python app.py`

## First run
1. Choose `1` and register a consenting test participant. Use a unique ID.
2. Look at the webcam and press SPACE to save samples (up to 30); press Q to stop.
3. Choose `2` to train the model.
4. Choose `3` to start attendance. Press Q to stop.
5. Choose `4` to view records or `5` to export `attendance_report.csv`.

## Project structure
- `app.py`: registration, training, recognition, attendance, reporting
- `requirements.txt`: Python dependencies
- `data/faces/`: locally captured face samples (created on first run)
- `data/trainer.yml`: trained recognition model (created after training)
- `data/labels.json`: model label mapping
- `data/attendance.db`: SQLite database

## Architecture
Webcam -> OpenCV Haar face detector -> normalized grayscale face -> LBPH recognizer
-> student lookup in SQLite -> unique daily attendance row -> report / CSV export.

## Troubleshooting
- Camera unavailable: close Zoom/Meet/Camera, check OS camera permissions, then retry.
- `cv2.face` missing: uninstall `opencv-python` if installed, then run
  `pip uninstall opencv-python opencv-contrib-python -y`
  `pip install -r requirements.txt`
- Low recognition accuracy: capture samples in even lighting, facing camera, with small head-angle changes; retrain after adding students.
- On some machines, Python 3.10/3.11 may install OpenCV wheels more reliably than newer Python versions.

## Privacy and limitations
This is a hackathon prototype, not a production attendance or security system. Obtain informed consent before capturing biometric samples, explain the purpose and retention period, restrict access to the computer, and provide a non-biometric/manual alternative. Face recognition can make mistakes and can be affected by lighting, camera quality, and appearance changes. It does not include liveness detection and may be vulnerable to photo spoofing. Do not use it for high-stakes decisions or real institutional attendance without formal approval, security review, and appropriate privacy safeguards.

 Face-detection-model
This is a type of model
![ss1](https://github.com/user-attachments/assets/b06f0fef-ad65-4a5a-9aec-fa1b53c7050f)
# Face Recognition Student Attendance System

This is a Flask-based web application that uses computer vision and face recognition to automate student attendance. It allows users to register with a name and ID, capture their face images, train a recognition model, and then recognize students in real time using a webcam.

 🔧 Features

- Add new students with name and ID
- Automatically capture 20 face images per student
- Train a face recognition model
- Recognize and mark attendance in real time
- Attendance saved in CSV files
- Delete registered users

 🖥️ Technologies Used

- Python
- Flask
- OpenCV
- face_recognition
- pandas
- joblib

📁 Directory Structure

 ├── app.py ├── templates/ │ ├── home.html │ └── listusers.html ├── static/ │ ├── faces/ # Stored face images by user │ └── face_recognition_model.pkl ├── Attendance/ │ └── Attendance-<date>.csv ├── haarcascade_frontalface_default.xml
 
## 🚀 Getting Started

### Prerequisites

Install the following Python libraries:

```bash
pip install flask opencv-python face_recognition pandas joblib
Also, make sure you have:

A webcam for capturing live video

haarcascade_frontalface_default.xml in the project directory

Note: On Windows, you may need to install cmake and dlib dependencies for face_recognition
📷 How to Use
Register Student

Go to the "Add User" form on the homepage.

Enter student name and ID.

The app will capture 20 images of the face via webcam.

Train Model

The model is automatically trained after capturing new faces.

Start Recognition

Click on the "Start" option.

The webcam will detect faces and mark attendance if a face is recognized.

View Attendance

Attendance is saved daily in Attendance/Attendance-<date>.csv.

Manage Users

Use the "List Users" page to view or delete registered users.

📝 Notes
Attendance will not be marked again for the same ID on the same day.

If all users are deleted, the trained model is also deleted and must be retrained.

📄 License
This project is open-source and free to use for educational purposes.

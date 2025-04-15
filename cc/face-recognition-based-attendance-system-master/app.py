import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
import joblib
import face_recognition
import pandas as pd

# Defining Flask App
app = Flask(__name__)

nimgs = 20  # Number of images to capture

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return []


# Function to extract features (encodings) from face
def extract_face_encoding(face_image):
    try:
        # Convert the image to RGB (required by face_recognition)
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Get the face encodings (embedding)
        encodings = face_recognition.face_encodings(rgb_image)
        if len(encodings) > 0:
            return encodings[0]  # Return the first (and only) face encoding
        else:
            print("No encoding found for the face")
            return None
    except Exception as e:
        print(f"Error in encoding face: {e}")
        return None


# Identify face using face_recognition model
def identify_face(face_image):
    try:
        # Load the saved face encodings and labels
        face_encodings_data = joblib.load('static/face_recognition_model.pkl')
        faces = face_encodings_data['faces']
        labels = face_encodings_data['labels']

        # Extract encoding for the input face
        encoding = extract_face_encoding(face_image)
        if encoding is None:
            return None

        # Compare the input face encoding to stored faces using Euclidean distance
        distances = face_recognition.face_distance(faces, encoding)
        min_distance_index = distances.argmin()

        # If the minimum distance is below a threshold, recognize the face
        if distances[min_distance_index] < 0.6:
            return labels[min_distance_index]  # Return the label (user)
        else:
            return None  # Unknown face
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None


# A function which trains the model on all the faces available in faces folder
def train_model():
    try:
        faces = []
        labels = []
        userlist = os.listdir('static/faces')
        for user in userlist:
            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                encoding = extract_face_encoding(img)
                if encoding is not None:
                    faces.append(encoding)
                    labels.append(user)
                else:
                    print(f"Skipping image {imgname} as no encoding was found.")

        if len(faces) > 0:
            # Store the encodings and their corresponding labels in a dictionary
            face_encodings = {'faces': faces, 'labels': labels}
            joblib.dump(face_encodings, 'static/face_recognition_model.pkl')
            print("Model trained successfully!")
        else:
            print("No faces found during training.")
    except Exception as e:
        print(f"Error during training model: {e}")


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names, rolls, times, l
    except Exception as e:
        print(f"Error extracting attendance data: {e}")
        return [], [], [], 0


# Add Attendance of a specific user
def add_attendance(name):
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")

        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
        else:
            print(f"Attendance already recorded for {name}")
    except Exception as e:
        print(f"Error adding attendance for {name}: {e}")


# A function to get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


# A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    # if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except Exception as e:
        print(f"Error while training model: {e}")

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Our main Face Recognition functionality.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y + h, x:x + w], (100, 100))  # Matching new image size for training

                identified_person = identify_face(face)  # Call the new face recognition function

                if identified_person:
                    add_attendance(identified_person)  # Mark attendance for the identified person
                    cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Add new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
        print(f"Created folder for new user: {userimagefolder}")

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:  # Capture an image every 5th frame
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                print(f"Captured image: {name}")
                i += 1
            j += 1
        if i >= nimgs:  # Stop when 20 images are captured
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # Train model after image capture
    try:
        print('Training Model')
        train_model()
    except Exception as e:
        print(f"Error during model training: {e}")

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

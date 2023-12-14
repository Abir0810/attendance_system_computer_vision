#!/usr/bin/env python
# coding: utf-8

# In[19]:


import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


# In[20]:


video_capture = cv2.VideoCapture(0)


# In[21]:


messi_image = face_recognition.load_image_file(r"E:\attendence\photos\messi.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]


abir_image = face_recognition.load_image_file(r"E:\attendence\photos\Abir.jpg")
abir_face_encoding = face_recognition.face_encodings(abir_image)[0]

ronaldo_image = face_recognition.load_image_file(r"E:\attendence\photos\ronaldo.jpg")
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]

arfat_image = face_recognition.load_image_file(r"E:\attendence\photos\Arfat.jpg")
arfat_face_encoding = face_recognition.face_encodings(arfat_image)[0]


# In[22]:


known_face_encodings = [
    messi_face_encoding,
    abir_face_encoding,
    ronaldo_face_encoding,
    arfat_face_encoding
]
known_faces_names = [
    "Messi",
    "Abir",
    "Ronaldo",
    "Arfat"
    
]


# In[23]:


students= known_faces_names.copy()


# In[24]:


face_locations = []
face_encodings = []
face_names = []
s = True


# In[25]:


now=datetime.now()
current_date=now.strftime("%Y-%m-%d")


# In[26]:


f=open(current_date+'.csv','w+',newline='')
Inwriter=csv.writer(f)


# In[27]:


while True:
    # Grab a single frame of video
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    Inwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()


# In[ ]:





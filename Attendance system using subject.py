#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import face_recognition
import cv2
import numpy as np
import os
import xlwt
from xlwt import Workbook
from datetime import date
import xlrd, xlwt
from xlutils.copy import copy as xl_copy
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


# In[ ]:





# In[ ]:


video_capture = cv2.VideoCapture(0)


# In[ ]:


messi_image = face_recognition.load_image_file(r"E:\attendence\photos\messi.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]


abir_image = face_recognition.load_image_file(r"E:\attendence\photos\Abir.jpg")
abir_face_encoding = face_recognition.face_encodings(abir_image)[0]

ronaldo_image = face_recognition.load_image_file(r"E:\attendence\photos\ronaldo.jpg")
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]


# In[ ]:


known_face_encodings = [
    messi_face_encoding,
    abir_face_encoding,
    ronaldo_face_encoding
]
known_faces_names = [
    "Messi",
    "Abir",
    "Ronaldo"    
]


# In[ ]:


students= known_faces_names.copy()


# In[ ]:


face_locations = []
face_encodings = []
face_names = []
s = True


# In[ ]:


rb = xlrd.open_workbook(r'E:\attendence\attendence.xls', formatting_info=True) 
wb = xl_copy(rb)
inp = input('Please give current subject lecture name')
sheet1 = wb.add_sheet(inp)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row=1
col=0
already_attendence_taken = ""


# In[ ]:


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
            if((already_attendence_taken != name) and (name != "Unknown")):
                sheet1.write(row, col, name )
                col =col+1
                sheet1.write(row, col, "Present" )
                row = row+1
                col = 0
                print("attendence taken")
                wb.save(r'E:\attendence\attendence.xls')
                already_attendence_taken = name
            else:
                print("next student") 
    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





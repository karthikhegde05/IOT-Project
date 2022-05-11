import numpy as np
import cv2

class Dnn_face_detector():

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")


    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

        self.net.setInput(blob)

        faces = self.net.forward()

        face_lst = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype('int')

                x1 -= x
                y1 -= y

                face_lst.append((x, y, x1, y1))
        return face_lst
            
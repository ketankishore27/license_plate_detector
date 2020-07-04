# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:36:23 2020

@author: Ketan
"""

import cv2
import numpy as np
import glob
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('CloudVision.json')

client = vision.ImageAnnotatorClient(credentials=credentials)

def google_vision_detect(client, content):
    
    image = vision.types.Image(content = content)
    response = client.text_detection(image = image)
    result = response.text_annotations[0].description
    return result.replace('\n', ' ').strip()

def detection_plate(net, output_layers, img_path):
            
    detection = []
    x, y, w, h = 0, 0, 0, 0
    # Loading image
    img = cv2.imread(img_path)
    result = img.copy()
    #img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                #print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            bg_color = (0,0,0)
            bg = np.full((img.shape), bg_color, dtype=np.uint8)
            text_color = (0,0,255)
            cv2.imwrite("temp.jpg", img[y:y+h, x:x+w])
            try:
                with open('temp.jpg', 'rb') as image_file:
                    content = image_file.read()
                text = google_vision_detect(client, content)
            except Exception as e:
                text = 'Could not Detect'
                print("Probably not detected anything, Encountered an exception --> {}".format(str(e)))
                
            cv2.putText(bg, text, (5,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, text_color, 1)
            X, Y, W, H = cv2.boundingRect(bg[:,:,2])
            result[Y:Y+H, X:X+W] = bg[Y:Y+H, X:X+W]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(img, pytesseract.image_to_string(img[y:y+h, x:x+w]), (x, y+30), font, 3, (0, 0, 255), 2)

    cv2.imshow("Image", result)
    cv2.imwrite("google_vision_output/{}".format(img_path.split('\\')[1]), result)
    key = cv2.waitKey(0)
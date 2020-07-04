# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:47:36 2020

@author: Ketan
"""

import cv2
import numpy as np
from detect_plates import *
#import glob
#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

net = cv2.dnn.readNet("yolov3_custom_final.weights", "yolov3_customs.cfg")

classes = ["Vehicle registration plate"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

images_path = glob.glob(r"testing_images/*.jpg")
#images_path = ["testing_images/number_plate3.jpg"]

detection = []
for img_path in images_path:
    detection_plate(net, output_layers, img_path)
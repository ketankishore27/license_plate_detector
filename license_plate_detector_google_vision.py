# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:35:32 2020

@author: Ketan
"""

import cv2
import numpy as np
import glob
from detection_google_vision import *

net = cv2.dnn.readNet("yolov3_custom_final.weights", "yolov3_customs.cfg")

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
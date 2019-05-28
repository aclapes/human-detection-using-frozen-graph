# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import os
import glob
import csv
from ast import literal_eval
import argparse


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', dest='input_file', default="/data/data4/RGBDT_data/*/rs/color/*.jpg",
                        help='First')
    parser.add_argument('--begin', dest='begin', type=int, default=0,
                        help='First')
    parser.add_argument('--end', dest='end', type=int, default=0,
                        help='End')
    parser.add_argument('--gpu-id', dest='gpu_id', default=0,
                        help='Gpu id')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    framepath_list = glob.glob(args.input_file)
    end = args.end
    if args.end < 1 or args.end > len(framepath_list):
        end = len(framepath_list)

    framepath_list = sorted(framepath_list)[args.begin:end]

    # Model 1: Trade-off speed/accuracy
    # http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
    # model_name = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

    # Model 2: Accuracy model
    # http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz
    model_name = 'faster_rcnn_nas_coco_2018_01_28'

    # OR download any other model from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    model_suffix = 'frozen_inference_graph.pb'
    model_path = os.path.join("models", model_name, model_suffix)

    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.3

    output_path = os.path.join('inferences', model_name)

    csv_filename = "human-detections." + model_name + "." + str(args.begin) + "-" + str(end) + ".csv"
    csvfile = open(csv_filename, 'w')

    fieldnames = ['filepath', 'box', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for frame_p in framepath_list:
        # read and resize if needed
        img = cv2.imread(frame_p)
        if img.shape[1] != 1280 or img.shape[0] != 720:
            img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        human_boxes = []
        for i in range(len(boxes)):
            # Class 1 represents human in COCO dataset labels
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                writer.writerow({'filepath': frame_p, 'box': boxes[i], 'score': scores[i]})
                # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(output_path, os.path.basename(frame_p)), img)

    csvfile.close()











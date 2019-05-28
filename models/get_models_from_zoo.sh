#!/bin/bash

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz

tar xvzf faster_rcnn_nas_coco_2018_01_28.tar.gz
tar xvzf faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz

rm faster_rcnn_nas_coco_2018_01_28.tar.gz
rm xvzf faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz


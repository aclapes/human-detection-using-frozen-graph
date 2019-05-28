import numpy as np
import csv
import ast
# import matplotlib.pyplot as plt
from collections import OrderedDict

csv_src_filename = "/Volumes/MacintoshHD/Users/aclapes/Downloads/human.faster_rcnn_nas_coco_2018_01_28.csv"
csv_dst_filename = "/Volumes/MacintoshHD/Users/aclapes/Downloads/human.faster_rcnn_nas_coco_2018_01_28.filtered.csv"

D = OrderedDict()  # read detections
scores = []
scores_filt = []
with open(csv_src_filename, 'r') as csv_src:
    reader = csv.DictReader(csv_src)
    for line in reader:
        frame_p = line['filepath']
        D.setdefault(frame_p, dict(boxes=[],scores=[]))
        D[frame_p]['boxes'].append(ast.literal_eval(line['box']))
        D[frame_p]['scores'].append(float(line['score']))
        scores.append(float(line['score']))

    with open(csv_dst_filename, 'w') as csv_dst:
        writer = csv.DictWriter(csv_dst, fieldnames=reader.fieldnames)
        for fp, d in D.iteritems():
            filt_idx = np.argmax(d['scores'])
            writer.writerow({'filepath': frame_p, 'box': d['boxes'][filt_idx], 'score': d['scores'][filt_idx]})
            scores_filt.append(d['scores'][filt_idx])

# plt.hist(scores, bins=np.linspace(0, 1., 21))
# plt.title('Histograma de probabilidades de las detecciones')
# plt.xlabel('#{Bounding boxes}')
# plt.xlabel('Probabilidad clase humano')
# plt.show()
#
# plt.hist(scores_filt, bins=np.linspace(0, 1., 21))
# plt.title('Histograma de probabilidades de las detecciones (filtered)')
# plt.xlabel('#{Bounding boxes}')
# plt.xlabel('Probabilidad clase humano')
# plt.show()
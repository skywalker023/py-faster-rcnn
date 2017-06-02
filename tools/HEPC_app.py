#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
For application use
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


from matplotlib import font_manager, rc
import json
import base64
import post_process as pp
from time import gmtime, strftime


CLASSES = ('__background__',  # always index 0
                         'cookedrice', 'katsudon', 'jjajangmyun', 'coldnoodle', 'ramyun', 'bulgogi', 'frystiredpork','braisedspicychicken','stirfriedanchovy',
                         'kimchi','tangsuyook','dumpling','tteokbokki','sundae','beansprout','rolledOmelette','porkCutlet','kkakdugi','friedChicken',
                         'galbijjim','galbitang','kimchiPancake','kimchiStew','seasonedSesame','koreanMeatball','doenjangStew','seaweedSoup','spicyNoodles',
                         'bibimbap','samgyetang','spam','yukgaejang','jabchae','beansproutSoup','friedShrimp','salad','udon','friedVege','jangjorim','bossam')

MENU=('배경',
      '쌀밥', '가츠동', '자장면', '냉면', '라면', '불고기', '제육볶음', '닭볶음탕', '멸치볶음', '배추김치', '탕수육', '만두', '떡볶이', '순대', '콩나물무침', '계란말이', '돈까스',
      '깍두기', '후라이드치킨', '갈비찜', '갈비탕', '김치전', '김치찌개', '깻잎무침', '동그랑땡', '된장찌개', '미역국', '비빔국수', '비빔밥', '삼계탕', '스팸', '육개장', '잡채',
      '콩나물국', '새우튀김', '샐러드', '우동', '채소튀김', '장조림', '보쌈')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_70000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

OUTPUT_DIR='/home/ciplab/PycharmProjects/server/images/output.jpg'
FONT="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
JSON_DIR='/home/ciplab/PycharmProjects/server/images/json.txt'

rare_menu=['jangjorim']

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im= cv2.imread(im_file)


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)

    # print("Scores shape: ", scores.shape, " Boxes shape: ", boxes.shape)
    # print("scores: ", scores, " Boxes: ", boxes)


    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    # CONF_THRESH = 0.5
    NMS_THRESH = 0.2

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    result = []
    idx_for_json=[]

    for cls_ind, cls in enumerate(CLASSES[1:]):

        if cls=='cookedrice':
            CONF_THRESH=0.4
        elif cls in rare_menu:
            CONF_THRESH = 0.95
        else:
            CONF_THRESH=0.05

        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]



        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            # print("bbox: ",bbox)
            # print("score: ",score)

            current_box=(np.append(bbox,score).tolist(),cls)

            # print("current box: ",current_box)

            result=pp.decide_boxes(result,current_box)

            # print("result: ",result)



    for inx, bb in enumerate(result):
    # bb=([x1,y1,x2,y2,scr],cls)
    #     ax.add_patch(
    #         plt.Rectangle((bb[0][0], bb[0][1]),
    #                       bb[0][2] - bb[0][0],
    #                       bb[0][3] - bb[0][1], fill=False,
    #                       edgecolor='grey', linewidth=3.5)
    #     )

        title = "  " + MENU[CLASSES.index(bb[1])] + "  "

        ax.text((bb[0][0] + bb[0][2]) / 2, bb[0][1],
                title,
                bbox=dict(facecolor='0.06', alpha=0.5),
                fontsize=19, color='white')


        idx_for_json.append(CLASSES.index(bb[1]))



    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    print('image detection complete!')
    return idx_for_json


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # print('xxxxxxx: prototxt',prototxt)
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    reload(sys)
    sys.setdefaultencoding('utf8')
    font_name = font_manager.FontProperties(fname=FONT).get_name()
    rc('font', family=font_name)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['input.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        idx_for_json=demo(net, im_name)


    plt.savefig(OUTPUT_DIR, transparent=True, bbox_inches='tight', pad_inches=0)
    print("Saved output image to "+ OUTPUT_DIR)

    with open(OUTPUT_DIR, "rb") as output:
        encoded_output = base64.b64encode(output.read())

    print("Encoded output image to byte array")

    json_data = {
        'classes': idx_for_json,
        'image': encoded_output
    }

    with open(JSON_DIR, 'w') as outfile:
        json.dump(json_data, outfile)


    print("Detected food: ", list(map(lambda x: str(unicode(MENU[x])), idx_for_json)))
    print("Wrote json data to "+JSON_DIR)
    print("All done! ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
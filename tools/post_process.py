import numpy as np


def get_covering_box(bbox1, bbox2):
    # bbox1 and bbox2 format: ([x1,y1,x2,y2, score], class)
    result = []

    for i in range(4):
        if i < 2:
            result.append(min(bbox1[0][i], bbox2[0][i]))
        else:
            result.append(max(bbox1[0][i], bbox2[0][i]))


    # print("cover box coords: ",result)
    if result == bbox1[0][:-1]:
        # print("cover box: ", result)
        print(bbox1)
        if bbox1[1]=='cookedrice':
            return bbox2
        return bbox1
    elif result == bbox2[0][:-1]:
        # print("cover box: ", result)
        print(bbox2)
        if bbox2[1]=='cookedrice':
            return bbox1
        return bbox2
    else:
        # print("bbox1: ",bbox1)
        # print("bbox2: ",bbox2)
        return None


def check_IoU(bbox1, bbox2, THRES=0.7):
    # bbox1 and bbox2 format: ([x1,y1,x2,y2, score], class)

    bb1 = bbox1[0][:-1]
    bb2 = bbox2[0][:-1]

    area1 = get_area(bb1)
    area2 = get_area(bb2)

    inter = get_intersection(bb1, bb2)
    union = get_union(area1, area2, inter)

    IoU = float(inter) / union
    # print("area1: ",area1)
    # print("area2: ",area2)
    # print("intersection: ", inter)
    # print("iou: ",IoU)
    if IoU >= THRES:
        return True
    else:
        return False


def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_intersection(bbox1, bbox2):

    w = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    h = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])

    if w>0 and h>0:
        return w * h
    else:
        return 0



def get_union(area1, area2, inter):
    return area1 + area2 - inter


def combine_box(bbox1, bbox2):
    # bbox1 and bbox2 format: ([x1,y1,x2,y2, score], class)
    combined_bbox = list(map(lambda x, y: (x + y) / 2, bbox1[0][:-1], bbox2[0][:-1]))

    # print(combined_bbox)

    if bbox1[0][-1] >= bbox2[0][-1]:
        combined_bbox.append(bbox1[0][-1])
        return (combined_bbox, bbox1[1])
    else:
        combined_bbox.append(bbox2[0][-1])
        return (combined_bbox, bbox2[1])


def decide_boxes(result, incoming):
    overwritten = False

    for idx, det in enumerate(result):

        cover_box = get_covering_box(det, incoming)

        if cover_box:
            # overwrite the bbox info
            # print("covered")
            result[idx] = cover_box
            overwritten = True
        elif check_IoU(det, incoming):

            # if IoU larger than THRES, combine two with max(score) cls
            # exception: cookedrice
            if det[1]!='cookedrice' and incoming[1]!='cookedrice':
                # print("combining : ",det[1], incoming[1])
                result[idx] = combine_box(det, incoming)
                overwritten = True
            # elif incoming[1]=='cookedrice':
            #     # if incoming is rice, ignore it
            #     continue
            elif det[1]=='cookedrice':
                # if the one in the result is rice, overwrite it with incoming
                result[idx]=incoming
                overwritten=True
        else:
            # if IoU is smaller, just add it to the result
            continue

    if not result:
        result.append(incoming)
    elif not overwritten:
        # print("append incoming: ", incoming[1], incoming[0])
        result.append(incoming)

    return result

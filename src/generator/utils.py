from PIL.Image import NONE
import numpy as np
import skimage.feature
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def clip_to_image(xy, image_size):
    return max(0, min(xy, image_size))


def gaussian_radius(height, width, min_overlap=0.7): 
  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return int(min(r1, r2, r3)) 


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(map, center, radius, k=1):
    heatmap = map[...,0]
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return map


def draw_bbox(img, xc, yc, w, h, write = False, write_dir = None, ind = 0):
    corner1 = (int(xc-w/2)*4, int(yc-h/2)*4)
    corner2 = (int(xc+w/2)*4, int(yc+h/2)*4)
    img = cv2.rectangle(img, corner1, corner2, (255, 0, 0), thickness=5)
    plt.imshow(img[:,:,::-1])
    plt.show()
    if write:
        cv2.imwrite(write_dir+'im_'+str(ind)+'.png', img)


def generate_center_heatmap(output_size, gt, img = NONE):
    max_x = 0
    min_x = 10
    max_y = 0
    min_y = 10
    for _ in range(len(gt)):
        max_x = gt[_]["x"] if gt[_]["x"]>max_x else max_x
        max_y = gt[_]["y"] if gt[_]["y"]>max_y else max_y
        min_x = gt[_]["x"] if gt[_]["x"]<min_x else min_x
        min_y = gt[_]["y"] if gt[_]["y"]<min_y else min_y
    xc = int((max_x+min_x)*output_size[0]/2)
    yc = int((max_y+min_y)*output_size[1]/2)
    map = np.zeros((output_size[0],output_size[1], 3)) 
    w = (max_x - min_x)*output_size[0]
    h = (max_y - min_y)*output_size[1]
    r = gaussian_radius(h, w)
    map = draw_gaussian(map, (xc,yc), r)
    map[yc,xc,1] = w
    map[yc,xc,2] = h

    if isinstance(img, np.ndarray):
        draw_bbox(img, xc, yc, w, h)
    # plt.imshow(map[...,0])
    # plt.show()

    return map


def decode_center(pred, min_distance=5):
    max_idx=skimage.feature.peak_local_max(
        pred[:,:,0],
        min_distance=min_distance,
        num_peaks = 1
    )
    yc = max_idx[:,0]
    xc = max_idx[:,1]
    return xc, yc


def convert_to_corners(boxes):
    return np.concatenate([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)


def compute_iou(boxes1, boxes2):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = np.maximum(boxes1_corners[..., :2], boxes2_corners[..., :2])   
    rd = np.minimum(boxes1_corners[..., 2:], boxes2_corners[..., 2:])   
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = np.maximum(
        boxes1_area + boxes2_area - intersection_area, 1e-8
    )
    return np.clip(intersection_area / union_area, 0.0, 1.0)

# def decode_centernet(hm):
#     w, h= hm.shape[0], hm.shape[1]
#     _, topk_inds = tf.math.top_k(tf.reshape(hm,[-1]))
#     topk_yc   = (topk_inds / w).int().float()
#     topk_xc   = (topk_inds % w).int().float()
#     return topk_xc, topk_yc
    

def visu_pred(data_generator, detector, res_path='src/results/'):
    for i in range(data_generator.__len__()):
        imgs = data_generator.__getitem__(i)[0]  # 0是Input, 1是label
        img_original = data_generator.getitem(i)
        
        pred = detector.predict(x=imgs)
        # we only process the first image of the batch
        hm = tf.expand_dims(pred[0][...,0], axis=2).numpy()
        (xc, yc) = decode_center(hm)
        w = pred[0][...,1][yc, xc]
        h = pred[0][...,2][yc, xc]
        ratio_h, ratio_w = img_original.shape[0]/imgs.shape[1], img_original.shape[1]/imgs.shape[2]      # imgs (batch, h, w, c)
        xc_original = ratio_w * xc
        yc_original = ratio_h * yc
        w_original = ratio_w * w
        h_original = ratio_h * h
        # draw_bbox(imgs[0].astype(int), xc, yc, w, h)
        draw_bbox(img_original.astype(int), xc_original, yc_original, w_original, h_original, True, res_path, i)
        
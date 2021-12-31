import tensorflow as tf
from generator.utils import compute_iou, decode_center
from param import *
import numpy as np
from model.DNN import create_DNN


def test():
    detector = create_DNN(params_data['input_size'])
    ckpt = tf.train.Checkpoint(detector)
    ckpt.restore(tf.train.latest_checkpoint(params_train["saved_model_path"])).expect_partial()
    IoU = 0
    for _ in range(len(test_generator)):
        img = test_generator[_][0]
        pred = detector.predict(img)
        hm = tf.expand_dims(pred[0][...,0], axis=2).numpy()
        (xc_pred, yc_pred) = decode_center(hm)
        w_pred = pred[0][...,1][yc_pred, xc_pred]
        h_pred = pred[0][...,2][yc_pred, xc_pred]
        bbox_pred = np.array([xc_pred.item(), yc_pred.item(), w_pred.item(), h_pred.item()])

        label = test_generator[_][1]
        hm_true = label[0,...,0]
        pos_mask = tf.equal(hm_true, 1)
        (yc_true, xc_true) = np.nonzero(pos_mask)
        w_true = label[0,...,1][pos_mask]
        h_true = label[0,...,2][pos_mask]
        bbox_true = np.array([xc_true.item(), yc_true.item(), w_true.item(), h_true.item()])

        IoU += compute_iou(bbox_pred, bbox_true)
    IoU /= len(test_generator)
    
if __name__ == "__main__":
    test()
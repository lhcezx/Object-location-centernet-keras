import tensorflow as tf


def compute_loss(y_true, y_pred, dimensions):
    batch_size=dimensions[0]
    
    hm_true = y_true[...,0]
    hm_pred = y_pred[...,0]
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.math.pow(1 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    hm_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)

    L1_loss_w = 0
    L1_loss_w += tf.math.abs(tf.reduce_sum(y_pred[...,1]*pos_mask - y_true[...,1]*pos_mask))

    L1_loss_h = 0
    L1_loss_h += tf.math.abs(tf.reduce_sum(y_pred[...,2]*pos_mask - y_true[...,2]*pos_mask))

    wh_loss = L1_loss_w + L1_loss_h
    return hm_loss + 0.1*wh_loss/tf.cast(batch_size, dtype=tf.float32) 


class Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=2, gamma=4, dimensions=(32,32,32)):
        super().__init__(name='FocalLoss', reduction=tf.keras.losses.Reduction.NONE)
        self.alpha = alpha
        self.gamma = gamma
        self.dimensions = dimensions
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        return compute_loss(y_true, y_pred, self.dimensions)


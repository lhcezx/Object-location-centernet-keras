import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dropout, Conv2DTranspose, BatchNormalization, ReLU
from keras.regularizers import l2


def create_center_heads(x):
    # hm header
    hm = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    hm = BatchNormalization()(hm)
    hm = ReLU()(hm)
    hm = Conv2D(1, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(hm)

    # wh header
    wh = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    wh = BatchNormalization()(wh)
    wh = ReLU()(wh)
    wh = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(wh)

    outputs = tf.concat([hm, wh], axis = 3)
    return outputs


def create_DNN(input_shape):
    inputs=tf.keras.layers.Input(shape=input_shape)	
    model_name = ''
    backbone = tf.keras.applications.ResNet50(include_top=False, input_tensor = inputs, weights = 'imagenet')
    C5 = backbone.outputs[-1]
    x = Dropout(rate=0.5)(C5)
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    x = create_center_heads(x)
    DNN = tf.keras.models.Model(inputs, x, name=model_name)
    return DNN



class debug(keras.Model):
    def __init__(self, input_shape, backbone, head, **kwarg):
        super().__init__(**kwarg)
        self.shape = input_shape
        self.backbone = backbone
        self.head = head
        self.DNN = create_DNN(input_shape, backbone, head)
        
    def call(self, x):
        x = self.DNN(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as t:
            y_pred = self(x, training = True)
            loss = self.compiled_loss(y, y_pred)

        vars = self.trainable_variables
        grad = t.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grad, vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def summary(self):
        model = create_DNN(self.shape, self.backbone, self.head)
        return model.summary()





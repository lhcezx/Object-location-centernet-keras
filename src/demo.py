from param import *
from model.DNN import *
from generator.utils import visu_pred

def demo():
    detector = create_DNN(params_data['input_size'])
    ckpt = tf.train.Checkpoint(detector)
    ckpt.restore(tf.train.latest_checkpoint(params_train["saved_model_path"])).expect_partial()
    visu_pred(test_generator, detector, res_path='src/results/')

if __name__ == "__main__":
    demo()
from tensorflow.python.eager.context import run_eager_op_as_function_enabled
from model.DNN import create_DNN
from param import *

Restore = True

def train():
    detector = create_DNN(params_data['input_size'])
    detector.compile(
        optimizer=params_train['optimizer'], 
        loss=params_train['loss'], 
        metrics=None,
        # run_eagerly=True
    )
    # detector.summary()
    if Restore:
        ckpt = tf.train.Checkpoint(detector)
        ckpt.restore(tf.train.latest_checkpoint(params_train["saved_model_path"]))
    else:
        print("Train from the begin")

    detector.fit(  
        x=training_generator,
        epochs = 20,
        validation_data = test_generator,
        callbacks = callbacks_list
    )


if __name__ == "__main__":
    train()

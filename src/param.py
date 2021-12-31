from model.losses import *
from generator.data_generator import DataGenerator
import os

params_data = {
    'input_size': (800, 800, 3),
    'output_size': (200, 200, 3),
    'batch_size': 4,
    'shuffle': True,
    'dataset_path': 'src/dataset/',
}

# Parameters for the model
params_train = { 
    'loss' : Loss(dimensions=(params_data['batch_size'],*params_data['output_size'])),
    'optimizer' : tf.keras.optimizers.Adam(learning_rate=0.0001),
    'saved_model_path' : 'src/ckpt'                             #you can specifiy the name of the model you want to save or load
}

training_generator = DataGenerator(
    batch_size=params_data['batch_size'], 
    input_size=params_data['input_size'], 
    output_size=params_data['output_size'], 
    dataset_dir=params_data['dataset_path'],
    shuffle=params_data['shuffle']
)

# and for test
test_generator = DataGenerator(
    batch_size=1, 
    input_size=params_data['input_size'], 
    output_size=params_data['output_size'], 
    dataset_dir=params_data['dataset_path'],
    shuffle=False,
    train=False
)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params_train['saved_model_path'], "weights" + "_epoch_{epoch}"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch"
        )
    ]
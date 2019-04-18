import csv
import cv2
import numpy as np
from model import model as m
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320


def load_data(csv_path='data/driving_log.csv'):
    """
    csv_path: string (path of .csv file where image name and mesurnments are stored).
    This function will be used to read image name from .csv file 
    and load images and its stearing measurnment.
    
    This function also flip the images and store as training data. 
    Obviously will be negative of original image.
    
    return: X_train, y_train
    """
    lines = []
    
    # Reading csv file and appending in a list
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
        lines = lines[1:]

    images = []
    measurnments = []

    # Get images and mesurnments. Also stored fliped images and negative of mesurnment.
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/{}'.format(filename)
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurnment = float(line[3])
        measurnments.append(measurnment)
        
        image = np.fliplr(image)
        images.append(image)
        measurnments.append(-1.0*measurnment)
        
    X_train = np.array(images)
    y_train = np.array(measurnments)

    return X_train, y_train


def train(X_train, y_train, save_model='model.h5'):
    """
    This function will be use to train model and save model for given training set.
    X_train: numpy array of training images
    y_train: numpy array of stearing mesurnments.
    save_model: string (name of model, default is model.h5)
    
    return: None
    """
    
    # Hyperparameters
    batch_size = 32
    epochs = 30
    learning_rate = 0.001
    
    # Loading model from model.py
    model = m(input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH)
    
    # Plot model as image
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    # If trained model exist already then load first for further training
    if tf.gfile.Exists(save_model):
        model.load_weights(save_model)
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    
    # Only save model which has best performed on validation set.
    # These are callbacks which are being used in "model.fit" call
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.2, shuffle=True)
    
    return

if __name__ == '__main__':
    # Load data
    X_train, y_train = load_data()
    # Train model
    train(X_train, y_train)
    
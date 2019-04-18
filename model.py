from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input, Dropout, GlobalAveragePooling2D, Cropping2D
from keras.applications.mobilenet import MobileNet
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input


def preprocess(image, input_size=160):
    """
    Preprocess images in batch. 
    It will resized image and normalize the image between -1.0 to 1, which is required for "MobileNet" model.
    image: image tensor
    input_size: int (size if input images, default is 160)
    
    return: tensor of resized and preprocessed images for "MobileNet" model
    """
    image = tf.image.resize_images(image, (input_size, input_size))
    return preprocess_input(image)

def model(input_height, input_width):
    """
    Model architecture is defined here. Image preprocessing is part of model architecture.
    This modle is "MobileNet" model of "ImageNet" without last fully connected layers. 
    Its first 40 layes are freezed (can not be trained).
    As "MibileNet" supports some specific images sizes so image preprocessing as 
    added as input layes Lambda function. Finally flatten the "MobileNet" output and added two Dense layer.
    
    input_height: int (height of images)
    input_width: int (width of images)
    
    return: model with i
    
    """
    input_size = 160
    weights_flag = 'imagenet'
    freeze_layer_count = 40
    
    mobilenet = MobileNet(weights=weights_flag, include_top=False, alpha=1.0, depth_multiplier=1, input_shape=(input_size, input_size, 3))
    
    for i, layer in enumerate(mobilenet.layers):
        if i < freeze_layer_count:
            layer.trainable = False
        else:
            layer.trainable = True
        
    image_input = Input(shape=(input_height, input_width, 3))
#     cropped_image = Cropping2D(((70, 0), (0, 0)))(image_input)
    resized_input = Lambda(lambda image: preprocess(image, input_size))(image_input)
    inp = mobilenet(resized_input)
    x = GlobalAveragePooling2D()(inp)
    x = Dropout(0.5)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    measurnment = Dense(1)(x)
    m = Model(inputs=image_input, outputs=measurnment)
    return m

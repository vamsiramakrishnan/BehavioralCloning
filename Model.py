
# coding: utf-8

# In[1]:

############ Data Processing #######################
import pandas as pd

########## Arithmetic ######################
import numpy as np
import cv2
from skimage import filters
from skimage import exposure, img_as_ubyte

################# Neural Network Model ##############
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout, Cropping2D, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import metrics
import h5py

################### Other utilites ###################
import os
import warnings


# ### Image Perturbation Techniques

# In[2]:

################ Augmentation functions ##########################
def random_brightness(image):
    image_ = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness_factor = .15 + np.random.uniform()
    image_[:, :, 2] = image_[:, :, 2] * brightness_factor
    image_ = cv2.cvtColor(image_, cv2.COLOR_HSV2RGB)
    return image_


def gaussian(image):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma_ = np.random.uniform(0.25, 1.)
        image_ = filters.gaussian(image, sigma=sigma_, multichannel=True)
        image_ = img_as_ubyte(image_, force_copy=True)
    return image_


def ahisteq(image):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_ = exposure.equalize_adapthist(image)
        image_ = img_as_ubyte(image_, force_copy=True)
    return image_


# ### Image Generators
# #### Key Features of the Training Generator
# - **Random Gaussian**( Noise and blurring to handle multiple resolutions of images) , **Brightness**( To allow the generation of multiple Images) and **Histogram Equalization** ( To help the model learn features easily)
# - **Colorspace shifting** ( The images could be read as RGB and BGR ) . So instead of making sure the conversion is correct we train the model with a 50% chance of **RGB** and 50% chance of BGR to make it invariant to color. 

# In[3]:

################### Batch Image Generator Function for model training ####
def create_training_dataset():
    while 1:
        X_ = []
        Y_ = []

        # Create a Random Normal Sample based on weights from the complete Data Frame
        ####### n = Number of Samples defined by the the batch size ###########
        Sample_ = complete_data.sample(
            n=BATCH_SIZE, weights=complete_data['weights'])
        for i in range(len(Sample_)):

            # Handles to randomize actions color space perturbation,
            # brightness, histogram equalization and flipping
            flip_coin1 = np.random.randint(0, 2)
            flip_coin2 = np.random.randint(0, 2)
            flip_coin3 = np.random.randint(0, 2)
            flip_coin4 = np.random.randint(0, 2)

            j = np.random.randint(1, 4)
            k = j + 7
            steer_ = Sample_.iloc[i][k]
            image_ = cv2.imread(Sample_.iloc[i][j], 1)

            if flip_coin2 == 0:
                image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            if flip_coin4 == 0:
                image_ = ahisteq(image_)
            if flip_coin3 == 0:
                image_ = random_brightness(image_)

            if flip_coin1 == 0:
                image_ = cv2.flip(image_, 1)
                steer_ = -1.0 * steer_

            X_.append(image_)
            Y_.append(steer_)
        X = np.array(X_)
        Y = np.array(Y_)

        yield X, Y


################### Batch Image Generator Function for model validation ##
def create_validation_dataset():
    while 1:
        ValSample_ = complete_data.sample(
            n=BATCH_SIZE, weights=complete_data['weights'])
        X_ = []
        Y_ = []
        for i in range(len(ValSample_)):
            j = 1
            k = j + 7
            image_ = cv2.imread(ValSample_.iloc[i][j])
            image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            steer_ = ValSample_.iloc[i][k]
            X_.append(image_)
            Y_.append(steer_)

        X = np.array(X_)
        Y = np.array(Y_)

        yield X, Y


# ### Model Helper Functions 

# In[4]:

################ Serialize and Save the Model ####################
def model_save(model_, model_name):
    model_json = model_.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model_.save_weights(model_name + ".h5")
    model_.save(model_name + "_full.h5")
    print("Saved Model to Disk")


############### Load the Model #############
def model_load(model_name, isFromCheckpoint_=True):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    if not isFromCheckpoint_:
        loaded_model.load_weights(model_name + ".h5")
    else:
        loaded_model.load_weights(CHECKPOINT_PATH)
    print("Model Loaded ")
    return loaded_model

def resize_image(image):
    # This import is required here otherwise the model cannot be loaded in
    # drive.py
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))


# ### Model Architecture
# 
# - The NVIDIA Architecture for end to end deep learning is used. 
# - Batch Normalization is implemented after every Conv2D and Dense Layers before activations (ELU)
# - Switched from ReLU to ELU after reading this paper-https://arxiv.org/abs/1511.07289 and the NVIDIA implementation
# - Dropouts are implemented before the flatten layer and before the output layer with a 50% probability
# - Tried by adding multiple dropouts but it did need seem to have an effect on improving validation losses. 
# 
# <pre>
# <font size="3.5"><em><strong>Layer (type)                 Output Shape              Param #</b></em></font>   
# =================================================================
# <b>cropping2d_1 (Cropping2D)</b>    (None, 90, 320, 3)        0         
# _________________________________________________________________
# <b>lambda_1 (Lambda)</b>            (None, 66, 200, 3)        0         
# _________________________________________________________________
# <b>lambda_2 (Lambda)</b>            (None, 66, 200, 3)        0         
# _________________________________________________________________
# <b>conv2d_1 (Conv2D)</b>            (None, 31, 98, 24)        1824      
# _________________________________________________________________
# <b>batch_normalization_1</b> (Batch (None, 31, 98, 24)        96        
# _________________________________________________________________
# <b>elu_1 (ELU)</b>                  (None, 31, 98, 24)        0         
# _________________________________________________________________
# <b>conv2d_2 (Conv2D)</b>            (None, 14, 47, 36)        21636     
# _________________________________________________________________
# <b>batch_normalization_2</b> (Batch (None, 14, 47, 36)        144       
# _________________________________________________________________
# <b>elu_2 (ELU)</b>                  (None, 14, 47, 36)        0         
# _________________________________________________________________
# <b>conv2d_3 (Conv2D)</b>            (None, 5, 22, 48)         43248     
# _________________________________________________________________
# <b>batch_normalization_3</b> (Batch (None, 5, 22, 48)         192       
# _________________________________________________________________
# <b>elu_3 (ELU)</b>                  (None, 5, 22, 48)         0         
# _________________________________________________________________
# <b>conv2d_4 (Conv2D)</b>            (None, 3, 20, 64)         27712     
# _________________________________________________________________
# <b>batch_normalization_4</b> (Batch (None, 3, 20, 64)         256       
# _________________________________________________________________
# <b>elu_4 (ELU)</b>                  (None, 3, 20, 64)         0         
# _________________________________________________________________
# <b>conv2d_5 (Conv2D)</b>            (None, 1, 18, 64)         36928     
# _________________________________________________________________
# <b>batch_normalization_5</b> (Batch (None, 1, 18, 64)         256       
# _________________________________________________________________
# <b>elu_5 (ELU)</b>                  (None, 1, 18, 64)         0         
# _________________________________________________________________
# <b>flatten_1 (Flatten)</b>          (None, 1152)              0         
# _________________________________________________________________
# <b>dropout_1 (Dropout)</b>          (None, 1152)              0         
# _________________________________________________________________
# <b>dense_1 (Dense)</b>              (None, 1164)              1342092   
# _________________________________________________________________
# <b>batch_normalization_6</b> (Batch (None, 1164)              4656      
# _________________________________________________________________
# <b>elu_6 (ELU)</b>                  (None, 1164)              0         
# _________________________________________________________________
# <b>dense_2 (Dense)</b>              (None, 100)               116500    
# _________________________________________________________________
# <b>batch_normalization_7</b> (Batch (None, 100)               400       
# _________________________________________________________________
# <b>elu_7 (ELU)</b>                  (None, 100)               0         
# _________________________________________________________________
# <b>dense_3 (Dense)</b>              (None, 50)                5050      
# _________________________________________________________________
# <b>batch_normalization_8</b> (Batch (None, 50)                200       
# _________________________________________________________________
# <b>elu_8 (ELU)</b>                  (None, 50)                0         
# _________________________________________________________________
# <b>dense_4 (Dense)</b>              (None, 10)                510       
# _________________________________________________________________
# <b>batch_normalization_9</b> (Batch (None, 10)                40        
# _________________________________________________________________
# <b>elu_9 (ELU)</b>                  (None, 10)                0         
# _________________________________________________________________
# <b>dropout_2 (Dropout)</b>          (None, 10)                0         
# _________________________________________________________________
# <b>dense_5 (Dense)</b>              (None, 1)                 11        
# 
# Total params: 1,601,751
# Trainable params: 1,598,631
# Non-trainable params: 3,120
# 
# </pre>

# In[ ]:

# Flags / Hyper Parameters
BATCH_SIZE = 32
TEST_TRAIN_RATIO = 0.2
EPOCHS = 10
MODEL_NAME = "20170522_Model"
UNLABELLED_DATASET = "Data.tar"
DROPOUT_RATE = 0.5
IS_LOADMODEL = True
IS_FROMCHECKPOINT = True
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "./" + MODEL_NAME + "_full.h5"
complete_data = pd.read_csv("./Complete_Data.csv")


# ### Choice of Hyper Parameters
# - **BATCH SIZE** is chosen at 32 due to the following reasons apart from choosing it proportional to the dataset size
#     - The Data is shared between GPU and CPU. 
#     - The Adaptive HistEqfunction can tend to delay the speed of computation. 
#     - 32 Seems to Strike the balance. 
# - The number of **EPOCHS** is chosen as 10 on the entire dataset but the model has been trained on 1 epoch on subsets where the car faced issues during turning in Track-2 - Uphill turns. 
# - **Dropout rate** is chosen as 50% at the end of the feature extractor and just before the output layer. 
# - The Default Learning rate of **1e-2** is not doing a great job in finding the optimum so a learning rate.

# In[6]:

########################## Generate Training Samples during runtime ######
train_generate = create_training_dataset()
valid_generate = create_validation_dataset()
# Load older model or create new model if required
if IS_LOADMODEL:

    model_ = model_load(MODEL_NAME, IS_FROMCHECKPOINT)
    model_.compile(loss='mse', optimizer=Adam(lr=0.0001))

    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, monitor='val_loss',
                                 save_best_only=True, save_weights_only=False, mode='auto')
    model_.fit_generator(train_generate, steps_per_epoch=5, epochs=EPOCHS, validation_data=valid_generate,
                         validation_steps=100, callbacks=[checkpoint], max_q_size=1, pickle_safe=True)
    model_save(model_, MODEL_NAME)
# Create new model if flag is set
else:
    ##################### Build Model Architecture ###########################
   # Crop 50 pixels from the top of the image and 20 from the bottom
    Model_ = Sequential()
    Model_.add(Cropping2D(data_format="channels_last", cropping=(
        (50, 20), (0, 0)), input_shape=(160, 320, 3)))
    Model_.add(Lambda(resize_image))
    Model_.add(Lambda(lambda x: x / 255 - 0.5))

    ################## Block-1 ########################
    Model_.add(Conv2D(24, (5, 5), kernel_initializer='he_normal', strides=(
        2, 2), padding='valid'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    ################# Block-2 ########################
    Model_.add(Conv2D(36, (5, 5), kernel_initializer='he_normal', strides=(
        2, 2), padding='valid'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    ################ Block-3 #########################
    Model_.add(Conv2D(48, (5, 5), kernel_initializer='he_normal', strides=(
        2, 2), padding='valid'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    ################ Block-4 #########################
    Model_.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', strides=(
        1, 1), padding='valid'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    ################ Block-4 #########################
    Model_.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', strides=(
        1, 1), padding='valid'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    Model_.add(Flatten())

    ################ End of Feature Extractor ####################
    ############### Start of Fully Connected Layers ##############
    Model_.add(Dropout(DROPOUT_RATE))
    #################### Layer-1 ##################################
    Model_.add(Dense(1164, kernel_initializer='he_normal'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())

    #################### Layer-2 ##################################
    Model_.add(Dense(100, kernel_initializer='he_normal'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())

    #################### Layer-3 ##################################
    Model_.add(Dense(50, kernel_initializer='he_normal'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())

    #################### Layer-4 ##################################
    Model_.add(Dense(10, kernel_initializer='he_normal'))
    Model_.add(BatchNormalization())
    Model_.add(ELU())
    Model_.add(Dropout(DROPOUT_RATE))

    #################### Layer-5 ##################################
    Model_.add(Dense(1))
    Model_.compile(loss='mse', optimizer=Adam(lr=0.0001))
    
    ################### Model Run ##############################
    Model_.fit_generator(train_generate, steps_per_epoch=300, epochs=EPOCHS,
                         validation_data=valid_generate, validation_steps=LEARNING_RATE)
    model_save(Model_, MODEL_NAME)


# In[ ]:




###### model  ######
###### imports ######
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

###### constants ######
path_to_model = "/Users/dan/ITC/jungo/model"


class My_Cnn:
    """
    Creating my CNN from Tensorflow
    must input dimensions shape of 1 image,
    choose loss_function: 'binary' , 'categorical', 'sparse'
    choose logits: True or False
    """
    def __init__(self, img_dim, loss_function='categorical', logits=True, loaded_model=0, debug_printing=0):
        # get shape of image for model input
        self.img_height = img_dim[0]
        self.img_width = img_dim[1]
        self.channels = img_dim[2]
        self.debug_prints = debug_printing
        self.loss_function = loss_function
        self.logits = logits
        # check if model is loaded or new.
        if loaded_model:
            self.model = self.load_model()
        else:
            self.model = self.create_model()
        # place holder for before model is trained
        self.history = None
        self.trained_epoch = None

    def create_model(self):
        """
        Create a CNN model
        :return: Tensorflow model
        """
        data_augmentation = self.get_preprocessing()
        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, self.channels)),
            # conv layer 1
            layers.Conv2D(16, 5, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            # conv layer 2
            layers.Conv2D(32, 3, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            # conv layer 3
            layers.Conv2D(64, 3, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            # conv layer 4
            layers.Conv2D(64, 3, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            # flattened
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.25),
        ])
        if (self.loss_function == 'binary') and self.logits:
            model.add(layers.Dense(1))
        elif (self.loss_function == 'binary') and not self.logits:
            model.add(layers.Dense(1, activation='sigmoid'))
        elif not self.logits:
            model.add(layers.Dense(2, activation='sigmoid'))
        else:
            model.add(layers.Dense(2))

        return model

    def get_preprocessing(self):
        """
        create a layer for preprocessing which is active only during training phase of the model
        :return: Sequential layers
        """
        data_augmentation = Sequential([
            # flip images to add a horizontal flip randomness to training
            tf.keras.layers.RandomFlip(mode='horizontal',
                                       input_shape=(self.img_height, self.img_width, self.channels)),
            # add Gaussian smoothing to each image as noise for training
            tf.keras.layers.GaussianNoise(0.2)
        ])
        return data_augmentation

    def train_model(self, train_set, val_set, epochs, patience, lr=0.001):
        """
        train the model
        :param train_set:
        :param val_set:
        :param epochs:
        :param patience:
        :param lr:
        :return:
        """
        # chooseing loss function
        if self.loss_function == 'sparse':
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.logits)
        elif self.loss_function == 'categorical':
            loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=self.logits)
        else:
            loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=self.logits)
        # compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer,
                           loss=loss_func,
                           metrics=['accuracy'])
        # Callbacks for training
        early_callback = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       restore_best_weights=True)
        noise_update_callback = PreprocessCallback(self.debug_prints)

        # training
        self.history = self.model.fit(train_set,
                                      validation_data=val_set,
                                      epochs=epochs,
                                      callbacks=[early_callback, noise_update_callback]
                                      )

        # in stopped epoch minus the patience indicates the selected epoch weights,
        # if not stopped then last epoch weights used
        if early_callback.stopped_epoch != 0:
            self.trained_epoch = early_callback.stopped_epoch - patience
        else:
            self.trained_epoch = early_callback.stopped_epoch

    def get_inference_time(self, img, loop_count=1000):
        """
        measure the inference time of the model.
        by measuring time before and after the model has predicted.
        :param img:
        :param loop_count:
        :return: Mean and STD of inference time
        """
        times_list = np.array([], dtype=float)
        for index in range(loop_count):
            before = tf.timestamp()
            self.model.predict(img, verbose=0)
            after = tf.timestamp()
            inf_time = (after-before)*1000
            times_list = np.append(times_list, inf_time)
        return np.mean(times_list), np.std(times_list)

    def print_inference_time(self, img, loop_count=1000):
        """
        print inference time
        :param img:
        :param loop_count:
        :return:
        """
        inf_time_avg, inf_time_std = self.get_inference_time(img, loop_count)
        print(f"Inference time avg for {inf_time_avg:.05f}ms +/- {inf_time_std:.05f}ms")

    def save_model(self):
        """
        save the model to dir
        :return: None
        """
        if not os.path.exists(path_to_model):
            print('error file path to model does not exist')
            exit(1)
        self.model.save(path_to_model)

    @staticmethod
    def load_model():
        """
        load model from dir
        :return: model
        """
        if not os.path.exists(path_to_model):
            print('error file path to model does not exist')
            exit(1)
        return tf.keras.models.load_model(path_to_model)


class PreprocessCallback(tf.keras.callbacks.Callback):
    """
    creating a custom callback to add change randomly the gussian filter in the preprocessing
    """
    def __init__(self, debug_print=0):
        self.debug_print = debug_print

    def on_epoch_begin(self, epoch, logs=None):
        # Gaussian layer update stddev
        self.model.layers[0].layers[1].stddev = random.uniform(0, 1)
        if self.debug_print:
            print('updating sttdev in training')
            print(self.model.layers[0].layers[1].stddev)



def main():
    print('model.py')
    model = My_Cnn([50, 50, 1])


if __name__ == '__main__':
    main()

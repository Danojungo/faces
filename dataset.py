###### dataset  ######
###### imports ######

import os
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory

###### constants ######


class Dataset:
    """
    Create a Data set from "tensorflow image_dataset_from_directory"
    with 1 constrained all images must be in the same directory and sorted into the
    sub directories according to each class
    A test set is created from the same structure but added "_test" at the end of the main directory
    labels_mode chooses which loss funcation to use:
    'int': sparse_categorical_crossentropy (********'sparse' on cnn file*****)
    'categorical': categorical_crossentropy
    'binary' : binary_crossentropy
    """
    def __init__(self, file_path, image_dims, labels_mode='categorical', batch_size=32, seed=0):
        # just in case of an error in file path
        if not os.path.exists(file_path):
            print('error file path to data set does not exist')
            exit(1)
        self.file_path = file_path
        self.img_height = image_dims[0]
        self.img_width = image_dims[1]
        self.channels = image_dims[2]
        self.batch_size = batch_size
        self.seed = seed
        self.label_mode = labels_mode
        # give a random seed to the test image
        self.ts_seed = np.random.randint(1000)
        self.train_set = self.create_train_set()
        self.val_set = self.create_val_set()
        self.test_set = self.create_test_set()

    def create_train_set(self):
        """
        create a train set of images
        :return: batch dataset
        """
        train_ds = image_dataset_from_directory(self.file_path,
                                                labels='inferred',
                                                label_mode=self.label_mode,
                                                color_mode='grayscale',
                                                validation_split=0.2,
                                                subset='training',
                                                seed=self.seed,
                                                image_size=(self.img_height, self.img_width),
                                                batch_size=self.batch_size)
        return train_ds

    def create_val_set(self):
        """
        Create a Validation set of images
        :return: batch dataset
        """
        val_ds = image_dataset_from_directory(self.file_path,
                                              labels='inferred',
                                              label_mode=self.label_mode,
                                              color_mode='grayscale',
                                              validation_split=0.2,
                                              subset='validation',
                                              seed=self.seed,
                                              image_size=(self.img_height, self.img_width),
                                              batch_size=self.batch_size)
        return val_ds

    def create_test_set(self):
        """
        create a Test set from unseen images.
        :return: batch dataset
        """
        test_path = self.file_path + "_test"
        if not os.path.exists(test_path):
            # just in case of an error in file path
            print('error file path to test data set does not exist')
            exit(1)
        test_ds = image_dataset_from_directory(test_path,
                                               labels='inferred',
                                               label_mode=self.label_mode,
                                               color_mode='grayscale',
                                               # validation_split=0.99,
                                               # subset='training',
                                               seed=self.ts_seed,
                                               image_size=(self.img_height, self.img_width),
                                               batch_size=1)
        return test_ds


def main():
    image_dims = [50, 50, 1]
    ds = Dataset("/home/dano/clearml_poc/face_validator_data", image_dims)
    print(f"Train DS{ds.train_set}")
    print(f"Val DS{ds.val_set}")
    print(f"Test DS{ds.test_set}")
    print(f'All DS have been made')


if __name__ == '__main__':
    main()

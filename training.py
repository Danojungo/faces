###### imports ######
import dataset
import model
from clearml import Task, Logger, OutputModel
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import tempfile
import helper_functions as hf

###### constants ######
path_to_images = "/home/dano/clearml_poc/face_validator_data"
image_dims = [50, 50, 1]

task = Task.init(project_name="face_validatior", task_name="logging parameters first")
params_dictionary = {'epochs': 10, 'lr': 0.0005, 'patience': 5, 'last_dense': 64, 'batch_size': 16}
task.connect(params_dictionary)
output_model = OutputModel(task=task, framework="tensorflow")
output_model.set_upload_destination(uri='/home/dano/clearml_poc')


def main():
    """
    this file creates and trains a model, then after analysing the plots enter 'y' to save or anything else to not save.
    :return:
    """
    data = dataset.Dataset(path_to_images, image_dims, labels_mode='categorical', seed=24,
                           batch_size=params_dictionary['batch_size'])
    cnn = model.My_Cnn(image_dims, logits=True, loss_function='categorical', loaded_model=0,
                       debug_printing=0, dense=params_dictionary['last_dense'])
    cnn.train_model(data.train_set, data.val_set, params_dictionary['epochs'],
                    params_dictionary['patience'], lr=params_dictionary['lr'])
    score = cnn.model.evaluate(data.test_set)
    history = cnn.history
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    Logger.current_logger().report_scalar(title='evaluate', series='score',
                                          value=score[0], iteration=params_dictionary['epochs'])
    Logger.current_logger().report_scalar(title='evaluate', series='accuracy',
                                          value=score[1], iteration=params_dictionary['epochs'])
    # inference time from dataset
    # cnn.print_inference_time(data.test_set, 1000)
    # print accuracy and loss plots
    # hf.print_history(cnn.history, cnn.trained_epoch)
    # choose if to save new created model or not.
    # user_save = input("save the model? ")
    # if user_save == 'y':
    #     cnn.save_model()
    output_folder = os.path.join('/home/dano/clearml_poc', 'saveexaple')
    model_store = ModelCheckpoint(filepath=os.path.join(output_folder, 'weight.hdf5'))
    task.update_output_model('/home/dano/clearml_poc')
    print('finished')


if __name__ == '__main__':
    main()

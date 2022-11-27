###### imports ######
import dataset
import model
import helper_functions as hf

###### constants ######
path_to_images = "/home/dano/clearml_poc/face_validator_data"
image_dims = [50, 50, 1]


def main():
    """
    this file creates and trains a model, then after analysing the plots enter 'y' to save or anything else to not save.
    :return:
    """
    data = dataset.Dataset(path_to_images, image_dims, labels_mode='categorical', seed=24, batch_size=16)
    cnn = model.My_Cnn(image_dims, logits=True, loss_function='categorical', loaded_model=0, debug_printing=0)
    cnn.train_model(data.train_set, data.val_set, 200, 20, lr=0.0005)
    cnn.model.evaluate(data.test_set)
    # inference time from dataset
    cnn.print_inference_time(data.test_set, 1000)
    # print accuracy and loss plots
    hf.print_history(cnn.history, cnn.trained_epoch)
    # choose if to save new created model or not.
    user_save = input("save the model? ")
    if user_save == 'y':
        cnn.save_model()
    print('finished')


if __name__ == '__main__':
    main()
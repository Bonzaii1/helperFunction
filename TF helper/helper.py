


def walk_through_directories(target_dir):
    import os
    """
    A function that will walk through all the folders and files in a directory and print them out for you

    args:
        `target_dir` (str) : the path of the directory in which you wish to walk through
    """

    # Walk through the pizza_steak directory and list the number of files
    for dirpath, dirnames, filenames in os.walk(target_dir):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}')


def get_class_names(target_folder):
    import pathlib
    import numpy as np

    """
    Extract all of the classes found in the directory. 
    NOTE: only works if files are in the following heirachy
    root
        - train
          - classname1
            -file1
            -file2
          - classname2
            -file1
            -file2
          - classname3
            -file1
            -file2
        - test
          -(Same idea as in the train folder)

    args:
        `target_folder` (str): Path of the folder that contains the classnames
    """

    data_dir = pathlib.Path(target_folder)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return class_names



def view_random_image(target_dir, target_class):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random
    # Setup the target directory (We'll view the images from here)
    target_folder = target_dir + target_class

    #Get random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)

    #Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')

    print(f"Image shape: {img.shape}") # show the shape of the image

    return img


def plot_loss_curves(history):
    import matplotlib.pyplot as plt
    """
    Returns separate loss curves for training and validation

    args:
        history (History object) : The history object returned by the `model.fit()` method
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history['loss'])) # How many epochs did we run for

    # Plot loss
    plt.plot(epochs, loss, label = 'training loss')
    plt.plot(epochs, val_loss, label = 'test loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    #Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label = 'training accuracy')
    plt.plot(epochs, val_accuracy, label = 'test accuracy')
    plt.title("accuracy")
    plt.xlabel("Epochs")
    plt.legend()
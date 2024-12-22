# Aygaz Görüntü İşleme Bootcamp 
* The project was prepared by İbrahim Okan Önal and Eda Şentürk.
* [Click to go to Kaggle Project ](https://www.kaggle.com/code/edaentrk/animal-detection-with-cnn)
* [Click to go to İbrahim Okan Önal Linkedin ](https://www.linkedin.com/in/okan-%C3%B6nal-437a5b248/)
* [Click to go to Eda Şentürk Linkedin ](https://www.linkedin.com/in/edasenturk/)

## Aygaz Görüntü İşleme Bootcamp 2024 
# Purpose of the Project:

- This project aims to create a Convolutional Neural Network (CNN) model for classifying animals.
- This model, which has the ability to effectively analyze and process visual data, obtains accurate results by recognizing and classifying different types of animals (e.g. "elephant", "fox", "rabbit"). **


# **1- Python Libraries :** 
* os:Used for file and directory operations, such as accessing folders or managing files.
* numpy:A powerful library for mathematical computations, arrays, and matrix operations.
* cv2:(OpenCV): Facilitates image processing tasks, including reading, editing, and manipulating images.
* PIL: is commonly used in image processing tasks.
* pytorch:is a popular deep learning framework used for a variety of reasons, particularly in research and production settings.
* sklearn:is widely used because of its ease of use, variety of machine learning algorithms, powerful data preprocessing tools, model evaluation capabilities, and integration with other libraries
* matplotlib:A library for data visualization, enabling the creation of graphs, histograms, and other visual representations.
* pandas: is used because it provides a rich set of tools for data manipulation, cleaning, and analysis, with its powerful DataFrame structure being the backbone of most data-related tasks.
  

#  **2- Using the Dataset :**
* This section defines the dataset and hyperparameter settings for deep learning model training. 
* The dataset is taken from the specified directory on Kaggle and only images belonging to specific animal classes such as collie, dolphin, elephant will be used. 
* All images are resized to 128x128 pixels to be compatible with the model and a maximum of 650 images are selected from each class. 
* During training, the dataset will be divided into 35 batches and the model will be trained for a total of 20 epochs. 
* For computations, cuda is used if a GPU is available; otherwise, it will work with the CPU. 
* These settings ensure that the data is processed and the model training is compatible with the hardware.

# **3- Preparing the Dataset :**
* This section covers the data preparation process for the deep learning model. First, a function is defined to load images and class labels. This function takes images belonging to target classes located in a specific file path, resizes them, and adds them to two separate lists with their class numbers. These lists are then converted to numpy arrays and returned as output.
* After the data loading process, the dataset is divided into training and test. This separation is done at 70% for training and 30% for testing. Training and test data are kept separate to evaluate the performance of the model.
* A series of transformation operations are defined to make the images suitable for the model. These operations include steps such as resizing the images, converting to grayscale, blurring, converting to tensor format, and normalizing. These transformations prepare the images in a format suitable for the model and at a scale.
* Finally, the dataset is organized for training and test data with PyTorch's Dataset and DataLoader structures. While Dataset is the basic unit of the dataset, DataLoader splits this data into small groups (mini-batches) so that the model can use the data more efficiently during training. While the training data is shuffled, the test data is loaded sequentially. This structure makes the data preparation process completely automatic and optimized for model training.


# **4- Designing the CNN Model :**
# Designing the CNN Model
* This section involves building and preparing a Convolutional Neural Network (CNN) model for training.

# Model Structure
* CNNModel Class: Inherits from PyTorch's nn.Module and defines a CNN model with two main components:

  * features: Handles feature extraction.
    * The first convolutional layer takes 3 input channels (e.g., RGB channels of a color image) and outputs 32 channels.
    * Batch normalization and an activation function (ReLU) are applied.
    * Max pooling reduces the spatial dimensions.
    * Dropout is applied to prevent overfitting.
    * The second convolutional layer takes 32 input channels and outputs 64 channels, repeating the same operations.
  * classifier: Responsible for classification. It is initialized as None and dynamically created during the forward pass based on the input shape.
* _initialize_classifier Method: Defines the classification layer dynamically based on the flattened size of the output from the features layers. The classification layer includes a fully connected (Linear) layer that outputs the number of classes.

* forward Method: Defines the forward pass of the model.
 * Input data is passed through the features layers.
If the classifier is not yet initialized, it is created dynamically using the flattened size of the output from the features layers and moved to the appropriate device (CPU or GPU).
 * The data is then passed through the classifier to produce the final outputs.
* Model Preparation for Training
  * Model Initialization: An instance of the model is created, with the number of classes set to the number of target classes.
  * Device Selection: The model is moved to GPU if available; otherwise, it remains on the CPU.
  * Loss Function: CrossEntropyLoss is used, which is suitable for multi-class classification problems.
  * Optimization Algorithm: The Adam optimizer is chosen to update the model's weights, with a learning rate of 0.001.


# **5- Testing the Model :** 
* This section focuses on training and evaluating the CNN model through iterative loops. 
* The train_model function handles the training process for one epoch, setting the model to training mode and iterating over the training data in batches. 
* For each batch, the inputs and labels are transferred to the selected device (GPU or CPU), the optimizer resets gradients, predictions are made, and the loss is calculated and backpropagated. 
* The optimizer then updates the model’s parameters, and the loss is accumulated to compute the average loss for the epoch. 
* The evaluate_model function assesses the model's performance on the test data by switching the model to evaluation mode, iterating over the test dataset without tracking gradients, and calculating the number of correct predictions for accuracy. 
* In the main loop, this training and evaluation process is repeated for a specified number of epochs. 
* After each epoch, the average training loss and test accuracy are logged and stored for performance tracking. 
* The loop concludes with a message indicating the end of training, ensuring the model is iteratively optimized and evaluated for its ability to generalize.


# **6- Manipulating images with different lights :**
 * Input: The function takes a list of images (images).
 * Process: For each image in the list, the function performs an intensity adjustment
 * using cv2.convertScaleAbs:
    * alpha=1.5: This is a scaling factor that multiplies the pixel values. It enhances the image's contrast by making the differences between pixel values more pronounced.
    * beta=25: This is an offset that adds a constant value to all pixel values. It brightens the image by increasing the intensity of all pixels.
    * These operations are applied to each image to adjust its brightness and contrast.
 * Output: The function returns a NumPy array of the manipulated images (manipulated_images), with the pixel values cast to np.float32 for consistency in image processing tasks.


# **7- Evaluation of the model with manipulated test set :**
* This section manipulates the test set and evaluates the model's performance on the altered images. 
* First, the get_manipulated_images function is used to adjust the brightness and contrast of the test set (X_test), resulting in the manipulated_test_images. 
* Then, a new dataset (manipulated_test_dataset) is created using the manipulated images and their corresponding labels (y_test), with the same transformation (transform) applied. 
* This dataset is loaded into a DataLoader (manipulated_test_loader) for efficient batching and processing. 
* The model's performance is then evaluated on this manipulated test set using the evaluate_model function, and the accuracy (manipulated_accuracy) is calculated. 
* Finally, the accuracy of the model on the manipulated test set is printed, showing how the model performs after the images have been altered.


# **8- Using color constancy algorithm on manipulated test set :**
This section of code applies the Gray World algorithm to correct the color balance of the manipulated test set, then plots the model’s accuracy over the training epochs.
* Applying Gray World Correction:

    * The function apply_gray_world(image) implements the Gray World algorithm for color correction. The algorithm works by adjusting the color balance of an image so that the average color (brightness) of the image is balanced across the red, green, and blue channels.
    * Step 1: cv2.mean(image)[:3] calculates the average intensity of the blue, green, and red channels in the image.
    * Step 2: The average intensity values of the blue, green, and red channels are used to compute a "gray value," which is the target average intensity for all channels.
    * Step 3: Scaling factors are calculated for each color channel by dividing the target gray value by the average intensity of that channel.
    * Step 4: The image is then adjusted by multiplying each pixel's value by the corresponding scaling factor. This step adjusts the image to match the desired average color balance.
    * Step 5: np.clip() ensures that pixel values stay within the valid range of [0, 255], and the image is converted back to an unsigned 8-bit integer format (np.uint8).
The apply_gray_world function is applied to each manipulated image in the manipulated_test_images list, resulting in the color-corrected images stored in X_test_corrected.


# **9-Testing the model with the color constancy applied test set :**
* Plotting Accuracy Over Epochs:

    * The x list contains the epoch numbers (from 1 to 20), and the y list contains the accuracy values from the accuracy_list.
    * plt.plot(x, y) creates a plot showing how the accuracy changes with each epoch during training.
    * plt.xlabel("Epoch") and plt.ylabel("Accuracy") label the x-axis and y-axis of the plot, respectively.
    * plt.show() displays the plot, allowing us to visualize the model’s performance improvement over time.

# **10- Comparing and reporting the success of different test sets :**
* This section aims to visualize the model’s accuracy on different datasets. 
* The accuracy rates for three different datasets, called “Original Data,” “Manipulated Data,” and “Color Consistency Applied Data,” are stored in a dictionary. 
* This dictionary is converted into a Pandas Series to create a bar chart. The chart shows the accuracy rate for each dataset on the y-axis. 
* This allows you to compare how the model’s performance changes on the original dataset compared to the manipulated and color consistent datasets.

# **Concusion :**
* In this project, a convolutional neural network (CNN) was designed and trained to classify animal images into specific categories. 
* Starting with data preprocessing, the dataset was carefully prepared by resizing the images, applying transformations, and splitting the data into training and test sets. 
* A custom AnimalDataset class and DataLoader were implemented to process the data effectively. 
* The CNN model was built using PyTorch, which included convolutional, pooling, and dropout layers to extract features and reduce overfitting. 
* The training and testing cycles optimized the model using cross-entropy loss and Adam optimizer, achieving accuracy improvement over multiple epochs.
* To evaluate the robustness of the model, processed datasets were created using contrast and brightness adjustments, and the model’s performance was evaluated on these modified datasets. 
* Additionally, a gray-world color consistency algorithm was applied to improve the processed data, resulting in improved accuracy compared to the uncorrected processed data. 
* Finally, a comparison of the model accuracies on the original, processed, and corrected datasets was visualized through a bar chart highlighting the model’s response to changes in data quality and preprocessing.
* Overall, the project demonstrated the importance of robust preprocessing, augmentation and correction techniques in improving model performance and robustness. The results highlight that data quality and consistency are critical to obtaining reliable and generalizable AI models.

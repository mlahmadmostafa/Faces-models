# This repo contains
- Face Landmark Detection Model (Done)
- Face bounding boxes (On work)
- Face identification (On work)
- 
# Landmark Detection Model
This project implements a convolutional neural network (CNN) for predicting facial landmarks using images from the CelebA dataset. The model is built with TensorFlow and Keras, designed to predict 10 landmarks for each face in the images.
![image](https://github.com/user-attachments/assets/a71168f8-61bd-471a-b65a-db900fca49fa)

### Model Overview

1. **Data Generator (`DataGen` class):**
   - The data generator is used to load and preprocess batches of images and corresponding landmarks.
   - Images are resized to 128x128 pixels, and landmark coordinates are normalized by dividing by the dimensions of the images (178x218).
   - The generator works in batches, which reduces memory usage and helps with training efficiency.

2. **Data Preprocessing:**
   - Landmark data is read from a CSV file and normalized for model input.
   - Each landmark's `x` and `y` coordinates are scaled to the range [0, 1], based on the image dimensions.

3. **Model Architecture (`CNN_LM` class):**
   - The model consists of three convolutional layers with ReLU activation, followed by max-pooling layers to reduce the spatial dimensions.
   - The final layer is a fully connected dense layer that outputs 10 values (representing the 10 landmarks).
   - The model is compiled with the Adam optimizer and mean squared error as the loss function, which is appropriate for regression tasks like landmark prediction.

4. **Training Process:**
   - The model is trained using the data generators for both training and validation data.
   - Callbacks like `ModelCheckpoint`, `EarlyStopping`, and `TensorBoard` are used to monitor training progress and prevent overfitting.

5. **Prediction:**
   - After training, the model predicts the landmark positions for unseen images.
   - The predicted landmarks are displayed alongside the ground truth landmarks to visualize the model’s performance.

6. **Visualization:**
   - A function (`plot_prediction`) is used to plot both the true and predicted landmarks on the images, providing a clear comparison.

### Evaluation and Usage

- After training, the model's predictions can be visualized using the `image_predictor` function, which loads a specific image, makes predictions, and displays both the predicted and actual landmark points.
- The best-performing model (based on validation loss) is saved and can be loaded for further predictions.

This model can be further enhanced by experimenting with different architectures, data augmentation techniques, and hyperparameters.

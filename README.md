# CIFAR-10 (CNN)
<span style="color:red">  In the code, note a section labeled "(DRAFT)," with commented cells. I initially intended to remove them as they represent early iterations, but I chose to retain them to check the outputs only.<span>

## Code Structure

 **1. Data Preprocessing:**

- The device has been set to utilize the Google Colab GPU for enhanced processing power.
- The dataset is loaded and subjected to the following transformations: resizing, sharpness adjustment, horizontal flipping, random rotation, and normalization.
- A batch size of 64 was chosen over 32 due to a slight improvement in performance, resulting in approximately a 2% higher accuracy. This choice does not significantly impact computational load, given the utilization of the T4 GPU.

**2. Model Building:**

- Initially, a straightforward model architecture has been employed, consisting of:
   - **Convolutional Layers:** Utilized for feature extraction, augmented by the ReLU activation function to introduce non-linearity.
   - **MaxPooling:** Applied after each convolutional layer to preserve essential information while reducing computational complexity.
   - **Fully Connected Layers:** Responsible for processing the flattened data and generating the model's output, mapping the condensed representation to 10 classes.
- The model was trained using **Cross-Entropy Loss**, a well-suited choice for multi-class classification tasks.
- The **Adam optimizer** was chosen with a learning rate of 0.001, following expert recommendations, and it remains adjustable for further fine-tuning.
   
 
**3. Training and Evaluation:**

 - **Training:** The model undergoes training for **10 epochs**, with a systematic approach of training and evaluation at each epoch to monitor its progress and identify potential signs of overfitting. *Evaluation metrics include Loss.*
 - **Evaluation Function:** This function helps assessing the model's performance on the test set deeply by providing the following:
   1) **Accuracy**
   2) **Classification Report:** provide a detailed breakdown of precision, recall, and F1-score for each class
   3) **Confusion Matrix:** to visualize the model's performance in terms of true positive, true negative, false positive, and false negative predictions.  

**4. Hyperparameter Tuning:**

 - **Exploring Different Architecture:** A more complex model (model2) has been employed to enhance feature extraction and classification performance compared to the previous model. This incorporates deeper and more complex structures, resulting in a notable 5% increase in accuracy.  
    -  Model Architecture inspired from the following [Reference](https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch)
- **Learning Rate Tuning:** The learning rate has been fine-tuned using the values [0.0001, 0.0003, 0.001, 0.003]. After careful evaluation, the learning rate of 0.001 outperformed the others in terms of model convergence and accuracy.

- **Number of Epochs:** Initially, an increased number of epochs showed improved performance. However, the model started to overfit, so the training was stopped at epoch 20 instead of 30.

- **Regularization:** Regularization has been introduced to push the number of epochs to 15 while preventing the model from overfitting. This strategy yielded an additional 2% accuracy without compromising generalization.
 
   

 
## Challenges / Solutions
**Limited Prior Experience with PyTorch**   
- I dedicated an hour to a [crash course](https://www.youtube.com/watch?v=OIenNRt2bjg) to become familiar with PyTorch's built-in functions and methods. This also included exploring PyTorch documentation for more in-depth understanding.

**Data Augmentation was providing a lower performance!!**
-  I revisited the [PyTorch documentation](https://pytorch.org/vision/0.10/auto_examples/plot_transforms.html#resize) and selected more effective techniques to improve data augmentation.

## Analysis & Future Improvement
For an averagely simple, not overly complex model *(as required)*, an expected accuracy range from 80% to 86% was anticipated based on a quick search. My model achieved a test accuracy of 81.95%, which is considered a satisfactory result. However, the model struggles to distinguish between similar classes, such as cats and dogs, as well as trucks and cars. It is evident that the model needs to capture more features, indicating the necessity for a more complex architecture.

**Areas for Improvement:**
- Eliminate code repetition by making the code object-oriented.
- Increase model complexity to better capture features.
- Augment data by adding more samples from classes with low F1 scores.
- Increase the number of epochs while optimizing regularization methods; consider adding dropout.

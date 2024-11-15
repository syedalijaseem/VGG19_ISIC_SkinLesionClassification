# Introduction

Skin cancer, especially melanoma, is one of the most common and aggressive cancers, where early detection can significantly boost survival rates. Accurately classifying skin lesions as benign or malignant is vital, but relying only on visual inspection often leads to errors. Deep learning has shown great promise in automating this process, although it typically requires large, labeled datasets to perform well, which can be difficult to obtain in specialized fields like dermatology.

Transfer learning addresses this challenge by using pre-trained models, often trained on large datasets like ImageNet, and adapts them for smaller, domain-specific datasets. With feature extraction, the pre-trained model serves as a fixed feature extractor, where only the classifier layer is retrained for tasks like skin cancer detection, helping to capture critical patterns even with limited data. Fine-tuning, on the other hand, adjusts the weights in the model’s later layers, enabling it to learn specific features such as differentiating between benign and malignant lesions to improve accuracy.

Together, feature extraction and fine-tuning make transfer learning a powerful tool for applying deep learning to medical image analysis, especially when labeled data is limited.

Beyond medical imaging, transfer learning has been applied in various ways across different computer vision tasks. For example:

- **Ahmad et al.** used incremental transfer learning for detecting deception, hitting 80% accuracy by combining physiological and oculomotor data. This approach also improved real-time performance, which is impressive given the limited computational resources involved [1].

- **Jonker et al.** took a different approach with image forgery detection, using fine-tuned EfficientNet and ResNet50V2 models that improved accuracy by about 15% and reached a final accuracy of 89.7% on unseen data. This work also addressed data-sharing and privacy constraints by using publicly available datasets [2].

- **Erol and Inkaya** used ensemble transfer learning with LSTM models in a sales forecasting setup, which boosted accuracy by 7-10% and cut down training time by up to 25%. They combined techniques like bagging and stacking to control overfitting and keep computational costs low, which is particularly useful in this context [3].

# Model Architecture

![VGG19 Architecture](/VGG19.png)

VGG19 is a deep convolutional neural network with 19 layers, including 16 convolutional layers and 3 fully connected layers. It uses 3x3 filters with a stride of 1 in the convolutional layers. This simple design helps the network capture spatial hierarchies in images while staying effective. Typically trained on large datasets like ImageNet, the network involves preprocessing steps like resizing input images to 224x224 pixels and normalizing pixel values. It also uses data augmentation techniques, such as random rotations and flips, to improve generalization.

The architecture, as shown in the attached image, is split into five blocks. Each block has convolutional layers followed by max-pooling layers, which reduce the spatial dimensions of the feature maps and make the network more efficient. The details of the blocks are as follows:

- The first block has two convolutional layers with 64 filters.
- The second block has two convolutional layers with 128 filters.
- The third block has four convolutional layers with 256 filters.
- The fourth block has four convolutional layers with 512 filters.
- The fifth block has four convolutional layers with 512 filters.

After the convolutional layers, the feature maps are flattened and passed through three fully connected layers, each with 4096 neurons. The final layer produces a 1000-class probability distribution using softmax.

The main idea behind VGG19 is the use of small 3x3 filters stacked deep, which helps the network learn more complex features while keeping things simple. The uniform structure, where each block uses the same filter size, makes it easy to understand and implement. Although the deep architecture helps with accuracy, it also means many parameters, which can be computationally heavy. Still, the simplicity and strong performance have made VGG19 a popular choice for transfer learning, where you can use pre-trained models for fine-tuning on other tasks.

VGG19 can be a great option for skin disease classification, and we can use it in two ways: feature extraction and fine-tuning. Both methods have their strengths and are suited for different scenarios depending on the amount of labeled data available and the task itself. Let’s dive into how VGG19 can be used for feature extraction first.

## Feature Extraction with VGG19

![VGG19 with feature extraction architecture](/Feature_Extraction.png)

Feature extraction involves using the pre-trained VGG19 model to extract useful features from input images, without changing the weights of the convolutional layers. The idea is to take advantage of the knowledge VGG19 has already gained by being trained on large datasets like ImageNet. The lower layers of the network, which detect basic features like edges, textures, and patterns, have learned representations that can be applied to new tasks, including chest X-ray image analysis.

To use VGG19 for feature extraction:

1. Remove the final classification layers (fully connected layers) of the network.
2. Keep the convolutional base, which consists of a sequence of convolutional and max-pooling layers, frozen, meaning its weights aren’t updated during training.
3. Pass the input image through the convolutional base to extract features. The output feature maps progress through deeper layers that capture increasingly complex features, eventually forming a feature map with a depth of 512.
4. Flatten the final feature map into a single vector, known as the "CNN feature vector."

The main advantage of this method is that it allows us to benefit from the general-purpose feature representations VGG19 has already learned without retraining the entire network. This is especially useful when working with small datasets or limited computational resources, as only the new classifier layers need to be trained, cutting down on the amount of data and computation required.

## Fine-Tuning with VGG19

![VGG19 with fine tuning architecture](/Fine_Tuning.png)

Fine-tuning is a more advanced technique, as it involves unfreezing some or all of the layers in the pre-trained VGG19 model and allowing their weights to be updated during training. This helps the model adjust its general features from ImageNet to better suit the specific characteristics of the dataset, such as skin disease images, making it more specialized for this new task. Fine-tuning is particularly helpful when the new dataset shares similarities with the original data (like ImageNet).

The fine-tuning process begins similarly to feature extraction:

1. Use the convolutional base to extract features.
2. Unlike in feature extraction, allow some layers of the convolutional base to be "trainable," meaning their weights are updated during training.
3. Typically, the initial layers, which capture general features like edges and textures, are kept frozen, while the later layers, which capture more complex, task-specific patterns, are fine-tuned.

In the image, we see:

- The convolutional layers (grey) extracting features and progressing through several blocks.
- A flatten layer (red) that reduces the spatial dimensions to create a 100352 x 1 feature map.

After pooling, fine-tuning adjusts the higher-level layers to improve performance on the new dataset. A dropout layer (blue) is added to prevent overfitting, followed by a fully connected layer (yellow) with ReLU activation. Finally, a softmax layer (orange) outputs the class probabilities, completing the fine-tuned network.

Fine-tuning works best when we have enough data for the new task, as it allows the model to adapt its learned features to the new dataset. Starting with robust pre-trained features helps avoid overfitting, while adjusting the higher layers improves the model’s ability to recognize subtle patterns in the new domain. The downside is that fine-tuning has a higher computational cost than feature extraction alone, as more layers are being updated. However, if sufficient data is available, fine-tuning can significantly improve accuracy, especially in tasks like skin disease classification, where recognizing subtle patterns is key.

## Conclusion

In short:

- **Feature extraction** is a quick way to adapt pre-trained models to new tasks with minimal data and computation.
- **Fine-tuning** requires more training but can lead to better results, especially when there’s enough data to make the adjustments.

Both methods take advantage of the powerful VGG19 model, but they differ in terms of data requirements, computational cost, and the potential to fine-tune the model for specific tasks like skin disease classification.

# Experiments and Results

## Experiment 1 (Feature Extraction and Fine-Tuning)

In the first experiment, two distinct models based on the VGG19 architecture were developed to classify images: one using feature extraction (Model 1) and the other utilizing fine-tuning (Model 2). Both models were trained for 10 epochs on the same dataset, allowing us to assess their performance across various metrics, including accuracy, precision, recall, and area under the curve (AUC).

### Feature Extraction Model (Model 1)

For the feature extraction model, the VGG19 layers were entirely frozen to leverage the pre-trained weights solely for feature extraction, while only the custom classification head was trainable. This approach allowed Model 1 to focus on classifying the extracted features without modifying the weights in the convolutional layers, preserving the general-purpose feature representations learned on the original ImageNet dataset.

#### Model 1 Architecture

| Model                        | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
| ---------------------------- | ---------------- | -------------------- | ------------------------ |
| Model 1 (Feature Extraction) | 20,024,385       | 2,101                | 20,022,284               |

During training, the model's performance was tracked in terms of both loss and accuracy for the training and validation datasets across each epoch. Model 1 initially showed relatively high validation accuracy due to the quality of the pre-trained feature extractor, achieving its best validation loss of 2.908 after epoch 9 and a final validation accuracy of 79.69%. However, because the VGG19 layers were frozen, the model’s learning capacity was constrained to the classification head alone, limiting its ability to adapt to finer distinctions in the dataset.

#### Training Performance for Model 1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| ----- | ------------- | --------------- | ----------------- | ------------------- |
| 1     | 0.75          | 3.2             | 82.50%            | 78.30%              |
| 5     | 0.56          | 2.95            | 84.20%            | 79.00%              |
| 9     | 0.54          | 2.908           | 85.00%            | 79.69%              |

While Model 1 achieved respectable accuracy levels, its limited flexibility in adjusting features resulted in a plateaued performance, as it could not fine-tune the deeper layers to capture dataset-specific nuances.

### Fine-Tuning Model (Model 2)

The fine-tuning model (Model 2) retained the VGG19 layers but allowed the last five layers to be trainable, striking a balance between leveraging pre-trained features and adapting these features to better fit the specific dataset. By unfreezing the last five layers, Model 2 was able to fine-tune these layers while preserving the knowledge in earlier layers, thereby enabling more effective learning of dataset-specific patterns.

#### Model 2 Architecture

| Model                 | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
| --------------------- | ---------------- | -------------------- | ------------------------ |
| Model 2 (Fine-Tuning) | 20,143,745       | 1,031,169            | 19,112,576               |

The training performance of Model 2 showed a clear improvement over Model 1, with the validation loss decreasing more rapidly and consistently. By epoch 3, Model 2’s validation loss reached 0.687, significantly lower than that of Model 1, and the model achieved a final validation accuracy of 80.47%. This outcome highlights the effectiveness of fine-tuning the deeper layers of the VGG19 model to capture subtle differences in the dataset that the feature extraction model could not.

#### Training Performance for Model 2

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| ----- | ------------- | --------------- | ----------------- | ------------------- |
| 1     | 0.68          | 1.5             | 83.00%            | 80.00%              |
| 3     | 0.47          | 0.687           | 85.50%            | 81.20%              |
| 10    | 0.44          | 0.55            | 87.30%            | 80.47%              |

Additionally, Model 2 outperformed Model 1 in precision and recall scores, demonstrating enhanced adaptability and generalization capabilities on the validation set. This model’s fine-tuned VGG19 layers allowed for effective feature differentiation that was specific to the given dataset, resulting in improved accuracy and lower validation loss.

### Performance Comparison

The table below provides a comparison of key performance metrics between Model 1 (Feature Extraction) and Model 2 (Fine-Tuning), highlighting the fine-tuning approach's advantages:

| Metric                    | Model 1 (Feature Extraction) | Model 2 (Fine-Tuning) |
| ------------------------- | ---------------------------- | --------------------- |
| Best Validation Loss      | 2.908                        | 0.687                 |
| Final Validation Accuracy | 79.69%                       | 80.47%                |
| Precision                 | Moderate                     | High                  |
| Recall                    | Moderate                     | High                  |
| AUC                       | Moderate                     | High                  |

# Training and Validation Performance Graphs

![Training and Validation Performance Graphs for Experiment 1](/Experiment_1.png)

The graphs illustrated above display the training and validation performance of both the feature extraction model (Model 1) and the fine-tuning model (Model 2) over 10 epochs in terms of accuracy and loss.

### 1. Feature Extraction - VGG19 Accuracy (Top Left)

- This graph presents the training and validation accuracy for Model 1, where only the classification head was trainable.
- The validation accuracy fluctuates over the epochs but stabilizes towards the end, achieving a final accuracy of around 79.69%.
- The initial high validation accuracy suggests that the pre-trained features were effective, although the model's limited capacity to adapt to dataset-specific features led to a plateau in performance.

### 2. Feature Extraction - VGG19 Loss (Top Right)

- This plot shows the training and validation loss for Model 1.
- The validation loss decreases consistently, with some fluctuations, reaching its best value of 2.908 at epoch 9.
- The consistent reduction in validation loss indicates the model’s ability to leverage the pre-trained features for classification while remaining constrained by the frozen VGG19 layers, limiting further adaptation to the dataset.

### 3. Fine-Tuning - VGG19 Accuracy (Bottom Left)

- The accuracy plot for Model 2 demonstrates the impact of allowing some of the VGG19 layers to be trainable.
- The validation accuracy varies more significantly between epochs compared to Model 1, reflecting the model’s flexibility in learning dataset-specific patterns.
- By the final epoch, the model achieves a slightly higher validation accuracy than Model 1, indicating improved adaptability and performance.

### 4. Fine-Tuning - VGG19 Loss (Bottom Right)

- The loss plot for Model 2 reveals a substantial decrease in validation loss, particularly within the first few epochs.
- This rapid reduction signifies that fine-tuning enabled the model to capture finer distinctions within the dataset, achieving a final validation loss of 0.687, much lower than that of Model 1.
- The lower loss confirms that fine-tuning resulted in a more effective model for the task.

### Summary

These graphs underscore the improved performance of the fine-tuning approach over feature extraction. By allowing specific layers of VGG19 to be trainable, Model 2 was able to achieve higher validation accuracy and lower loss, demonstrating greater flexibility and effectiveness in classifying the dataset accurately.

### Conclusion

The fine-tuning model demonstrated a better capacity for adaptation and generalization, achieving higher performance across metrics due to the flexibility of trainable VGG19 layers in addition to the classification head. The results indicate that fine-tuning is more effective for capturing dataset-specific nuances, enhancing the model's ability to accurately classify images compared to relying solely on feature extraction.

## Experiment 2: Model Improvements and Results

In Experiment 2, I aimed to improve the performance of both the feature extraction and fine-tuning models. This was achieved through several architectural and training modifications that enhanced the models' ability to classify images effectively. Below, I describe the changes made to each model and present their results.

### Feature Extraction Model (Model 1)

### Changes Made:

1. **Pre-trained Model**:
   - I continued using the pre-trained VGG19 model for feature extraction, keeping the weights frozen.
   - Only the final dense layer was trainable, allowing the model to use the pre-learned features from ImageNet without modifying the convolutional weights.
2. **Model Architecture**:
   - I maintained a simple architecture, with VGG19 layers followed by a dense classification layer.
   - The final layer was adjusted to match the output requirements for the specific dataset.
3. **Increased Training Duration**:
   - Compared to Experiment 1, I extended the training to 20 epochs (up from 10).
   - I included additional callbacks such as early stopping and learning rate scheduling to prevent overfitting and optimize convergence.

### Model Summary:

| Model              | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
| ------------------ | ---------------- | -------------------- | ------------------------ |
| Feature Extraction | 143,667,240      | 128,000              | 143,539,240              |

### Training Performance:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| ----- | ------------- | --------------- | ----------------- | ------------------- |
| 1     | 1.02          | 1.15            | 74%               | 72%                 |
| 2     | 0.85          | 1.05            | 80%               | 77%                 |
| 3     | 0.73          | 0.98            | 82%               | 80%                 |
| 4     | 0.65          | 0.91            | 84%               | 82%                 |
| 5     | 0.60          | 0.89            | 85%               | 85%                 |

- The feature extraction model showed steady improvement in both training and validation accuracy, reaching 85% accuracy on the validation set by epoch 5.
- The training loss consistently decreased, indicating that the model effectively learned from the pre-trained features.
- However, this model was limited by the frozen layers, preventing further adaptation to the dataset-specific features.

### Fine-Tuning Model (Model 2)

### Changes Made:

1. **Unfreezing Layers**:

   - I unfroze the last two layers of the pre-trained VGG19 model to fine-tune the weights during training.
   - This allowed the model to refine the lower-level features, making the model more adaptable to the specific dataset.

2. **Learning Rate Adjustment**:

   - I used a lower learning rate for the pre-trained layers to avoid distorting the valuable features learned from ImageNet, while the new classifier layer had a higher learning rate.

3. **Extended Training**:
   - Similar to the feature extraction model, I increased the training duration to 20 epochs and used early stopping to prevent overfitting.
   - This allowed the model to train for longer periods without degradation in performance.

### Model Summary:

| Model       | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
| ----------- | ---------------- | -------------------- | ------------------------ |
| Fine-Tuning | 143,667,240      | 130,000,000          | 13,667,240               |

### Training Performance:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| ----- | ------------- | --------------- | ----------------- | ------------------- |
| 1     | 1.23          | 1.40            | 70%               | 68%                 |
| 3     | 0.88          | 0.72            | 85%               | 81%                 |
| 5     | 0.55          | 0.45            | 88%               | 83%                 |
| 10    | 0.30          | 0.25            | 91%               | 87%                 |

- The fine-tuning model showed much faster improvement compared to the feature extraction model, particularly in validation loss and accuracy.
- By epoch 5, it achieved an 83% accuracy on the validation set, and by epoch 10, the accuracy increased to 87%.
- Fine-tuning allowed the model to better adapt to dataset-specific features, leading to a more precise classification.
- The addition of unfreezing the last two layers provided flexibility, enabling the model to learn finer details in the dataset while preserving the high-level features from the pre-trained layers.
- The learning rate adjustments helped ensure that the pre-trained features were not overfitted or destroyed during the fine-tuning process.

## Training and Validation Performance Graphs for Experiment 2

![Training and Validation Performance Graphs for Experiment 2](/Experiment_2.png)

The graphs illustrated below present the training and validation performance of the feature extraction model (Model 1) and fine-tuning model (Model 2) for Experiment 2, showing the results in terms of accuracy and loss over 9 epochs.

### 1. Feature Extraction - VGG19 Accuracy (Top Left)

- This plot shows that, in Experiment 2, both the training and validation accuracy for Model 1 improved steadily over the epochs.
- The model achieved a final validation accuracy close to 80%, higher than in Experiment 1.
- The upward trend in accuracy indicates that the extended training duration and architectural adjustments allowed the model to learn more effectively from the dataset, reaching a higher level of performance compared to Experiment 1.

### 2. Feature Extraction - VGG19 Loss (Top Right)

- The training and validation loss for Model 1 decreased substantially in Experiment 2, with validation loss stabilizing around a low value by epoch 4.
- This is a significant improvement over Experiment 1, where the validation loss remained relatively high and fluctuated.
- The lower and more stable loss in Experiment 2 suggests that the additional training epochs and early stopping mechanisms helped the model achieve better convergence, reflecting a more robust and effective learning process.

### 3. Fine-Tuning - VGG19 Accuracy (Bottom Left)

- For Model 2, the accuracy plot shows a marked improvement in Experiment 2, with validation accuracy reaching above 80% and demonstrating a clearer upward trend compared to Experiment 1.
- This improvement highlights the benefits of unfreezing additional layers and applying a lower learning rate to pre-trained layers, which allowed the model to better adapt to the dataset’s specific features.

### 4. Fine-Tuning - VGG19 Loss (Bottom Right)

- In Experiment 2, the validation loss for Model 2 decreased consistently over the epochs and stabilized at a much lower level compared to Experiment 1.
- This rapid decline in validation loss, reaching around 0.5 by the final epoch, indicates that the fine-tuning approach was more effective in this experiment, allowing the model to better capture complex patterns within the data.
- The reduced loss also suggests enhanced generalization, as the model achieved a lower and more stable loss on the validation set.

## Comparison with Experiment 1

Comparing the graphs of Experiment 2 with those of Experiment 1, several improvements are evident:

- **Higher Accuracy**:
  - Both models achieved higher validation accuracy in Experiment 2, with Model 1 reaching close to 80% and Model 2 exceeding this mark.
  - The improvements in accuracy are likely due to the architectural changes and extended training time, allowing both models to leverage dataset-specific features more effectively.
- **Lower Validation Loss**:
  - Both models showed a substantial reduction in validation loss in Experiment 2 compared to Experiment 1.
  - The validation loss for Model 1 dropped consistently and stabilized, while Model 2 achieved a much lower final loss, indicating improved model convergence and reduced overfitting.
- **Enhanced Stability**:
  - The loss and accuracy curves in Experiment 2 are generally smoother and more stable, especially in the case of the fine-tuning model.
  - This stability suggests that the adjustments in training strategy, including early stopping and learning rate scheduling, helped the models converge more efficiently.

## Comparative Results Between Experiment 1 and Experiment 2:

| Metric                        | Experiment 1 (Feature Extraction) | Experiment 2 (Feature Extraction) | Experiment 1 (Fine-Tuning) | Experiment 2 (Fine-Tuning) |
| ----------------------------- | --------------------------------- | --------------------------------- | -------------------------- | -------------------------- |
| **Best Validation Loss**      | 2.908                             | 0.89                              | 0.687                      | 0.25                       |
| **Final Validation Accuracy** | 79.69%                            | 85%                               | 80.47%                     | 87%                        |
| **Precision**                 | Moderate                          | High                              | Moderate                   | High                       |
| **Recall**                    | Moderate                          | High                              | Moderate                   | High                       |
| **AUC**                       | Moderate                          | High                              | Moderate                   | High                       |

## Conclusion

In conclusion, the results from Experiment 2 demonstrate significant improvements over Experiment 1 for both feature extraction and fine-tuning models. The changes implemented in Experiment 2 allowed both models to achieve higher accuracy, lower validation loss, and more stable performance, underscoring the effectiveness of these modifications.

# Summary and Discussion

In this study, two models based on the VGG19 architecture were developed and tested for image classification tasks: a feature extraction model and a fine-tuning model. Both models were evaluated across two experiments, with adjustments in training strategy and architecture for the second experiment, allowing us to analyze performance differences and identify the model best suited for the task.

## Best Performing Model

The fine-tuning model (Model 2) achieved the best results across both experiments. In Experiment 2, this model reached a validation accuracy of 87% and a validation loss of 0.25, outperforming the feature extraction model in terms of both accuracy and generalization. This improvement can be attributed to the fine-tuning approach, which allowed the model to adjust specific layers of the VGG19 architecture to better capture dataset-specific patterns. By unfreezing the last few layers of VGG19, the fine-tuning model was able to adapt pre-trained features more effectively, resulting in improved performance on the validation set.

## Reasons for Performance Differences

Several factors contributed to the differences in performance between the feature extraction and fine-tuning models:

1. **Adaptability to Dataset**:  
   The feature extraction model, which used frozen VGG19 layers, had limited flexibility. Its performance was constrained to the original ImageNet features, which may not fully capture the unique characteristics of the new dataset. In contrast, the fine-tuning model was able to refine features in the last few layers, allowing it to learn dataset-specific patterns and make more accurate classifications.

2. **Learning Capacity**:  
   The fine-tuning model had a higher capacity for learning by updating the weights of some convolutional layers, which allowed it to capture subtle distinctions between classes. This additional learning capacity helped the model achieve better accuracy and lower loss in Experiment 2.

3. **Training Duration and Stability**:  
   In Experiment 2, both models were trained for a longer period, with early stopping and learning rate scheduling. These adjustments helped the models converge more effectively and reduced overfitting. However, the fine-tuning model responded better to these changes, demonstrating a smoother learning curve and lower final validation loss.

4. **Trade-Off Between Flexibility and Generalization**:  
   While the feature extraction model exhibited some degree of generalization from the pre-trained features, it lacked the adaptability needed to fine-tune its representations. The fine-tuning model, though more flexible, was at a higher risk of overfitting; however, careful training adjustments mitigated this risk, allowing the model to generalize effectively on the validation set.

## Future Improvements

To further enhance model performance, the following strategies are planned:

1. **Layer-Wise Fine-Tuning**:  
   Instead of unfreezing only the last few layers, a gradual unfreezing strategy could be applied, where layers are progressively unfrozen as training advances. This could help improve feature learning without compromising generalization.

2. **Data Augmentation and Regularization**:  
   Adding more extensive data augmentation techniques, such as random rotations, flips, and color variations, may improve the model's robustness. Additionally, applying dropout and L2 regularization could help prevent overfitting, especially in the fine-tuning model.

3. **Model Optimization and Transfer Learning**:  
   Testing other pre-trained architectures, such as ResNet or EfficientNet, could yield better results due to their design optimizations and efficient parameter usage. Experimenting with different transfer learning models could help identify an architecture better suited to this specific dataset.

4. **Hyperparameter Tuning**:  
   A thorough grid search or Bayesian optimization of hyperparameters (learning rate, batch size, optimizer type, etc.) could further improve convergence speed and stability.

## Challenges and Limitations

1. **Reduction in Accuracy or High Loss**:  
   Both models showed fluctuations in accuracy and loss in some epochs, possibly due to insufficient learning rate tuning or mini-batch variance. Additionally, the feature extraction model’s performance plateaued quickly, as it was unable to learn new features beyond the pre-trained layers.

2. **Slow Training**:  
   Fine-tuning required more computation and time as additional layers were trainable, leading to slower training times. Optimizing the model’s pipeline or using mixed-precision training could alleviate this issue and speed up the process.

3. **Overfitting Risk**:  
   The flexibility of the fine-tuning model also presented a risk of overfitting, especially if training duration was extended. Regularization techniques, careful layer selection, and the early stopping mechanism were essential to manage this risk and prevent performance degradation.

## Conclusion

In summary, the fine-tuning model proved to be the most effective approach in this study, demonstrating improved adaptability and accuracy. Future work will focus on further enhancing the model’s generalization and efficiency through optimization and regularization strategies. This continued development will allow the model to achieve even better performance and maintain robustness on diverse datasets.

# References

1. M. I. A. A. A. M. S. Ahmad, "Detecting Deception in Natural Environments Using Incremental," in _ICMI '24: Proceedings of the 26th International Conference on Multimodal Interaction_, San Jose, 2024.

2. J. M. M. W. L. B. Jonker S, "Detecting Post Editing of Multimedia Images using Transfer," _ACM Transactions on Multimedia Computing, Communications and Applications_, vol. 20, no. 6, pp. 1-22, 2024.

3. I. T. Erol B, "Ensemble Deep Transfer Learning Approaches for Sales," in _ICACS '23: Proceedings of the 7th International Conference on Algorithms, Computing and Systems_, Larissa, 2023.

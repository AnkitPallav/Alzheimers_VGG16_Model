

# 🧠 Alzheimer's Disease Classification using VGG16

This project leverages transfer learning with the VGG16 convolutional neural network to classify MRI brain images into four categories of Alzheimer's disease: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The aim is to support early diagnosis and improve disease stage detection accuracy.

## 📁 Dataset

- **Source**: [Kaggle – Alzheimer's Dataset](https://www.kaggle.com/datasets/avrast/alzeihmer)
- **Classes**:
  - Non-Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented
- Data was split into `train/` and `test/` directories.

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- Google Colab
- NumPy, Matplotlib, Seaborn
- Scikit-learn

## 🏗️ Model Architecture

- **Base model**: VGG16 (pre-trained on ImageNet, top layers removed)
- **Custom layers**:
  - `GlobalAveragePooling2D`
  - Dense layer with 256 units and ReLU activation
  - Dropout (0.5)
  - Output Dense layer with Softmax (4 units)
- **Trainable Layers**: Last few convolutional blocks of VGG16

## 📈 Model Training

- Image data was rescaled and augmented (rotation, zoom, width/height shift, horizontal flip)
- Validation split: 20%
- Optimizer: Adam (`lr=1e-4`)
- Loss: Categorical Crossentropy
- Early stopping with patience of 5 epochs
- Class imbalance handled with `class_weight`

## ✅ Performance

| Metric              | Before Tuning | After Tuning |
|---------------------|---------------|--------------|
| Validation Accuracy | 60%           | 80%          |
| Test Accuracy       | ~80%          | -            |

- Classification report includes precision, recall, and F1-score for each class
- Confusion matrix analyzed for misclassification trends

## 🔧 Files Included

- `alzheimers_model_builder.py`: Contains model training code
- `evaluate_model.py`: Generates metrics and evaluation reports


## 🧪 Future Improvements

- Use more recent architectures like EfficientNet or ResNet50
- Experiment with Grad-CAM for interpretability
- Hyperparameter optimization using Keras Tuner
- Deploy as a web app using Streamlit or Flask

## 📜 License

This project is for academic and research purposes. Please cite the dataset and acknowledge authors if used in publications.

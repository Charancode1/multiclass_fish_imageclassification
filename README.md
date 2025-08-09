Brain Tumor Classification – CNN Model
This project classifies brain MRI scans into four categories:

Glioma

Meningioma

No Tumor

Pituitary

The model is built using TensorFlow/Keras and trained on a dataset of MRI images.
The preprocessing pipeline ensures all images are resized, normalized, and augmented before training.

📂 Dataset Structure
The dataset should be organized as:

kotlin
Copy
Edit
data/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   ├── pituitary/
│
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   ├── pituitary/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    ├── pituitary/
🛠 Requirements
Install the required Python libraries:

bash
Copy
Edit
pip install tensorflow numpy matplotlib scikit-learn
🚀 How to Run
Clone this repository or copy the code into your local environment.

Place the dataset inside the data folder following the structure above.

Run the script to train and evaluate the model:

bash
Copy
Edit
python brain_tumor_classification.py
The model will:

Preprocess images (resize to 224×224, normalize, augment).

Train using CNN architecture with early stopping.

Save the trained model as brain_tumor_model.h5.

Display training accuracy and loss graphs.

Evaluate on the test dataset and print classification metrics.

📊 Model Performance
The training script will output:

Accuracy & Loss curves

Final Test Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

⚙ Model Architecture
Input Layer: 224×224×3 images

Convolutional Layers: 3 layers with ReLU activation

Pooling Layers: MaxPooling2D after each conv block

Fully Connected Layers: Dense + Dropout

Output Layer: Softmax (4 classes)

📈 Example Output
Training Accuracy Graph

Validation Accuracy Graph

Confusion Matrix

Sample Predictions

🧠 Future Improvements
Use Transfer Learning with ResNet50, VGG16, or EfficientNet.

Add more augmentation techniques for robustness.

Implement Grad-CAM to visualize model focus.

📜 License
This project is for educational purposes only. Dataset rights belong to their respective owners.


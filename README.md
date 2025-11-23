Car Brand Classification Using VGG16 & ResNet50

This project is developed as part of Deep Learning – Assignment 2, GM University.

The goal is to build and compare two CNN architectures — VGG16 and ResNet50 — using a publicly available dataset of car brand images.

🚗 Dataset

Contains 26 car brands

Images resized to 224 × 224

Pixel values normalized

Data augmentation applied:

Rotation

Horizontal flip

Zoom

Shear

Rescale

🧠 Models Implemented
1️⃣ VGG16

Pretrained on ImageNet

Last layers replaced

Initially trained with base layers frozen

Later, deeper layers unfrozen for fine-tuning

Total training: 25 epochs

2️⃣ ResNet50

To be implemented next

Will use same hyperparameters as VGG16:

Epochs = 25

Batch size = 32

Optimizer = Adam

Learning rate = 1e-4

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion matrix

Accuracy graph

Loss graph

🔍 Comparative Analysis

After both models are trained, a comparison will be done to determine:

Which model performed better?

Why it performed better (architecture-based reasoning)?

Where each model struggles?

🗂 Project Structure
car_brand_classification/
 ├── dataset/              # Ignored from GitHub
 ├── vgg16_model.ipynb     # VGG16 training notebook
 ├── resnet_model.ipynb    # ResNet50 training notebook
 ├── vgg16_best.h5         # Best VGG16 model
 ├── vgg16_finetuned_best.h5
 ├── README.md
 └── .gitignore

🧑‍💻 How to Run the Project
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
jupyter notebook

🏁 Status

✔ VGG16 model completed

⏳ ResNet50 training pending (next task)
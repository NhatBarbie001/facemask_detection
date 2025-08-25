# Face Mask Detection 🩺😷

This project is a **deep learning application** that detects whether a person is wearing a mask or not, based on face images.  
It uses **Convolutional Neural Networks (CNNs)** with **Keras/TensorFlow**, and includes data preprocessing, augmentation, training, and inference.

---

## 📂 Project Structure
# facemask_detection

face-mask-detection/
│── code/
│ │── data_utils.py # Utility functions: load JSON annotations, gamma correction
│ │── model.py # CNN model definition
│ │── train.py # Training & inference script
│── README.md # Project description



---

## ⚙️ Requirements
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
We use the Face Mask Detection dataset:

annotations/ : JSON annotation files

images/ : Face images

train.csv : Training metadata

submission.csv: Submission template

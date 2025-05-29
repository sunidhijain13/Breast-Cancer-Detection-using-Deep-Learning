# 🧠 Breast Cancer Detection using Deep Learning

This project was developed as the **Final Year Bachelor of Engineering (B.E.) Project** for the **Computer Engineering Department at SITRC, Savitribai Phule Pune University**. It presents a deep learning-based system for the early and accurate classification of breast cancer tumors as **benign** or **malignant**, based on histopathology images.

This work has been **peer-reviewed and published** in the **Journal of Emerging Technologies and Innovative Research (JETIR)** in Volume 10, Issue 5 (May 2023).

---

## 📚 Research Publication

> **Title**: *Breast Cancer Classification and Detection using Deep Learning*  
> **Authors**: Sunidhi Jain, Kajal Patil, Komal Bhadane, Poonam Jadhav  
> **Published in**: [JETIR, May 2023, Volume 10 Issue 5](https://www.jetir.org/view?paper=JETIR2305636)  
> **ISSN**: 2349-5162 | UGC Approved Journal No: 63975  
> **Paper ID**: JETIR2305636

---

## 🗂️ Project Structure

```
breast-cancer-detection/
├── data/                         # Contains the BreakHis dataset
├── notebooks/                   # Jupyter notebook for training + evaluation
├── breast_cancer_classification.ipynb
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

Install requirements via:

```bash
pip install -r requirements.txt
```

Main libraries:
- TensorFlow / Keras
- scikit-learn
- OpenCV
- matplotlib / seaborn
- numpy / pandas

---

## 🧪 Dataset

- **Name**: BreakHis v1 (Breast Histopathology Images)
- **Source**: [Kaggle – ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis)
- **Classes**: `benign` and `malignant`
- **Preprocessing**:
  - Resized to 224x224
  - Normalized
  - 80/20 train-validation split
  - Flattened folder structure

---

## 🏗️ Model Architecture

- 3 Convolutional layers
- MaxPooling after each conv layer
- Dense layer with ReLU activation
- Final sigmoid layer for binary classification
- Optimizer: Adam
- Loss: Binary Crossentropy

---

## 📊 Results

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | ~95%      |
| Precision      | ~96%      |
| Recall         | ~93%      |
| F1 Score       | ~94%      |

Includes:
- Accuracy/Loss curves
- Confusion matrix
- Classification report

---

## 💾 Model Saving

```python
model.save("breast_cancer_cnn_model.keras")
```

---

## ✨ Future Improvements

- Implement transfer learning (e.g., ResNet50, InceptionV3)
- Deploy as a Streamlit or Flask app
- Add support for real-time image upload and prediction
- Add Grad-CAM for visual explanation

---

## 👩‍💻 Author

**Sunidhi Jain**  
 
[GitHub](https://github.com/sunidhijain13) • [LinkedIn](https://linkedin.com/in/sunidhijain13)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

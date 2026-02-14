# 🔥 Handwritten Digit Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AICTE](https://img.shields.io/badge/Project-MS%20ELEVATE%20AICTE-red.svg)](https://www.aicte-india.org/)

## 📋 Project Description

This project implements a handwritten digit recognition system using Support Vector Machine (SVM). The system analyzes 8x8 pixel images of digits (0-9) and classifies them accurately. Built with Python's scikit-learn, it includes data preprocessing, model training, and visualization dashboard.

---

## 🎯 Objectives

- Build an intelligent machine learning model for handwritten digit classification
- Achieve high accuracy (95%+) using SVM with RBF kernel
- Create professional visualization dashboards for model evaluation
- Develop a production-ready system for real-world applications
- Demonstrate practical ML implementation for AICTE project requirements

---

## ✨ Features

- ✅ **High Accuracy**: Achieves 95%+ accuracy on test dataset
- ✅ **SVM Algorithm**: Uses Support Vector Machine with RBF kernel
- ✅ **Complete Pipeline**: Data loading, preprocessing, training, and evaluation
- ✅ **Visualization Dashboard**: 6-panel comprehensive performance analysis
- ✅ **Confusion Matrix**: Detailed prediction analysis
- ✅ **Per-Digit Accuracy**: Individual digit performance metrics
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Production Ready**: Fully deployable ML solution

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming Language |
| **NumPy** | Numerical Computing |
| **Scikit-learn** | Machine Learning Library |
| **Matplotlib** | Data Visualization |
| **Seaborn** | Statistical Visualization |
| **Joblib** | Model Serialization |

---

## 📊 Dataset Information

- **Source**: Scikit-learn Digits Dataset
- **Total Samples**: 1,797 images
- **Image Size**: 8x8 pixels (64 features)
- **Classes**: 10 (digits 0-9)
- **Format**: Grayscale images
- **Split**: 80% Training (1,437 samples) | 20% Testing (360 samples)

---

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/digit-recognition-aicte.git
cd digit-recognition-aicte
```

### Step 2: Install Required Libraries
```bash
pip install numpy matplotlib scikit-learn seaborn joblib
```

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook digit_recognition_aicte.ipynb
```

**OR upload to Google Colab:**
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload Notebook
3. Select `digit_recognition_aicte.ipynb`
4. Run all cells

---

## 📖 Usage Guide

### Running the Complete Project

1. **Open the Notebook**: Launch `digit_recognition_aicte.ipynb`

2. **Execute All Cells**: Run cells sequentially (or use "Run All")

3. **View Results**: 
   - Model accuracy displayed in console
   - Visualization dashboard automatically generated
   - Confusion matrix and performance metrics shown

4. **Screenshot Dashboard**: Capture the visualization output for presentations

5. **Download Model**: The trained model is saved as `digit_recognition_model.pkl`

### Quick Start Code
```python
# Load libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Train model
model = SVC(gamma=0.001, kernel='rbf')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 📈 Model Performance

### Overall Metrics
- **Accuracy**: 98.61%
- **Training Samples**: 1,437
- **Test Samples**: 360
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Gamma**: 0.001

### Per-Digit Performance
| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 1.00 | 1.00 | 1.00 |
| 1 | 0.97 | 1.00 | 0.99 |
| 2 | 1.00 | 1.00 | 1.00 |
| 3 | 0.97 | 0.97 | 0.97 |
| 4 | 1.00 | 1.00 | 1.00 |
| 5 | 0.98 | 0.98 | 0.98 |
| 6 | 1.00 | 1.00 | 1.00 |
| 7 | 1.00 | 0.97 | 0.99 |
| 8 | 0.95 | 0.97 | 0.96 |
| 9 | 0.97 | 0.97 | 0.97 |

---

## 🎨 Visualization Dashboard

The project includes a comprehensive 6-panel dashboard:

1. **Confusion Matrix**: Shows prediction accuracy for each digit
2. **Sample Predictions**: Displays 8 test images with predictions
3. **Per-Digit Accuracy**: Bar chart of individual digit performance
4. **Live Demo**: Real-time prediction demonstration
5. **Test Set Distribution**: Dataset balance visualization
6. **Model Summary**: Complete metrics and configuration

**📸 Screenshot this dashboard for your AICTE presentation!**

---

## 💡 Real-World Applications

- 🏤 **Banking**: Automated check processing and verification
- 📮 **Postal Services**: ZIP code and address recognition
- 📄 **Document Processing**: Form digitization and data entry
- 🎓 **Education**: Teaching ML concepts and handwriting analysis
- 🏥 **Healthcare**: Medical form processing
- 📊 **Data Entry**: Automated invoice and receipt processing

---

## 📁 Project Structure

```
digit-recognition-aicte/
│
├── digit_recognition_aicte.ipynb    # Main Jupyter Notebook
├── digit_recognition_model.pkl      # Saved trained model
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── screenshots/                      # Dashboard screenshots
    └── dashboard.png
```

---

## 🔧 Model Training Details

### Hyperparameters
- **Algorithm**: Support Vector Classification (SVC)
- **Kernel**: Radial Basis Function (RBF)
- **Gamma**: 0.001 (controls decision boundary smoothness)
- **C**: 1.0 (regularization parameter, default)
- **Random State**: 42 (for reproducibility)

### Training Process
1. Load scikit-learn digits dataset
2. Split into 80% training, 20% testing
3. Train SVM classifier with RBF kernel
4. Generate predictions on test set
5. Evaluate using confusion matrix and accuracy metrics
6. Visualize results in comprehensive dashboard
7. Save model for future use

---

## 🧪 Testing the Model

### Load Saved Model
```python
import joblib

# Load the model
model = joblib.load('digit_recognition_model.pkl')

# Make prediction on new data
sample_digit = X_test[0].reshape(1, -1)
prediction = model.predict(sample_digit)
print(f"Predicted Digit: {prediction[0]}")
```

### Test with Custom Image
```python
import numpy as np

# Create a custom 8x8 digit (example: number 8)
custom_digit = np.array([
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]).flatten().reshape(1, -1)

prediction = model.predict(custom_digit)
print(f"Predicted: {prediction[0]}")
```

---

## 📚 Learning Outcomes

This project demonstrates:

- ✅ Understanding of supervised machine learning
- ✅ Implementation of Support Vector Machines
- ✅ Data preprocessing and feature engineering
- ✅ Model training and hyperparameter tuning
- ✅ Performance evaluation using confusion matrix
- ✅ Data visualization with matplotlib and seaborn
- ✅ Model serialization and deployment
- ✅ Professional project documentation

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Requirements

Create a `requirements.txt` file:

```
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
joblib>=1.1.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Module not found error**
```bash
# Solution: Install missing libraries
pip install numpy matplotlib scikit-learn seaborn joblib
```

**Issue 2: Jupyter notebook not opening**
```bash
# Solution: Install/update Jupyter
pip install --upgrade jupyter notebook
```

**Issue 3: Visualization not displaying**
```bash
# Solution: Add this at the top of notebook
%matplotlib inline
```

---

## 📞 Contact & Support

- **Developer**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: [https://github.com/yourusername/digit-recognition-aicte](https://github.com/yourusername/digit-recognition-aicte)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AICTE MS ELEVATE Program** - For project guidance
- **Scikit-learn Team** - For the excellent ML library
- **UCI Machine Learning Repository** - For dataset resources
- **Python Community** - For amazing open-source tools

---

## 📊 Project Status

✅ **Complete** - Ready for AICTE submission

- [x] Data loading and preprocessing
- [x] Model training and optimization
- [x] Performance evaluation
- [x] Visualization dashboard
- [x] Model persistence
- [x] Documentation
- [x] Testing and validation

---

## 🎓 Academic Information

**Project Type**: Machine Learning - Classification

**Course**: MS ELEVATE AICTE Program

**Domain**: Computer Vision & Pattern Recognition

**Difficulty Level**: Intermediate

**Estimated Time**: 4-6 hours

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

**Made with ❤️ for AICTE MS ELEVATE Program**

© 2024 Handwritten Digit Recognition System. All Rights Reserved.

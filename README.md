# 🍷 Red Wine Quality Classification Using PyTorch
A PyTorch neural network classification project to predict red wine quality using physicochemical features from a CSV dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A deep learning classification project built using **PyTorch** to predict the quality of red wine based on physicochemical features such as acidity, sugar, chlorides, density, pH, sulphates, and alcohol.

---

## 📌 Project Overview

This project demonstrates a complete **CSV-based neural network classification pipeline** using PyTorch.
The model takes wine chemical properties as input and predicts the wine quality class.

---

## 🎯 Objective

The main objective of this project is to build a neural network model that can classify red wine quality using a structured CSV dataset.
---

## 📂 Dataset

The dataset used in this project is:
```text
winequality-red.csv
```

### Target Column

```text
quality
```

### Input Features

The model uses the following 11 input features:

| Feature |
|---|
| fixed acidity |
| volatile acidity |
| citric acid |
| residual sugar |
| chlorides |
| free sulfur dioxide |
| total sulfur dioxide |
| density |
| pH |
| sulphates |
| alcohol |

---

## 🧠 Project Workflow

```text
CSV Dataset
    ↓
Data Loading using Pandas
    ↓
Feature and Target Separation
    ↓
Label Encoding
    ↓
Train-Test Split
    ↓
Feature Scaling
    ↓
Tensor Conversion
    ↓
PyTorch DataLoader
    ↓
Neural Network Model
    ↓
Training Loop
    ↓
Accuracy Evaluation
```

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch

---

## 🏗️ Model Architecture

The neural network architecture used in this project:

```text
Input Layer: 11 features
Hidden Layer 1: 64 neurons + ReLU
Hidden Layer 2: 32 neurons + ReLU
Output Layer: 6 classes
```

### Architecture Flow

```text
11 → 64 → 32 → 6
```

---

## 📊 Classification Classes

The original wine quality values are:

```text
3, 4, 5, 6, 7, 8
```
These are encoded into:

```text
0, 1, 2, 3, 4, 5
```

This encoding is required because PyTorch `CrossEntropyLoss` expects class labels starting from `0`.

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/Kajal805-M/Red-Wine-Calssification-Using-PyTorch.git
```

### 2. Move into the Project Folder

```bash
cd Red-Wine-Calssification-Using-PyTorch
```

### 3. Create Virtual Environment

```bash
python -m venv myenv
```

### 4. Activate Virtual Environment

For Windows PowerShell:

```bash
myenv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Training File

```bash
python train.py
```

---

## 📈 Output

The project displays:

- Dataset shape
- Missing values
- Input features
- Number of classes
- Model architecture
- Training loss
- Test accuracy
- Classification report

Example output:

```text
Epoch [10/100], Loss: 1.2456
Epoch [20/100], Loss: 1.1023
Epoch [30/100], Loss: 0.9821

Test Accuracy:
##62.50%
```

---

## 📌 Why Feature Scaling?

Feature scaling is important because the dataset contains features with different ranges.

For example:

- `alcohol`
- `density`
- `sulphates`
- `total sulfur dioxide`

These features have different value ranges, so `StandardScaler` is used to bring them to a similar scale.

---

## 📌 Why PyTorch?

PyTorch is used because it provides:

- Easy tensor operations
- Simple neural network creation
- Automatic gradient calculation
- Flexible training loop
- Strong support for deep learning experiments

---

## 📌 Why CrossEntropyLoss?

This is a multi-class classification problem.

Therefore, the project uses:

```python
nn.CrossEntropyLoss()
```

It is suitable for classification tasks where the model predicts one class out of multiple classes.

---

## 📁 Project Structure

```text
Red-Wine-Calssification-Using-PyTorch/
│
├── winequality-red.csv
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Important Concepts Covered

This project covers:

- CSV data loading
- Data preprocessing
- Train-test split
- Feature scaling
- Label encoding
- Tensor conversion
- PyTorch Dataset and DataLoader
- Neural network creation
- Training loop
- Loss calculation
- Backpropagation
- Model evaluation
- Accuracy calculation

---

## 🧾 Requirements

```text
pandas
numpy
scikit-learn
torch
```

---

## 🧠 Learning Outcome

After completing this project, you will understand how to build a complete PyTorch classification pipeline using a real-world CSV dataset.

You will learn how data is processed, converted into tensors, passed through a neural network, trained using backpropagation, and evaluated using accuracy.

---

## 📌 Future Improvements

- Add confusion matrix
- Add training loss visualization
- Handle class imbalance
- Try deeper neural networks
- Add dropout layers
- Add model saving and loading
- Deploy using Streamlit

---

## 👩‍💻 Author

**Kajal Maurya**

GitHub: [Kajal805-M](https://github.com/Kajal805-M)

---

## ⭐ Support

If you found this project helpful, give it a ⭐ on GitHub.

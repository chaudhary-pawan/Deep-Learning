# Deep Learning Concepts & Implementations

This repository provides a comprehensive learning path in **Deep Learning**, ranging from fundamental algorithms like the **Perceptron** to modern **Neural Network implementations** using TensorFlow/Keras.  
It is designed for both beginners exploring the basics and practitioners seeking practical examples.

---

## 📂 Repository Overview
The repository focuses on:
- Implementing deep learning concepts in a structured manner
- Covering both theoretical foundations and hands-on coding
- Providing step-by-step guidance for setup and execution

---

## 🚀 Projects

### 1. Customer Churn Prediction
- **Goal:** Predict whether a customer will churn (leave) or stay.  
- **Approach:**  
  - Implemented using **TensorFlow/Keras**  
  - Feed-forward neural network for **binary classification**  
  - Achieved **79.25% accuracy**  
- **Concepts Covered:**  
  - Data preprocessing  
  - Model architecture design  
  - Training, validation, and evaluation  

---



### 2. Handwritten Digit Classification using ANN
- **Goal:** Classified handwritten digits using deep learning techniques.
- **Approach:**  
  - Implemented using TensorFlow/Keras  
  - Utilizes the MNIST dataset for training and evaluation  
  - Employs a Artificial neural network (ANN) architecture  
  - Achieved high accuracy on test data  
- **Concepts Covered:**  
  - Image preprocessing (using configuring pixel data for each of 60K images  
  - ANN model design  
  - Training, validation, and evaluation
 
---

### 3. Graduate Admission Prediction Using ANN

This project demonstrates a neural network-based approach to predict graduate admissions using an Artificial Neural Network (ANN).  
It includes data preprocessing, model building, training, and evaluation steps.

**Key Features:**
- Input features: GRE, TOEFL, university rating, SOP, LOR, CGPA, research experience
- Output: Probability of admission
- Model: Multi-layer ANN using Keras/TensorFlow
- Visualization of training and test results

You can find the full implementation in the file:  
`Graduate Admission Predction using ANN.ipynb`  

**Usage:**
1. Open the notebook and follow the instructions to preprocess data and train the model.
2. Adjust model parameters as needed for experimentation.
3. Evaluate predictions and visualize results.

---

## Course Practicals📑

### 1. Perceptron Algorithm (From Scratch)
- **Goal:** Demonstrate the fundamentals of neural networks.  
- **Approach:**  
  - Implemented **Perceptron algorithm** from scratch in Python  
  - Explains how neurons learn through weight updates  
- **Concepts Covered:**  
  - Perceptron learning rule  
  - Binary classification  
  - Visualization of decision boundaries  

---

##4 Dropout Classification Example

This section demonstrates a binary classification task using a neural network in TensorFlow/Keras. The dataset consists of 2D points, each labeled as either 0 or 1. The notebook guides you through visualizing the data, building a neural network model, and training it to distinguish between the two classes.

**Workflow Overview:**
1. **Data Preparation:**  
   - 2D points (`X`) and binary labels (`y`) are loaded as numpy arrays.
   - The data is visualized using a scatter plot to show class separation.

2. **Model Construction:**  
   - A simple feedforward neural network is built using Keras `Sequential` API.
   - The model consists of two hidden layers (128 units each, ReLU activation) and an output layer with sigmoid activation for binary classification.
   - Example:
     ```python
     model = Sequential()
     model.add(Dense(128, input_dim=2, activation="relu"))
     model.add(Dense(128, activation="relu"))
     model.add(Dense(1, activation="sigmoid"))
     ```

3. **Training:**  
   - The model is compiled with binary cross-entropy loss and Adam optimizer.
   - Training is performed for 500 epochs with validation split to monitor accuracy and loss.

4. **Visualization:**  
   - The training process and data separation can be visualized for better understanding.

**To run this example:**
- Access the full notebook here: [dropout_classification_example.ipynb](https://github.com/chaudhary-pawan/Deep-Learning/blob/main/dropout_classification_example.ipynb)
- Try it interactively in Google Colab:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/dropout_classification_example.ipynb)

This notebook provides a hands-on introduction to neural network classification and is a great starting point for experimenting with dropout and other regularization techniques.


---


### 5. Backpropagation for Classification

**File:** [`backpropagation_classification.ipynb`](backpropagation_classification.ipynb)

This notebook demonstrates how to implement backpropagation for a simple binary classification problem using a neural network. You can run this notebook interactively using [Google Colab](https://colab.research.google.com/), Jupyter Notebook, or any compatible environment.


---

### 6. Backpropagation for Regression

**File:** [`backpropagation_regression.ipynb`](backpropagation_regression.ipynb)

This notebook provides an implementation of backpropagation for a regression task. It focuses on predicting continuous values using a neural network and includes step-by-step explanations.


**Quick Start:**
---python
# Open and run all cells in backpropagation_classification.ipynb
# It uses numpy and pandas for data handling and neural network calculations.
---
---

#### Running in Google Colab

You may run either notebook in Colab by clicking the "Open in Colab" badge at the top of each notebook, or by uploading it directly to https://colab.research.google.com/.

#### Requirements

- Python 3.x
- numpy
- pandas 

Install requirements (if running locally):
```bash
pip install numpy pandas
```

#### Tips

- Read the markdown cells for explanations and theoretical background.
- Modify the data or parameters to experiment with different scenarios.
- Each notebook is self-contained and can be run independently.

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/chaudhary-pawan/Deep-Learning.git
   cd Deep-Learning



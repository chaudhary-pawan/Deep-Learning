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

 
**To run this example:**
- Access the full notebook here: [customer_churn_prediction.ipynb](https://github.com/chaudhary-pawan/Deep-Learning/blob/main/customer_churn_prediction.ipynb)
- Try it interactively in Google Colab:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/customer_churn_prediction.ipynb)

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
 
  
**To run this example:**
- Access the full notebook here: [Handwritten_Digit_Classification_using_ANN.ipynb](https://github.com/chaudhary-pawan/Deep-Learning/blob/main/Handwritten_Digit_Classification_using_ANN.ipynb)
- Try it interactively in Google Colab:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/Handwritten_Digit_Classification_using_ANN.ipynb)

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


**To run this example:**
- Access the full notebook here: [Graduate_admission_prediction_using_ANN.ipynb](https://github.com/chaudhary-pawan/Deep-Learning/blob/main/Graduate_admission_prediction_using_ANN.ipynb)
- Try it interactively in Google Colab:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/Graduate_admission_prediction_using_ANN.ipynb)
  
---

### 4. Next word predictor using LSTM

Next Word Predictor using LSTM is a small, educational project that demonstrates how to build a simple next-word prediction model with TensorFlow / Keras. The project walks through preprocessing text, creating training sequences, building an LSTM-based neural network, training the model, and using it to predict the next word(s) given a short seed phrase.

This repository contains a Jupyter Notebook (Next_word_predictor_using_LSTM.ipynb) that implements a minimal but complete pipeline intended for learning and experimentation — not production use.

** Key ideas and components **
- Data: A plain-text FAQ excerpt is used as the training corpus inside the notebook (tokenized via Keras Tokenizer).
- Preprocessing: Text is lowercased and tokenized; training examples are created by sliding-window sequence generation (prefixes mapped to a following word).
- Model architecture: Embedding layer → LSTM layer(s) → Dense output with softmax over the vocabulary.
- Training: Categorical cross-entropy loss and an optimizer (e.g., Adam) train the model to predict the next token in the sequence.
- Inference: A seed text is tokenized and passed to the model to predict the next token; repeated prediction can generate multiple words.

** Who is this for **
- Beginners learning practical NLP preprocessing and sequence modeling.
- Students exploring how recurrent models (LSTM) learn local language patterns.
- Anyone wanting a small, runnable example to extend for language modeling, text generation, or autocomplete prototypes.

**To run this example:**
- Access the full notebook here: [Next_word_predictor_using_LSTM.ipynb](https://github.com/chaudhary-pawan/Deep-Learning/blob/main/Next_word_predictor_using_LSTM.ipynb)
- Try it interactively in Google Colab:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/Next_word_predictor_using_LSTM.ipynb)


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


### 2. Backpropagation for Classification

**File:** [`backpropagation_classification.ipynb`](backpropagation_classification.ipynb)

This notebook demonstrates how to implement backpropagation for a simple binary classification problem using a neural network. You can run this notebook interactively using [Google Colab](https://colab.research.google.com/), Jupyter Notebook, or any compatible environment.


---

### 3. Backpropagation for Regression

**File:** [`backpropagation_regression.ipynb`](backpropagation_regression.ipynb)

This notebook provides an implementation of backpropagation for a regression task. It focuses on predicting continuous values using a neural network and includes step-by-step explanations.

---


### 4. Dropout Classification Example

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

### 5. Zero Initialization with Sigmoid Neural Network (`zero_initialization_sigmoid.ipynb`)

This notebook demonstrates the impact of weight initialization in neural networks, specifically focusing on **zero initialization** with sigmoid activations. Using a simple 2D U-shape dataset, it walks through the following steps:

- **Data Loading & Visualization**: Loads a sample dataset (`ushape.csv`) with two input features and a binary class, then visualizes the data distribution.
- **Model Construction**: Builds a feedforward neural network using Keras, with one hidden layer of 10 neurons and sigmoid activation functions.
- **Zero Initialization**: Manually sets all model weights and biases to zero before training, overriding Keras's default random initialization.
- **Training & Evaluation**: Trains the network on the dataset using binary cross-entropy loss and Adam optimizer. Demonstrates the poor performance resulting from zero initialization (accuracy remains at 50%).
- **Decision Boundary Plotting**: Uses `mlxtend` to visualize the decision regions learned by the network, showing lack of learning due to symmetric initialization.

**Key Learning Point:**  
Zero initialization causes all neurons in each layer to learn the same features, preventing the network from breaking symmetry and learning meaningful patterns. This notebook serves as a hands-on illustration of why proper weight initialization is crucial for neural network training.

**Dependencies:**  
- Python (with numpy, pandas, matplotlib, tensorflow, keras, mlxtend)
- Data file: `ushape.csv` (should be present in the working directory)

---

### 6. Zero Initialization with ReLU Notebook

This notebook demonstrates the concept of initializing neural network weights to zero (or a constant value) and visualizes its effect on learning using a simple dataset.

**File:** [`zero_initialization_relu.ipynb`](zero_initialization_relu.ipynb)

### Key Highlights
- Loads and visualizes a U-shaped dataset for binary classification.
- Builds a simple neural network using Keras with two dense layers.
- Manually sets the weights of the model to a constant value (e.g., 0.5) before training.
- Trains the model for 100 epochs and observes the accuracy and loss.
- Plots the decision boundary learned by the model.
- Illustrates the impact of weight initialization on training dynamics.


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/zero_initialization_relu.ipynb)

---

**Usage:**  
Open the notebook in Jupyter or Google Colab, ensuring the data file is available, and follow the steps to observe the effects of zero initialization on model training.
**Quick Start:**
---python
#### Open and run all cells in backpropagation_classification.ipynb
#### It uses numpy and pandas for data handling and neural network calculations.
---
---

#### Running in Google Colab

You may run either notebook in Colab by clicking the "Open in Colab" badge at the top of each notebook, or by uploading it directly to https://colab.research.google.com/.

#### Requirements

- Python 3.x
- numpy
- pandas 


```

### 7. Keras padding in CNN architecture demo — Short summary

This notebook demonstrates how Keras Conv2D padding and stride choices affect spatial dimensions, parameter counts, and overall model size using the MNIST dataset. It loads MNIST and builds two simple Sequential CNNs to compare behaviors: Model A uses three Conv2D layers with `padding='valid'`, which reduces spatial size each layer; Model B uses `padding='same'` with `strides=(2,2)` to downsample while preserving kernel-centered outputs when stride=1. Each model ends with `Flatten` and `Dense` layers so `model.summary()` highlights differences in output shapes and total/trainable parameters. The example shows that `valid` padding shrinks feature maps by `kernel_size - 1` per layer dimension, while `same` preserves spatial dimensions (subject to strides). Using strides for controlled downsampling reduces the flattened vector size and drastically lowers Dense parameters, explaining why early downsampling, pooling, or global pooling is often preferred over huge fully connected layers. The notebook is runnable in Colab or locally with TensorFlow 2.x and Jupyter (Python 3.8+, `tensorflow>=2.6`). A Keras warning about passing `input_shape` to Conv2D in `Sequential` may appear but is informational. The notebook is concise, aimed at learners exploring CNN design choices; the principles generalize beyond MNIST.


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/Keras_padding_in_CNN_architecture_demo.ipynb)

### 8. Using ResNet50 pretrained model to classify an image

This notebook demonstrates using a pretrained ResNet50 model (trained on ImageNet) to classify a single image. It shows how to load the model, prepare an input image, run inference, and decode the top predictions.

#### What this notebook contains
- Loads Keras' ResNet50 model with ImageNet weights.
- Demonstrates image preprocessing (resizing to 224×224, converting to array, applying `preprocess_input`).
- Runs model inference and decodes the top-3 predictions with `decode_predictions`.
- Example included: classifying an image saved as `/content/chair.jpg` which yields chair-related classes.

#### Key files / cells
- Cell: import required modules (tensorflow.keras.applications.resnet50, image preprocessing, numpy).
- Cell: `model = ResNet50(weights='imagenet')` — downloads and loads pretrained weights.
- Cell: load & preprocess an image (`image.load_img`, `image.img_to_array`, `np.expand_dims`, `preprocess_input`).
- Cell: `model.predict(x)` and `decode_predictions(preds, top=3)` to print results.

#### Example output
Predicted: [
  ('n02791124', 'barber_chair', 0.9978),
  ('n04099969', 'rocking_chair', 0.0006),
  ('n03376595', 'folding_chair', 0.0005)
]

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/ResNet50(pretrained_model).ipynb)

#### Credits
- Uses Keras Applications ResNet50 and ImageNet pretrained weights provided by TensorFlow / Keras.

#### Tips

- Read the markdown cells for explanations and theoretical background.
- Modify the data or parameters to experiment with different scenarios.
- Each notebook is self-contained and can be run independently.

---

### 9. Deep RNNs

This notebook demonstrates building, comparing, and training simple recurrent neural networks on the IMDb movie-review sentiment dataset. It is intended as an educational example to show how SimpleRNN, LSTM and GRU layers behave on a short text classification task.

## Highlights
- Dataset: IMDb (binary sentiment classification), using the top 10,000 words.
- Preprocessing: sequences are padded to fixed length (maxlen = 100).
- Embedding: a Keras Embedding layer (vocab_size=10000, embed_dim=32) converts token ids to vectors.
- Models shown:
  - Stacked SimpleRNN (two SimpleRNN layers)
  - Stacked LSTM (two LSTM layers)
  - Stacked GRU (two GRU layers)
- Training setup:
  - Optimizer: Adam
  - Loss: binary_crossentropy
  - Metric: accuracy
  - Example training in the notebook: epochs=5, batch_size=32, validation_split=0.2
- Example results (from a sample run): training accuracy reached >95% while validation accuracy was around ~80% — results will vary depending on environment and random seed.


## Notes & tips
- SimpleRNN is useful for demonstrations, but LSTM/GRU typically handle long-range dependencies better (less vanishing gradient).
- The notebook uses very small recurrent unit sizes for clarity. For production or serious experiments, increase units, add dropout, and consider pretrained embeddings (GloVe, FastText) or transformer-based models for improved accuracy.
- To prevent overfitting: use dropout/recurrent_dropout, early stopping with validation loss, or reduce network capacity.
- Results are nondeterministic across runs; set random seeds if reproducibility is required.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaudhary-pawan/Deep-Learning/blob/main/Deep_RNNs.ipynb)

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/chaudhary-pawan/Deep-Learning.git
   cd Deep-Learning



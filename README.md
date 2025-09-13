# Deep Learning Repository

This repository contains deep learning projects and implementations focusing on fundamental concepts and practical applications. The collection includes neural network implementations using both modern frameworks and from-scratch implementations to provide a comprehensive understanding of deep learning principles.

## 🎯 Repository Overview

This repository serves as a practical guide to deep learning, covering fundamental algorithms and real-world applications. Each project is designed to demonstrate key concepts with hands-on implementations that bridge theory and practice.

### 📂 Projects and Topics Covered

#### 1. **Customer Churn Prediction** (`customer-churn-prediction.ipynb`)
- **Topic**: Binary classification using neural networks
- **Framework**: TensorFlow/Keras
- **Concepts**: 
  - Feed-forward neural networks
  - Binary classification
  - Data preprocessing and feature engineering
  - Model evaluation and performance metrics
- **Dataset**: Credit card customer churn data (Kaggle)
- **Architecture**: 2-layer neural network with sigmoid activation
- **Accuracy**: ~79.25% on test data

#### 2. **Perceptron Algorithm** (`Perceptron-trick.ipynb`)
- **Topic**: Fundamental building block of neural networks
- **Implementation**: From-scratch Python implementation
- **Concepts**:
  - Perceptron learning algorithm
  - Linear separability
  - Weight updates and convergence
  - Binary classification fundamentals
- **Visualization**: Matplotlib scatter plots for data visualization

## 🚀 Getting Started

### Prerequisites

- **Python Version**: 3.11+ (Python 3.11.13 or 3.12.4 recommended)
- **Environment**: Jupyter Notebook or JupyterLab
- **Hardware**: CPU-based implementation (no GPU required)

### Dependencies

Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

Or using conda:

```bash
conda install numpy pandas scikit-learn tensorflow matplotlib
```

#### Core Libraries:
- **NumPy** (>=1.21.0) - Numerical computing
- **Pandas** (>=1.3.0) - Data manipulation and analysis
- **Scikit-learn** (>=1.0.0) - Machine learning utilities
- **TensorFlow** (>=2.8.0) - Deep learning framework
- **Matplotlib** (>=3.5.0) - Data visualization

### Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chaudhary-pawan/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Set up Python environment** (recommended):
   ```bash
   python -m venv deep_learning_env
   source deep_learning_env/bin/activate  # On Windows: deep_learning_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # If available
   # or install manually:
   pip install numpy pandas scikit-learn tensorflow matplotlib jupyter
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## 🔧 Running the Projects

### Customer Churn Prediction

1. Open `customer-churn-prediction.ipynb`
2. Ensure you have the dataset or modify the data loading path
3. Run all cells sequentially
4. The notebook will:
   - Load and preprocess the data
   - Create a neural network model
   - Train the model for 10 epochs
   - Evaluate model performance

**Note**: The notebook references Kaggle dataset paths. You may need to:
- Download the dataset from [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
- Update file paths in the data loading section

### Perceptron Algorithm

1. Open `Perceptron-trick.ipynb`
2. Run all cells to see the perceptron implementation
3. The notebook demonstrates:
   - Synthetic data generation
   - Perceptron algorithm from scratch
   - Visualization of classification results

**Note**: The current implementation has some incomplete sections that may need debugging.

## 📊 Key Concepts Demonstrated

### Neural Networks Fundamentals
- **Feed-forward architecture**: Basic neural network structure
- **Activation functions**: Sigmoid activation for binary classification
- **Loss functions**: Binary cross-entropy for classification
- **Optimization**: Adam optimizer for gradient descent

### Machine Learning Pipeline
- **Data preprocessing**: Feature scaling and normalization
- **Train-test split**: Proper data splitting for evaluation
- **Model evaluation**: Accuracy metrics and performance assessment
- **Feature engineering**: One-hot encoding for categorical variables

### Algorithmic Understanding
- **Perceptron learning rule**: Weight update mechanism
- **Linear separability**: Understanding decision boundaries
- **Convergence criteria**: Algorithm stopping conditions

## 📈 Future Enhancements

Potential areas for expansion:
- **Convolutional Neural Networks (CNNs)** for image classification
- **Recurrent Neural Networks (RNNs/LSTMs)** for sequence data
- **Advanced architectures**: ResNet, Transformer models
- **Regularization techniques**: Dropout, batch normalization
- **Hyperparameter tuning**: Grid search and optimization
- **More diverse datasets**: Computer vision, NLP applications

## 🤝 Contributing

We welcome contributions to enhance this learning repository! Here's how you can contribute:

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-algorithm`
3. **Add your implementation** with proper documentation
4. **Include example usage** and expected outputs
5. **Submit a pull request** with detailed description

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 style guidelines
- **Documentation**: Include docstrings and markdown explanations
- **Testing**: Add simple test cases or validation examples
- **Educational Value**: Focus on learning and understanding
- **Reproducibility**: Ensure code runs with specified dependencies

### Areas for Contribution
- Additional deep learning algorithms and architectures
- Improved visualizations and explanatory content
- Performance optimizations and best practices
- Real-world dataset integrations
- Tutorial-style documentation improvements

## 📚 Learning Resources

### Recommended Books
- **"Deep Learning" by Ian Goodfellow**: Comprehensive theoretical foundation
- **"Hands-On Machine Learning" by Aurélien Géron**: Practical implementations
- **"Pattern Recognition and Machine Learning" by Christopher Bishop**: Mathematical foundations

### Online Courses
- **Andrew Ng's Deep Learning Specialization** (Coursera)
- **Fast.ai Practical Deep Learning** course
- **CS231n: Convolutional Neural Networks** (Stanford)

### Documentation and References
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NumPy Documentation](https://numpy.org/doc/)

## 📄 Dataset Information

### Customer Churn Dataset
- **Source**: Kaggle - Credit Card Customer Churn Prediction
- **Features**: 14 attributes including demographics and account information
- **Target**: Binary classification (churned/not churned)
- **Size**: 10,000 customer records
- **License**: Please refer to Kaggle dataset terms of use

## 🐛 Known Issues

- **Perceptron notebook**: Contains incomplete implementation with undefined `step` function
- **Data paths**: Hardcoded Kaggle paths may need adjustment for local use
- **Dependencies**: Some notebooks may require specific TensorFlow versions

## 📞 Contact and Support

- **Repository Owner**: [chaudhary-pawan](https://github.com/chaudhary-pawan)
- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community interaction

## 📜 License

This repository is intended for educational purposes. Please ensure you comply with:
- Dataset licensing terms (especially Kaggle datasets)
- Library and framework licenses
- Academic and commercial use guidelines

---

**Happy Learning! 🚀**

*This repository is maintained as an educational resource for the deep learning community. Contributions, feedback, and suggestions are always welcome!*
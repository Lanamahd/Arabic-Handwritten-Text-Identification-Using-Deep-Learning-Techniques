# Arabic Handwritten Text Identification Using Deep Learning Techniques

This project focuses on developing and implementing deep learning techniques for identifying Arabic handwritten text. The goal is to process, classify, and analyze handwritten text images to achieve accurate recognition, leveraging advanced Convolutional Neural Networks (CNNs) and other deep learning architectures.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Installation and Usage](#installation-and-usage)
6. [File Descriptions](#file-descriptions)
7. [Technologies Used](#technologies-used)
8. [License](#license)

---

## Project Overview

Arabic handwritten text identification is a challenging task due to the complexity of the Arabic script, which includes varying shapes, ligatures, and contextual dependencies. This project explores deep learning techniques for overcoming these challenges and achieving high accuracy in text recognition. The project pipeline includes data preprocessing, model implementation, training, evaluation, and visualization of results.

---

## Dataset

The dataset used for this project consists of images of handwritten Arabic text, divided into isolated words per user. It includes:
- **Images**: Handwritten text samples.
- **Labels**: Corresponding ground truth labels for text samples.

You can find the dataset in the `isolated_words_per_user` directory of this repository.

---

## Methodology

### 1. Data Preprocessing
- Normalization of input images.
- Resizing images to a standard size.
- Data augmentation to improve generalization.

### 2. Model Architecture
- **Convolutional Neural Network (CNN)**:
  - Three convolutional layers with ReLU activation.
  - Max pooling for dimensionality reduction.
  - Fully connected layers for classification.
  - Dropout and batch normalization for regularization.

### 3. Training and Optimization
- Loss Function: Cross-Entropy Loss.
- Optimizer: Adam optimizer with a learning rate of 0.001.
- Training for 25 epochs using a batch size of 32.
- GPU acceleration for faster computation.

### 4. Evaluation Metrics
- Accuracy: Percentage of correctly classified samples.
- Loss: Measurement of error during training and testing phases.

---

## Results

The model achieved the following performance metrics:
- **Training Accuracy**: 93.35% after 25 epochs.
- **Test Accuracy**: 27.07% (indicating overfitting due to limited dataset size or complexity).
- **Visualization**: Learning curves and confusion matrices were plotted to analyze the model's performance.

---

## Installation and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later.
- Required libraries: `torch`, `numpy`, `matplotlib`, and `pandas`.

Install dependencies using:
    ```
    pip install torch numpy matplotlib pandas
    ```
### Running the Project
1. Clone the repository:

    ```bash
    git clone https://github.com/Lanamahd/Arabic-Handwritten-Text-Identification-Using-Deep-Learning-Techniques.git
    cd Arabic-Handwritten-Text-Identification-Using-Deep-Learning-Techniques

2. Open the Jupyter Notebooks:
  - **Part 1**: `1210123_Jouwana_Lana_1210455_vision-project_part1.ipynb`
  - **Part 2**: `projectCV_part2 (7).ipynb`

3. Follow the notebook instructions to preprocess the data, train the model, and evaluate its performance.

---

# File Descriptions

### **`1210123_Jouwana_Lana_1210455_vision-project_part1.ipynb`**
- Jupyter Notebook containing the first part of the analysis, including initial preprocessing and model setup.

### **`projectCV_part2 (7).ipynb`**
- Jupyter Notebook for the second part of the analysis, including model training and evaluation.

### **`projectCV (3).pdf`**
- A comprehensive PDF report summarizing the project's objectives, methods, results, and future directions.

### **`isolated_words_per_user/`**
- Directory containing the dataset of isolated Arabic handwritten words.

---

# Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Visualization**: Matplotlib
- **Development Environment**: Jupyter Notebook

---

# License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

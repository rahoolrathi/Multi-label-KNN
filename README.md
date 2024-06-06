# Multi-label-KNN: k-Nearest Neighbor Based Algorithm for Multi-label Classification

## 1. Introduction
The aim of this project is to develop a robust and efficient algorithm specifically designed for multi-label classification, leveraging the k-Nearest Neighbor (k-NN) approach. This project seeks to address the complexities of predicting multiple labels for each instance in datasets where traditional single-label classification methods are insufficient.

Multi-label classification is prevalent in various domains such as text categorization, bioinformatics, image and video annotation, and more. For instance, in text categorization, a single document can belong to multiple categories. Traditional classification methods, which assign only one label to each instance, fail to capture this complexity. The ML-kNN algorithm aims to fill this gap by effectively predicting multiple labels for each instance, thereby improving classification performance and broadening the applicability of machine learning models.

## 2. Background (Research & Project Selection)
Multi-label classification is prevalent in various domains such as text categorization, bioinformatics, and scene classification. Traditional classification techniques often fall short in these scenarios due to their inability to handle the complexity and interdependencies between multiple labels. For instance, in text categorization, a document can belong to multiple categories, or in bioinformatics, a gene can be associated with multiple functions. Existing methods like Binary Relevance (which treats each label as an independent single-label classification problem) or Label Powerset (which transforms the multi-label problem into a multi-class problem) do not adequately capture the relationships between labels and often result in suboptimal performance.

The kNN algorithm is a simple, yet effective, lazy learning technique traditionally used for single-label classification. It operates by finding the 'k' nearest neighbors to a query instance and then making predictions based on the majority label among these neighbors. This project leverages the well-established kNN algorithm and adapts it for multi-label scenarios, presenting a novel approach known as ML-kNN. By extending kNN to consider multiple labels and their correlations, ML-kNN aims to provide a more accurate and efficient solution for multi-label classification tasks.

## 3. Project Specification

### 3.1 Algorithm Development:
- Extend the k-NN algorithm to handle multi-label classification.
- Identify k-nearest neighbors for a given instance and utilize their label sets to predict the labels for the new instance.

### 3.2 Implementation:
- Develop the ML-kNN algorithm using Python.
- Utilize libraries such as NumPy and Scikit-learn for fundamental operations.

### 3.3 Dataset:
- Utilize real-world multi-label datasets, particularly from bioinformatics, for training and testing the algorithm.

### 3.4 Evaluation:
- Assess the performance of ML-kNN using metrics tailored for multi-label classification, such as Hamming Loss, One-error, Coverage, Ranking Loss, and Average Precision.

## 4. Problem Analysis

### 4.1 Challenges:
- Label Correlation: Labels in multi-label classification tasks are often correlated, and independent prediction of labels can lead to suboptimal results.
- Computational Complexity: The complexity of the classification task increases with the number of labels, making efficient computation challenging.
- Evaluation Metrics: Traditional single-label evaluation metrics are not suitable for multi-label tasks, necessitating the use of specialized metrics.

### 4.2 ML-kNN Approach:
- Label Correlations: By analyzing the label sets of the k-nearest neighbors, ML-kNN captures the interdependencies between labels.
- Lazy Learning: ML-kNN is a lazy learner, meaning it defers the learning process until a prediction is required, allowing it to use the training data directly during prediction.

## 5. Solution Design

### 5.1 Algorithm Steps:
- Neighbor Identification: Identify the k-nearest neighbors for a given test instance based on a distance metric (e.g., Euclidean distance).
- Label Set Analysis: Analyze the label sets of the k-nearest neighbors to compute the posterior probabilities of each label using the maximum a posteriori (MAP) principle.
- Label Prediction: Predict the labels for the test instance based on the computed probabilities.

### 5.2 Functionality:
- Flexible k: Allows specification of the value of k, providing flexibility in neighborhood size.
- Label Probability Calculation: Computes the probability of each label based on the label distribution in the neighborhood.
- Multi-label Output: Predicts a set of labels for each instance rather than a single label.

### 5.3 Features:
- Scalability: Capable of handling large datasets with numerous labels.
- Simplicity: Easy to implement and understand.
- Effectiveness: Incorporates label correlations, leading to improved prediction accuracy.

## 6. Implementation & Testing

### 6.1 Implementation Details:
- Language: Python
- Libraries: NumPy for numerical operations, Scikit-learn for basic k-NN functionalities, and custom code for multi-label extensions.

### 6.2 Testing Plan:
- Datasets: Use publicly available multi-label datasets from domains like bioinformatics.
- Evaluation Metrics: Utilize multi-label specific metrics such as Hamming Loss, One-error, Coverage, Ranking Loss, and Average Precision.
- Test Scenarios: Test the algorithm with different values of k, varying dataset sizes, and different label distributions.

### 6.3 Steps:
- Data Preprocessing: Clean and preprocess the dataset to ensure compatibility with the algorithm.
- Algorithm Implementation: Code the ML-kNN algorithm, ensuring it can handle multi-label data.
- Initial Testing: Run initial tests to ensure the algorithm works as expected.
- Evaluation: Use the specified metrics to evaluate the algorithm's performance.
- Optimization: Refine the algorithm based on evaluation results to improve accuracy and efficiency.

## 7. Results

### 7.1 Confusion Matrix for a Sample Test Case:
<img src="https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/c7219b48-28b3-4232-8fcf-9158e0ae99a1" alt="Confusion Matrix">

### 7.2 Performance Metrics Comparison:
<img src="https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/ebf873f8-5053-4bf8-b174-58f40deb6245" alt="Performance Metrics">
<img src="(https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/813c884e-3aa5-4b26-8e35-51299aa7c80e" alt="Performance Metrics">
<img src="https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/73cee5c6-7ed8-492c-a378-64ea5ac294c7" alt="Performance Metrics">
<img src="https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/7c9cec91-58b5-4d73-981e-8bce91d3028d" alt="Performance Metrics">
### 7.3 Summary of Results:
<img src="https://github.com/rahoolrathi/Multi-label-KNN/assets/129182364/9697760d-6b4e-4563-829e-c96d8595168a" alt="Results">
These results indicate that ML-kNN performs well across various multi-label evaluation metrics, showcasing its effectiveness in predicting multiple labels accurately.

### 7.4 Detailed Analysis:
- **Hamming Loss:** Measures the fraction of incorrect labels to the total number of labels. A lower value indicates better performance.
- **One-error:** Indicates the rate at which the top-ranked predicted label is not in the true label set.
- **Coverage:** Measures how many steps are needed to cover all true labels.
- **Ranking Loss:** Reflects the fraction of label pairs that are incorrectly ordered.
- **Average Precision:** Computes the average precision across all relevant labels.

## 8. Conclusion
The ML-kNN algorithm successfully extends the traditional kNN approach to handle multi-label classification efficiently. Through extensive testing on bioinformatics datasets, ML-kNN demonstrated competitive performance compared to existing multi-label learning algorithms. Its simplicity, combined with the ability to capture label correlations, makes it a practical choice for real-world multi-label classification problems.

Further optimization of the algorithm for larger datasets and different domains will enhance its scalability and performance. Exploring advanced distance metrics could improve neighbor identification, leading to better accuracy. Additionally, integrating ML-kNN with other machine learning frameworks could expand its applicability and ease of use. This project highlights the potential of lazy learning approaches in multi-label classification, paving the way for further research and development in this area.

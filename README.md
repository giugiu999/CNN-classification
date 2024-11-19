# NotMNIST Image Classification and Object Detection Project

This project aims to classify the NotMNIST RGB dataset using a Convolutional Neural Network (CNN) and perform sliding window detection on the NotMNIST-DL dataset to locate two distinct letter objects.

---

## Files Overview

- **A6_main.py**: Handles model training and evaluation.
- **A6_utils.py**: Contains utility functions for data processing and evaluation.
- **A6_submission.py**: Requires implementation of the following key components:
  - Define the CNN architecture.
  - Initialize network weights.
  - Implement forward propagation.
  - Select the best patches for the sliding window detection task.
- **A6_run.ipynb**: Jupyter Notebook to set up the environment, monitor training, and demonstrate usage.

---

## Methodology

1. **Classification Task**:
   - **Dataset**: NotMNIST-RGB (colored images with random backgrounds).
   - **Objective**: Train a CNN to classify letters (A-J) and the background.

2. **Sliding Window Detection Task**:
   - **Dataset**: NotMNIST-DL (images containing two random letter objects).
   - **Objective**: Use a sliding window detector to locate the positions of two letters and evaluate classification accuracy and Intersection over Union (IOU).

---

## How to Run

1. **Environment Setup**:
   - Use `A6_run.ipynb` to configure the Python environment and install dependencies.

2. **Train the Model**:
   - Modify `A6_main.py` or use custom training code to optimize the CNN.
   - Set sliding window parameters and training datasets to improve classification and detection performance.

3. **Evaluate the Model**:
   - Ensure compatibility of `A6_submission.py` with the original `A6_main.py`.
   - Test the trained model on the provided datasets to assess classification and detection metrics.

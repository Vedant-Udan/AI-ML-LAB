# AI/ML Lab 1 - Digit Recognition

## Overview

This project demonstrates a simple digit recognition application using a Convolutional Neural Network (CNN) implemented in PyTorch. The Streamlit app (`app.py`) allows users to upload an image, make predictions, and check if the prediction is correct.

## Requirements

Make sure to install the required dependencies before running the application. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. The app will open in your default web browser. Upload an image, and the app will display the prediction along with the option to mark it as a wrong prediction.

3. If marked as a wrong prediction, you will be prompted to enter the expected target label, and the image and prediction information will be saved in the ./wrong_result folder.

## Files

- `app.py`: Streamlit app for digit recognition.  
- `model_architecture.py`: Python file containing the `SimpleCNN` class for the model architecture.  
- `AIML_LAB_1.ipynb`: Jupyter Notebook file containing the model code.

## Notes

- The model is loaded from the `model.pth` file.
- The `AIML_LAB_1.ipynb` file contains the detailed code for the CNN model.

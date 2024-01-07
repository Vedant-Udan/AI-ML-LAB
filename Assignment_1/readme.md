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
4. Run the `re_train.py` script to fine-tune the model based on wrong predictions.

```bash
python re_train.py
```

This script loads the original model, loads wrong prediction data, fine-tunes the model, and saves the updated weights.

## Automating Fine-Tuning with Cron Job

To automate the fine-tuning process, you can use `crontab`. Edit your crontab file with the following steps:

1. Open the crontab file for editing:

    ```bash
    crontab -e
    ```

2. Add a line to schedule the `re_train.py` script. For example, to run it every day at 3 AM:

    ```bash
    0 3 * * * ./re_train.py >> /path/to/fine_tuning.log 2>&1
    ```

3. Save and exit the editor.

Now, the `re_train.py` script will be executed automatically according to the schedule you've defined, helping you fine-tune the model regularly.


## Files

- `app.py`: Streamlit app for digit recognition.  
- `utils.py`: Python file containing the `SimpleCNN` class for the model architecture and other required functions.  
- `AIML_LAB_1.ipynb`: Jupyter Notebook file containing the model code.
- `re_train.py`: Python script for fine-tuning the model on wrong prediction data.

## Notes

- The model is loaded from the `model.pth` file.
- The `AIML_LAB_1.ipynb` file contains the detailed code for the CNN model.

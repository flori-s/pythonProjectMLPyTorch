# Python Project ML PyTorch

This project is a machine learning application using PyTorch to predict the similarity between CVs and job vacancies. It includes data loading, feature preparation, model training, and prediction functionalities.

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/flori-s/pythonProjectMLPyTorch
    cd pythonProjectMLPyTorch
    ```

2. **Install the required dependencies**:

    If you have `pip` installed, run:

    ```sh
    pip install -r requirements.txt
    ```

    If you encounter the following error:

    ```sh
    zsh: command not found: pip
    ```

    This means `pip` is not installed on your system. To install `pip`, follow these steps:

    - For **Python 3**:

      ```sh
      python3 -m ensurepip --upgrade
      ```

      or

      ```sh
      python3 -m pip install --upgrade pip
      ```

## Running the Project

1. Ensure you have the necessary data files in the `data` directory:
    - `cv_data.csv`
    - `vacature_data.csv`
    - `labels.csv`

2. **Run the main script**:

    If you're using Python 3 (which is likely), you may need to use `python3` instead of `python`. Try running:

    ```sh
    python3 Main.py
    ```

    If you encounter the following error:

    ```sh
    zsh: command not found: python
    ```

    This means the `python` command is not found on your system, likely because you're using Python 3 and the command is `python3`. 

    **Solution**:
    - Try running the script with `python3`:

      ```sh
      python3 Main.py

## Project Structure

- `Main.py`: The main script to run the project.
- `DataController.py`: Contains functions for loading data and preparing features.
- `modelTorch.py`: Defines the neural network model and functions for training and prediction.
- `requirements.txt`: Lists the required Python packages.

## What This Project Does

1. **Install Requirements**: Installs the necessary Python packages listed in `requirements.txt`.
2. **Load Data**: Loads CV, vacancy, and label data from CSV files.
3. **Prepare Features**: Combines and vectorizes text data from CVs and vacancies using TF-IDF.
4. **Train Model**: Trains a simple neural network model to predict similarity scores between CVs and vacancies.
5. **Predict Similarity**: Uses the trained model to predict similarity scores for the provided data.
6. **Display Results**: Displays the prediction results and allows for user feedback to update the model.


## Feedback Mechanism

After running the main script, the program will display the predicted match percentages for each CV and job vacancy. 

For example:

``` sh
Results for Lisa de Bruin:
+---------------+--------------------------+------------------------------+
| Name          | Vacancy Title            |   Predicted Match Percentage |
+===============+==========================+==============================+
| Lisa de Bruin | Senior Data Scientist    |                      90.6602 |
+---------------+--------------------------+------------------------------+
| Lisa de Bruin | Junior Software Engineer |                      83.8105 |
+---------------+--------------------------+------------------------------+
| Lisa de Bruin | Frontend Developer       |                      83.5005 |
+---------------+--------------------------+------------------------------+
| Lisa de Bruin | Backend Developer        |                      78.1615 |
+---------------+--------------------------+------------------------------+
| Lisa de Bruin | DevOps Engineer          |                      73.5628 |
+---------------+--------------------------+------------------------------+
```
The program will then ask for feedback on whether the predicted match for each vacancy is correct:
``` sh
Is the match for Senior Data Scientist correct? (yes/no):
```
- If you type `yes`, the program will move on to the next prediction.
- If you type `no`, the program will prompt you to enter the correct match percentage. This feedback will be used to update the model and improve its accuracy.

For example:
``` sh
Is the match for Senior Data Scientist correct? (yes/no): no
Enter the correct match percentage for Senior Data Scientist: 95.0
```
The model will then be updated with the new data, and the updated labels will be saved to `data/labels.csv`.

This feedback mechanism helps to continuously improve the model by incorporating user feedback into the training process.

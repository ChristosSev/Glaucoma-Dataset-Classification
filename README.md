# Glaucoma_Dataset_Classification

Fine-tuning a pretrained Resnet For binary classification of referable glaucoma (RG) and non-referable glaucoma (NRG).

## Dataset

This project utilizes the "EYEPACS-AIROGS-Light" dataset, available on Kaggle: [https://www.kaggle.com/datasets/deathtrooper/eyepacs-airogs-light](https://www.kaggle.com/datasets/deathtrooper/eyepacs-airogs-light). The dataset contains raw eye images and is pre-split into training, validation, and testing sets.

## Repository Setup

1.  **Download the Dataset:**
    * Download the "EYEPACS-AIROGS-Light" dataset from the Kaggle link provided.
    * The downloaded dataset will be in a folder named "release-raw".

2.  **Clone the Repository:**
    * Clone this repository to your local machine.
    * move the "release-raw" folder into your cloned repository directory.

3.  **Directory Structure:**
    Your repository should have the following directory structure:

    ```
    Glaucoma_Dataset_Classification/
    ├── release-raw/
    │   ├── train/
    │   │   ├── RG/
    │   │   │   └── ... (RG images)
    │   │   └── NRG/
    │   │       └── ... (NRG images)
    │   ├── val/
    │   │   ├── RG/
    │   │   │   └── ... (RG images)
    │   │   └── NRG/
    │   │       └── ... (NRG images)
    │   └── test/
    │       ├── RG/
    │       │   └── ... (RG images)
    │       └── NRG/
    │           └── ... (NRG images)
    ├── train.py
    ├── ... (other files)
    ```

## Training and Testing the Model

1.  **Open a Terminal:**
    * Open a terminal or command prompt and navigate to the root directory of the cloned repository.

2.  **Run Training:**

    * **Default Parameters:**
        To train the model with default parameters, execute the following command:

        ```bash
        python3 train.py
        ```

    * **Hyperparameter Search:**
        To perform hyperparameter search and find the best parameters for the model, execute the following command:

        ```bash
        python3 train.py --use_hyperparam_search
        ```

    

# Deep Learning Approaches for Detecting Absence of Lung Sliding in Ultrasound Videos

## Overview

The primary goal of this project is to develop and evaluate deep learning models capable of accurately detecting the absence of lung sliding in lung ultrasound (LUS) videos. This detection is crucial for diagnosing conditions such as pneumothorax, which requires timely and accurate intervention.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Training a Model](#training-a-model)
   - [Testing a Model](#testing-a-model)
4. [Model Architecture](#model-architecture)
5. [Dataset](#dataset)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

## Introduction

Pneumothorax is a potentially life-threatening condition characterized by the presence of air in the pleural cavity, which can impair respiratory function. Identifying lung sliding in ultrasound videos is a key component of pneumothorax assessment. Our research proposes a novel deep learning approach that integrates optical flow analysis with Convolutional Neural Networks (CNNs) for automated lung sliding classification.

## Installation

To set up the project environment, follow these steps:

1. Navigate to the project directory:
    ```bash
    cd your_repository
    ```

2. Install the required packages using pip and the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

This command will install all the necessary Python packages specified in the `requirements.txt` file.

## Usage

### Training a Model

To train a model, execute the following command in the terminal:

```bash
python train_model.py --new_checkpoint_name <new_checkpoint_name> --load_checkpoint_name <load_checkpoint_name> --epochs <epochs>
```

**CLI Options:**

- `--new_checkpoint_name`, `-n`: Specifies the name of the checkpoint to save after training on every epoch. The default name is `model`.
- `--load_checkpoint_name`, `-l`: Specifies the path to the model weights to load before training. By default, no pre-trained weights are loaded.
- `--epochs`, `-e`: Specifies the number of epochs to train the model.

**Example:**
To train a model for 100 epochs and save the checkpoint as `my_model`, use the following command:

```bash
python train_model.py -n my_model -e 10
```

Additionally, users can modify hyperparameters using the JSON configuration file located at `config/hyperparameters.json`. This file allows users to specify hyperparameters such as learning rate, batch size, etc. The value `auto` indicates that the parameter uses the default value, or it is calculated dynamically before training.

### Testing a Model

To test a model using a checkpoint file, run the following command:

```bash
python test_model.py <checkpoint_name>
```

**Arguments:**

- `checkpoint_name`: Specifies the name of the checkpoint file to load for testing.

**Example:**
To test a model using a checkpoint named `my_model_checkpoint.pth`, use the following command:

```bash
python test_model.py my_model_checkpoint.pth
```

This command will load the checkpoint file `my_model_checkpoint.pth` located in the models directory and perform testing on the model.

## Model Architecture

Our proposed architecture combines optical flow analysis with CNNs to capture the temporal dynamics of lung motion and extract meaningful features for classification. We experimented with several model variations, including 3D convolutional models and recurrent models.

## Dataset

The dataset comprises 171 ultrasound videos of lung examinations. These videos vary in length from 20 to 900 frames and are stored in grayscale PNG format. The dataset includes two classes: "Lung sliding present" and "Lung sliding absent".

## Evaluation

Evaluate the model using the following command:

```bash
python evaluate.py --model_dir path_to_saved_model --data_dir path_to_evaluation_data
```

The evaluation script will output the model's performance metrics, including recall and specificity, which are critical for medical applications.

## Results

Our experiments demonstrated that the proposed deep learning approach achieved robust performance in detecting the absence of lung sliding. The final model variations provided valuable insights into the strengths and limitations of each approach.

## Contributing

We welcome contributions to improve the project. Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was supported by the Technical University of Košice. We thank our mentors and colleagues for their valuable input and support throughout this project.

For more detailed information about the project, please refer to the [article](article/article.pdf) and the [documentation](docs/docs.pdf).

---

Feel free to explore and contribute to our project. For any questions or issues, please open an issue on the GitHub repository.

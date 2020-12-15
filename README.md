# UW ECE 562 (Au 2020) - MNIST GAN/ACGAN 

## Requirements

- Install Pytorch base on the system and GPU availability on your machine:
    ```
    Pytorch >= 1.6.0
    ```

- Install the required packages through conda:
    ```bash
    conda create --name <env> --file requirements.txt
    ```

## Dataset download
- Download the MNIST dataset:
    ```bash
    sh download.sh
    ```

## Get your hands dirty
- Train your GAN/CGAN/ACGAN network. By default, the logs, the intermediate results and the saved models are located in `logs/`, `samples/` and `models/`, respectively.
    ```bash
    sh train.sh
    ```

- Test your generator. By default, it will load the model saved in the given epoch and generate the output images to `outputs/`.
    ```bash
    sh eval.sh
    ```

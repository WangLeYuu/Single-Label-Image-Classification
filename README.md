# Single-Label-Image-Classification

## Features

> **Code Decoupling:** Decouple all data loading, network model construction, model training and validation, and model format conversion
>
> **Rich Content:** Providing rich evaluation indicators and functional functions

## Function Functionality

> - checkpoints: stores the weights of the trained model;
> - datasets: Store datasets. And partition the dataset;
> - log_dir: Stores training logs. Including losses and accuracy during training and validation;
> - option. py: stores all the parameters required for the entire project;
> - utils.py: Stores various functions. Including folder creation, drawing accuracy and loss changes, result prediction, etc;
> - getdata.py: Build a data pipeline. It defines the mean and variance functions for calculating all graphs in the dataset;
> - model. py: Building a neural network model;
> - train.py: Train the model;
> - evaluate. py: Evaluate the training model. There are three prediction methods to choose from, namely: predicting a single image, predicting multiple images, and predicting images in the entire directory;
> - pth2onnx: Convert the pth model to the onnx model;
> - onnx_inference.py: Use the. onnx model to infer data;
> - split data: Divide the original dataset.

## Requirements

Required:

> matplotlib==3.8.3
> numpy==1.26.4
> onnx==1.16.1
> onnxruntime==1.18.0
> Pillow==9.5.0
> Pillow==10.3.0
> scikit_learn==1.5.0
> scipy==1.13.1
> torch==2.2.1
> torchsummary==1.5.1
> torchvision==0.17.1
> torchviz==0.0.2
> tqdm==4.66.2



You can install these dependencies via pip:

```python
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Prepare your dataset and place it according to the following requirements:

> ```python
> datasets
>     original_img
>         class1
>         class2
>         -----
> ```

### 2. Split Your Dataset

Run the following script to obtain the partitioned dataset, but you need to pay attention to modifying some paths:

```python
python split_data.py
```

Then you can obtain the following data structure:

> ```python
> datasets
>     original_img
>         class1
>         class2
>         -----
>     train
>         class1
>         class2
>         -----
>     val
>         class1
>         class2
>         -----
>     test
>         class1
>         class2
>         -----
> ```

### 3. Modify Network 

Modify your network structure according to the actual situation of your dataset, and pay attention to modifying the last FC layer of the network to meet your classification needs.



Run the following script to view your network structure:



```python
python model.py
```

### 4. Train Your Dataset

Run the following script to train your dataset and output various parameters during the training and validation processes:



```python
python train.py
```

### 5. Evaluate Your Model

Running the following code can evaluate the accuracy of your model. You can choose to evaluate the entire folder or individual images:



```python
python evaluate.py
```

### 6. Model Format Conversion



If you need to convert the pytorch model to an onnx model for subsequent model deployment, you can run the following code and verify the accuracy of the onnx model:

```python
python pth2onnx.py

python onnx_inference.py
```

## License

This project is licensed under the Apache 2.0 license. For detailed information, please refer to the LICENSE file.
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgement

> Kaggle Dataset: [Blood Cell Images (kaggle.com)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)



> CSDN: [王乐予-CSDN博客](https://blog.csdn.net/qq_42856191?type=blog)


















































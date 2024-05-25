import torch
import argparse
import numpy as np
import onnxruntime
import time
from getdata import MyData


def infer_test(model_path, data_loader, device):
    if device == 'cpu':
        print("using CPUExecutionProvider")
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        print("using CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total = 0.0
    correct = 0
    start_time = time.time()
    for batch, data in enumerate(data_loader):
        X, y = data
        X = X.numpy()
        y = y.numpy()

        output = session.run([output_name], {input_name: X})[0]
        y_pred = np.argmax(output, axis=1)

        if y[0] == y_pred[0]:
            correct += 1
        total += 1
    end_time = time.time()
    print(end_time - start_time)
    print("accuracy is {}%".format(correct / total * 100.0))


def main():
    input_model_path = "./checkpoints/best.onnx"
    device = input("cpu or gpu?")

    # benchmark(input_model_path, device)

    dataloaders = MyData()

    infer_test(input_model_path, dataloaders['test'], device)


if __name__ == "__main__":
    main()


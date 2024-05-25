import torchvision
import torch
from torch.autograd import Variable
import onnx
from model import My_CNN, ResNet
from option import get_args
opt = get_args()

model = ResNet()
ckpt = torch.load(opt.test_model_path, map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()
input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 224, 224))

torch.onnx.export(model, input, './checkpoints/best.onnx', input_names=input_name, output_names=output_name, verbose=True)

# check .onnx model
onnx_model = onnx.load("./checkpoints/best.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
from model import My_CNN, ResNet
from option import get_args
opt = get_args()

model = ResNet()
ckpt = torch.load(opt.test_model_path, map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()

x = torch.ones((1, 3, 224, 224)).cuda()

model_trt = torch2trt(model, [x])

torch.save(model_trt.state_dict(), './checkpoints/best_trt.pth')

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('./checkpoints/best_trt.pth'))


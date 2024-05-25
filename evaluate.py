from model import My_CNN, ResNet
from getdata import MyData, data_augmentation
import torch.utils.data
from option import get_args
from utils import visual_image_single, visual_image_multi, visual_img_dir


opt = get_args()

model = ResNet()
ckpt = torch.load(opt.test_model_path, map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()

data_transform = data_augmentation()        # test single image
transform_test = data_transform['test']

dataloaders = MyData()                      # test dir
dataloader = dataloaders['test']

class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

if __name__ == '__main__':

    # visual_image_single(opt.test_img_path, transform_test, model, class_names)
    # visual_image_multi(dataloader, model, class_names)
    visual_img_dir(dataloader, model, class_names=class_names)
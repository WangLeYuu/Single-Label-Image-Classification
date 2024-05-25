from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from option import get_args
import torch
import numpy as np
import matplotlib.pyplot as plt
opt = get_args()

def data_augmentation():
    '''
    Data augmentation
    :return: Transformer with data augmentation operation
    '''
    data_transform = {
        'train': transforms.Compose([
            # transforms.RandomRotation(45),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(p=0.025),
            transforms.ToTensor(),  # HWC -> CHW
            transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
        ])
    }
    return data_transform


def MyData():

    data_transform = data_augmentation()

    image_datasets = {
        'train': ImageFolder(opt.dataset_train, data_transform['train']),
        'val': ImageFolder(opt.dataset_test, data_transform['val']),
        'test': ImageFolder(opt.dataset_test, data_transform['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=opt.batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=opt.batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=opt.batch_size, shuffle=True)
    }
    return dataloaders


"""
Image visualization
"""
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.6786, 0.6413, 0.6605])
    std = np.array([0.2599, 0.2595, 0.2569])
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

"""
Calculate the mean and variance of all images in the dataset
"""
def get_mean_and_std(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


if __name__ == '__main__':
    mena_std_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(opt.dataset_train, transform=mena_std_transform)
    print(dataset.class_to_idx)
    mean, std = get_mean_and_std(dataset)
    print(mean)
    print(std)
    dataloader = MyData()
    inputs, classes = next(iter(dataloader['train']))
    out = make_grid(inputs)
    class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    imshow(out, title=[class_names[x] for x in classes])





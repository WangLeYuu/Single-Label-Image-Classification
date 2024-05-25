import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch import optim
from torchsummary import summary
from model import My_CNN, ResNet
from getdata import MyData
from utils import draw_number, EarlyStopping, make_dir
from option import get_args
opt = get_args()

make_dir()
file = open(opt.logging_txt, 'w')
writer = SummaryWriter()


# set EarlyStopping
early_stopping = EarlyStopping(save_path=opt.checkpoints, patience=5)

def train_best(model, num_epoch, dataloaders, optimizer, lr, loss_function):

    model.to(opt.device)
    train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt = [], [], [], []  # Preserve the loss and accuracy of training and validation processes for drawing line graphs
    best_loss = np.inf

    for epoch in range(start_epoch + 1, opt.epochs):
        print("---------Start Epoch: {}/{} Lr：{}---------".format(epoch, opt.epochs, lr.get_last_lr()))
        for phase in ['train', 'val']:

            loss_sum, acc_sum = 0, 0
            step = 0            # Retrieve all data and record each batch
            all_step = 0        # Record how many data were taken

            for (inputs, labels) in tqdm(dataloaders[phase], position=0):
                if phase == 'train':
                    model.train()
                if phase == 'val':
                    model.eval()
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)
                optimizer.zero_grad()  # Gradient zeroing to prevent accumulation

                a = inputs.size(0)  # How many images were taken in each batch
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                _, pred = torch.max(outputs, 1)  # Return the maximum value and index of each row
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * inputs.size(0)  # Loss
                acc_sum += torch.sum(pred == labels.data)

                step += 1
                all_step += a

                print("[Epoch: {}/{}]  [step = {}]  [{}_loss = {:.3f}, {}_acc = {:.3f}]".
                      format(epoch, opt.epochs, all_step, phase, loss_sum / all_step, phase, acc_sum.double() / all_step))

            # Preserve training loss and accuracy for each epoch
            if phase == 'train':
                train_loss = loss_sum / len(dataloaders[phase].dataset)
                train_acc = acc_sum.double() / len(dataloaders[phase].dataset)
                train_acc = np.float32(train_acc.cpu().numpy())
                train_loss_plt.append(train_loss)
                train_acc_plt.append(train_acc)

            else:
                val_loss = loss_sum / len(dataloaders[phase].dataset)
                val_acc = acc_sum.double() / len(dataloaders[phase].dataset)
                val_acc = np.float32(val_acc.cpu().numpy())
                val_loss_plt.append(val_loss)
                val_acc_plt.append(val_acc)

                writer.add_scalars('loss', {
                    'train': train_loss,
                    'val': val_loss,
                }, global_step=epoch+1-start_epoch)
                writer.add_scalars('acc', {
                    'train': train_acc,
                    'val': val_acc,
                }, global_step=epoch + 1 - start_epoch)
                writer.close()



        print("EPOCH = {}/{}  train_loss = {:.3f}, train_acc = {:.3f}, val_loss = {:.3f}, val_acc = {:.3f}, lr_rate = {}, \n".
              format(epoch, num_epoch, train_loss, train_acc, val_loss, val_acc, lr.get_last_lr()))
        file.write("EPOCH = {}/{}  train_loss = {:.3f}, train_acc = {:.3f}, val_loss = {:.3f}, val_acc = {:.3f}, lr_rate = {}, \n".
              format(epoch, num_epoch, train_loss, train_acc, val_loss, val_acc, lr.get_last_lr()))


        if epoch % 2 == 0:
            torch.save(model, opt.checkpoints + 'model_{}.pth'.format(epoch))

        # Set training early stop
        early_stopping(val_loss, model)
        # When the early stop condition is reached, early_stop will be set to True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # Jump out of iteration and end training

    draw_number(np.arange(0, epoch+1-start_epoch, 1), train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt)


if __name__ == '__main__':
    model = ResNet()
    # model = nn.DataParallel(model)
    model.to(opt.device)
    print(summary(model, (3, opt.loadsize, opt.loadsize), opt.batch_size))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    if opt.pretrained:
        checkpoint = torch.load(opt.checkpoints + opt.which_epoch)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load Epoch {} Success！'.format(start_epoch))
    else:
        start_epoch = 0
        print('No saved model, training will start from scratch!')

    dataloaders = MyData()

    train_best(model, opt.epochs, dataloaders, optimizer, sch, loss_function)

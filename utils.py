import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from option import get_args
opt = get_args()


"""
Makr Dirs
"""
def make_dir():
    if os.path.exists(opt.log_dir) == True:
        pass
    else:
        os.mkdir(opt.log_dir)
    if os.path.exists(opt.checkpoints) == True:
        pass
    else:
        os.mkdir(opt.checkpoints)

"""
Set early stop function
"""
class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience, verbose=False, delta=0):
        """
        Args:
            save_path : Model Save Folder
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


"""
Draw changes in loss and accuracy
"""
def draw_number(epochs, train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt):

    color = ['red', 'blue', 'green', 'orange']
    marker = ['o', '*', 'p', '+']
    linestyle = ['-', '--', '-.', ':']

    plt.plot(epochs, train_loss_plt, color=color[0], marker=marker[0], linestyle=linestyle[0], label="trainingsets-loss")
    plt.plot(epochs, train_acc_plt, color=color[1], marker=marker[1], linestyle=linestyle[1], label="trainingsets-acc")
    plt.plot(epochs, val_loss_plt, color=color[2], marker=marker[2], linestyle=linestyle[2], label="validationsets-loss")
    plt.plot(epochs, val_acc_plt, color=color[3], marker=marker[3], linestyle=linestyle[3], label="validationsets-acc")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Loss and accuracy changes in training and validation sets")
    plt.savefig("Loss_Accuracy.jpg")
    plt.show()


"""
Single image visualization prediction
"""
def visual_image_single(img_path, transform_test, model, class_names):
    image = Image.open(img_path).convert('RGB')
    img = transform_test(image)
    img = img.unsqueeze_(0)
    out = model(img)
    pred_softmax = F.softmax(out, dim=1)
    top_n = torch.topk(pred_softmax, len(class_names))
    confs = top_n[0].cpu().detach().numpy().squeeze().tolist()      # Prediction probabilities for all categories
    confs_max = max(confs)      # Maximum probability value
    confs_max_position = confs.index(confs_max)     # The location of the maximum probability value
    print('Pre:{}   Conf:{:.3f}'.format(class_names[confs_max_position], confs_max))
    plt.axis('off')
    plt.title('Pre:{}   Conf:{:.3f}'.format(class_names[confs_max_position], confs_max))
    plt.imshow(image)
    plt.show()


"""
Visualization prediction of multiple images
"""
def visual_image_multi(dataloader, model, class_names):
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(images)):
                plt.subplot(4, 4, i + 1)
                plt.title("Prediction:{}\nTarget:{}".format(class_names[predicted[i]], class_names[labels[i]]), fontsize=8)
                img = images[i].swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                plt.imshow(img)
                plt.axis('off')
            plt.show()


"""
Predict the entire folder and output a confusion matrix
"""
def get_confusion_matrix(trues, preds, labels):
    conf_matrix = confusion_matrix(trues, preds, labels=[i for i in range(len(labels))])
    return conf_matrix

def plot_confusion_matrix(conf_matrix, labels):
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    indices = range(conf_matrix.shape[0])
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
          plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def get_roc_auc(trues, preds, labels):
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), preds.ravel())     # Compute micro-average ROC curve and ROC area
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))     # First aggregate all false positive rates

    mean_tpr = np.zeros_like(all_fpr)       # Then interpolate all ROC curves at this points
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("ROC.jpg")
    plt.show()

def visual_img_dir(dataloader, model, class_names):
    """
    normalize: True:Display percentage, False: Display the number of items
    """
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)  # The proportion of all correctly judged data (TP+TN) to the total accuracy value.
        precision = precision_score(y_true, y_pred, average='macro')  # The proportion of true positive classes (TP) among all judged positive classes (TP+FP) with accuracy.
        recall = recall_score(y_true, y_pred, average='macro')  # The proportion of true positive class (TP+FN) in the recall rate determined to be positive class (TP).
        f1 = f1_score(y_true, y_pred, average='macro')  # F1 score assigns the same weight to Precision score and Recall score to measure their accuracy performance, making it an alternative to accuracy metrics (it does not require us to know the total number of samples).
        # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        conf_matrix = get_confusion_matrix(y_true, y_pred, labels=class_names)
        print('Classification Report:\n', classification_report(y_true, y_pred))  # Classification report
        print("[accuracy:{:.4f}]  [precision:{:.4f}]  [recall:{:.4f}]  [f1:{:.4f}]".format(accuracy, precision, recall, f1))
        plot_confusion_matrix(conf_matrix, labels=class_names)

        test_trues = label_binarize(y_true, classes=[i for i in range(len(class_names))])
        test_preds = label_binarize(y_pred, classes=[i for i in range(len(class_names))])
        get_roc_auc(test_trues, test_preds, class_names)


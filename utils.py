import matplotlib.pyplot as plt
import collections
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import time, copy
import pickle
import seaborn as sn
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report


def get_weighted_sampler(train_labels, cls_labels):
    counter = collections.Counter(train_labels)
    od_counter = collections.OrderedDict(sorted(counter.items()))
    class_count = [od_counter[i] for i in range(len(cls_labels))]
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
    train_labels = [int(i) for i in train_labels]
    target_list = torch.tensor(train_labels, dtype=torch.int64)
    class_weights_all = class_weights[target_list]
    
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    return weighted_sampler



def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                num_epochs=100, fold=0, device='cuda', model_dir='/trained_models/'):

    since = time.time()

    val_acc_history, val_loss_history = [], []
    train_acc_history, train_loss_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)


                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_dir + '/model_f{}.pth'.format(fold))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                print('\rEpoch {}/{} {} Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch+1, num_epochs, phase, epoch_loss, epoch_acc), end="")

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                print('\rEpoch {}/{} {} Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch+1, num_epochs, phase, epoch_loss, epoch_acc), end="")

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, (train_loss_history, train_acc_history, val_loss_history, val_acc_history)




def test_model(model, dataloader, class_labels, device='cpu', save_dir='/outputs/'):
    y_pred, y_true = [], []
    model.eval()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)


            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(labels.cpu().tolist())


    cfm = confusion_matrix(y_true, y_pred)
    print("Original confusion matrix:")
    print(cfm)

    num_classes = len(class_labels)  # = 5


    if cfm.shape[0] < num_classes:
        padded = np.zeros((num_classes, num_classes), dtype=int)
        unique = sorted(set(y_true))     # e.g. [2]

        for idx, cls in enumerate(unique):
            padded[cls, :cfm.shape[1]] = cfm[idx]

        cfm = padded

    print("Padded 5x5 confusion matrix:")
    print(cfm)


    FP = cfm.sum(axis=0) - np.diag(cfm)
    FN = cfm.sum(axis=1) - np.diag(cfm)
    TP = np.diag(cfm)
    TN = cfm.sum() - (FP + FN + TP)

    def sd(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    TPR = sd(TP, TP + FN)
    TNR = sd(TN, TN + FP)
    PPV = sd(TP, TP + FP)
    F1  = sd(2*(PPV*TPR), (PPV + TPR))
    ACC = sd(TP+TN, TP+FP+FN+TN)


    class_length_weights = np.ones(num_classes) / num_classes

    avg_ppv = np.dot(PPV, class_length_weights)
    avg_tpr = np.dot(TPR, class_length_weights)
    avg_tnr = np.dot(TNR, class_length_weights)
    avg_acc = np.dot(ACC, class_length_weights)
    avg_f1  = np.dot(F1,  class_length_weights)

    AVG = np.array([avg_tpr, avg_tnr, avg_f1, avg_ppv, avg_acc])

    eval_mat = np.array([TPR, TNR, F1, PPV, ACC]).T
    eval_mat = np.vstack([eval_mat, AVG])

    df = pd.DataFrame(
        eval_mat * 100,
        columns=['Sens', 'Spec', 'F1', 'Prec', 'Acc'],
        index=class_labels + ['Avg.']
    )

    print(df)

import monai
import torch
from monai.data import Dataset

import fusionset
import transform
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ############################
    # Parameters
    ############################
    BATCHSIZE = 1
    K = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    from models.multifusion import FusionNet
    Architectures_cls = ['ResNet', 'FusionNet']
    architecture_cls = Architectures_cls[1]
    MODELPATH_cls = 'checkpoints/' + architecture_cls + '/net_' + str(K) + '_best_epoch_90.pth'
    model = FusionNet(num_classes=2)
    print('#parameters_cls:', sum(param.numel() for param in model.parameters()))
    model = model.to(device)
    model.load_state_dict(torch.load(MODELPATH_cls))
    running_corrects = 0.
    ############################
    # load data
    ###########################
    val_dict = fusionset.rtDict('/root/autodl-tmp/Dataset_perprocessed/test/MRI', '/root/autodl-tmp/Dataset_perprocessed/test/PET')
    val_dataset = Dataset(data=val_dict, transform=transform.fusionTransform_val)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    model.eval()
    labels = []
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        for iteration, data in enumerate(val_loader):
            mri, pet, label = data["mri"].to(device), data["pet"].to(device), data["label"].to(device)
            label = torch.squeeze(label)
            pred = model(mri, pet)
            probs, preds = torch.max(pred, 1)
            labels.append(label.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
            print(preds)
            if preds == 0:
                pred_probs.append(1 - probs.cpu().numpy())
                print(preds, 1 - probs.cpu().numpy())
            else:
                pred_probs.append(probs.cpu().numpy())
                print(preds, probs.cpu().numpy())
            running_corrects += torch.sum(preds == label)
        test_acc = running_corrects.double() / (len(val_loader) * BATCHSIZE)

        print('ACC: ', test_acc)

    f1 = f1_score(labels, pred_labels, average='macro')
    recall = recall_score(labels, pred_labels, average='macro')

    auc_score = roc_auc_score(labels, pred_probs)
    print('F1: ', f1, 'REC: ', recall, 'AUC: ', auc_score)
    fpr, tpr, thresholds = roc_curve(labels, pred_probs)
    plt.title('auc')
    plt.plot(fpr, tpr, c='r', linewidth=1)
    plt.ylabel('false positive rate')
    plt.xlabel('true positive rate')
    plt.savefig('statistics/' + architecture_cls + '/test_auc.tif')
    plt.show()

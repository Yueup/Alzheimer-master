import time
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import dataset.multimodal as multimodal
import transform
import monai
import torch.optim as optim
from monai.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_auc_score
from utils import *
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                        help='number of classification classes')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Architectures = ['ResNet', 'FusionNet']
    architecture = Architectures[1]

    train_dict = multimodal.Multi_modal_data('/root/autodl-tmp/Dataset_perprocessed/train/MRI', '/root/autodl-tmp/Dataset_perprocessed/train/PET')
    val_dict = multimodal.Multi_modal_data('/root/autodl-tmp/Dataset_perprocessed/test/MRI', '/root/autodl-tmp/Dataset_perprocessed/test/PET')
    train_dataset = Dataset(data=train_dict, transform=transform.fusionTransform_train)
    val_dataset = Dataset(data=val_dict, transform=transform.fusionTransform_val)
    train_loader = monai.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = monai.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    from models.FusionClsN import FusionNet
    model = FusionNet(num_classes=parser.num_classes).to(device)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # Train the net
    results_loss = {'loss': [], 'val_loss': []}
    results_acc = {'train_acc': [], 'val_acc': []}
    results_auc = {'auc': []}
    # for saving best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 2.
    best_acc = 0.
    best_epoch = 0

    since = time.time()
    logpath = 'statistics/' + architecture + '/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logger = logger(logpath + 'train_' + architecture + '.log')
    logger.info('start training!')
    for epoch in range(1, parser.epochs + 1):
        # adjust_lr(optimizer=optimizer, init_lr=0.001, epoch=epoch, decay_rate=0.1, decay_epoch=20)
        epochloss = {'loss': [], 'val_loss': []}
        epochacc = {'train_acc': [], 'val_acc': []}
        epochauc = {'auc': []}
        model.train()
        running_corrects = 0
        epoch_acc = 0.0
        for iteration, data in enumerate(train_loader):
            mri, pet, label = data["mri"].to(device), data["pet"].to(device), data["label"].to(device)
            label = torch.squeeze(label)
            optimizer.zero_grad()
            pred = model(mri, pet)
            _, preds = torch.max(pred, 1)
            loss = criterion_cls(pred, label)
            loss.backward()
            optimizer.step()
            running_corrects += torch.eq(preds, label).sum().item()
            epochloss['loss'].append(loss.item())
        results_loss['loss'].append(np.mean(epochloss['loss']))
        epoch_acc = running_corrects / (len(train_loader) * parser.batch_size)
        results_acc['train_acc'].append(epoch_acc)
        ############################
        # validate the net
        ############################
        model.eval()
        val_running_corrects = 0
        val_epoch_acc = 0.0
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
                loss = criterion_cls(pred, label)
                epochloss['val_loss'].append(loss.item())
                val_running_corrects += torch.eq(preds, label).sum().item()
            results_loss['val_loss'].append(np.mean(epochloss['val_loss']))
            val_epoch_acc = val_running_corrects / (len(val_loader) * parser.batch_size)
            results_acc['val_acc'].append(val_epoch_acc)
        logger.info("Average: Epoch/Epoches {}/{}\t"
                    "train epoch loss {:.3f}\t"
                    "val epoch loss {:.3f}\t"
                    "train epoch acc {:.3f}\t"
                    "val epoch acc {:.3f}\t".format(epoch, parser.epochs, np.mean(epochloss['loss']),
                                                np.mean(epochloss['val_loss']),
                                                epoch_acc, val_epoch_acc))
        scheduler.step()
        # saving the best model parameters
        if np.mean(epochloss['val_loss']) < best_loss:
            best_loss = np.mean(epochloss['val_loss'])
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            filepath = 'checkpoints/' + architecture + '/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            torch.save(best_model_wts, filepath + '/net_best_epoch_%d.pth' % best_epoch)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('finish training!')

    ############################
    # save the results
    ############################
    data_frame = pd.DataFrame(
        data={'loss': results_loss['loss'],
              'val_loss': results_loss['val_loss']},
        index=range(1, parser.epochs + 1))
    data_frame.to_csv('statistics/' + architecture + '/train_results.csv', index_label='Epoch')
    #
    # ############################
    # # plot the results
    # ############################
    # LOSS
    plt.figure()
    plt.title("Loss During Training and Validating")
    x = np.arange(1, parser.epochs+1, dtype=np.int32)
    y_train = np.asarray(results_loss['loss']).astype('float64')
    y_val = np.asarray(results_loss['val_loss']).astype('float64')
    model_train = make_interp_spline(x, y_train)
    model_val = make_interp_spline(x, y_val)
    xs = np.linspace(1, parser.epochs, 500)
    ys_train = model_train(xs)
    ys_val = model_val(xs)
    plt.plot(xs, ys_train, label="Train")
    plt.plot(xs, ys_val, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('statistics/' + architecture + '/train_epoch_losses.tif')
    plt.show()
    # ACC
    plt.figure()
    plt.title("Accuracy During Training and Validating")
    x = np.arange(1, parser.epochs+1, dtype=np.int32)
    y_train = np.asarray(results_acc['train_acc']).astype('float64')
    y_val = np.asarray(results_acc['val_acc']).astype('float64')
    model_train = make_interp_spline(x, y_train)
    model_val = make_interp_spline(x, y_val)
    xs = np.linspace(1, parser.epochs, 500)
    ys_train = model_train(xs)
    ys_val = model_val(xs)
    plt.plot(xs, ys_train, label="Train")
    plt.plot(xs, ys_val, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('statistics/' + architecture + '/train_epoch_accuracy.tif')
    plt.show()

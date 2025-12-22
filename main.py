########################################################################################################################
#
# The main program is used for model training and testing.
#
########################################################################################################################

import  torch
import random
import numpy as np
import utils
from model import HGNN
import Train
import torch.optim as optim
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_fold_from_npy(fold_index, input_dir='Data/DNA_735_fold5/'):
    train_data = np.load(os.path.join(input_dir, f'train_fold_{fold_index}.npy'), allow_pickle=True)
    valid_data = np.load(os.path.join(input_dir, f'valid_fold_{fold_index}.npy'), allow_pickle=True)
    return train_data.tolist(), valid_data.tolist()

def mian():
    set_seed(1995) # 1995
    best_epochs = []
    valid_accs = []
    valid_spes = []
    valid_recalls = []
    valid_mccs = []
    valid_f1s = []
    valid_pres = []
    valid_aucs = []
    valid_pr_aucs = []
    flag = "test"
    Model = HGNN(n_feat=6229, hidden_dim=512, out_dim=256)


    optimizer = optim.Adam(Model.parameters(), lr=0.0001, weight_decay=1e-5) # 0.0001, 1e-4
    Model_path = "Modelpt/"

    if flag == "train":
        train_data, valid_data = load_fold_from_npy(1)
        train_dataSet = utils.ProDataset(train_data)
        valid_dataset = utils.ProDataset(valid_data)

        best_epoch, valid_mcc, val_acc, val_spe, val_pre, val_recall, val_f1, val_auc, val_pr_auc = Train.Train(Model, train_dataSet, valid_dataset, 10, optimizer, 1)# 10
        best_epochs.append(str(best_epoch))
        valid_mccs.append(valid_mcc)
        valid_accs.append(val_acc)
        valid_spes.append(val_spe)
        valid_pres.append(val_pre)
        valid_recalls.append(val_recall)
        valid_f1s.append(val_f1)
        valid_aucs.append(val_auc)
        valid_pr_aucs.append(val_pr_auc)
        print("Down!")
        print(f"MCC:{valid_mccs}")
        print(f"acc:{valid_accs}")
        print(f"spe:{valid_spes}")
        print(f"pre:{valid_pres}")
        print(f"recall:{valid_recalls}")
        print(f"f1:{valid_f1s}")
        print(f"auc:{valid_aucs}")
        print(f"auppr:{valid_pr_aucs}")
        print("\n\nBest epoch: " + " ".join(best_epochs))
        print("Average MCC of {} fold：{:.4f}".format(5, sum(valid_mccs) / 5))
        print("Average acc of {} fold：{:.4f}".format(5, sum(valid_accs) / 5))
        print("Average spe of {} fold：{:.4f}".format(5, sum(valid_spes) / 5))
        print("Average pre of {} fold：{:.4f}".format(5, sum(valid_pres) / 5))
        print("Average recall of {} fold：{:.4f}".format(5, sum(valid_recalls) / 5))
        print("Average f1 of {} fold：{:.4f}".format(5, sum(valid_f1s) / 5))
        print("Average auc of {} fold：{:.4f}".format(5, sum(valid_aucs) / 5))
        print("Average pr_auc of {} fold：{:.4f}".format(5, sum(valid_pr_aucs) / 5))
    else:
        file_path = 'example/exampleDNA.fasta'
        test_data = utils.load_data(file_path)
        test_dataset = utils.ProDataset(test_data)
        test_results = Train.Test(Model, test_dataset, Model_path)
        print(test_results)

if __name__ == '__main__':
      mian()





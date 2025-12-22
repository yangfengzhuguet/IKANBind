########################################################################################################################
#
# This program is used for model training and testing.
########################################################################################################################

import torch
import os
from sklearn.metrics import average_precision_score,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve,precision_recall_curve,auc,matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Model_Path = "Model/"

def metrics_calculate(y_true,y_pred,best_threshold = None):
    if best_threshold == None:
        best_mcc = 0
        best_threshold = 0

        for j in range(0, 100):
            threshold = j / 100000  # pls change this threshold according to your code
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]

    correct_samples = sum(a == b for a, b in zip(binary_pred, y_true))
    accuracy = correct_samples / len(y_true)

    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)
    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    results = {
        'accuracy':accuracy,
        'spe':spe,
        'precision': pre,
        'recall': recall,
        'f1':f1,
        'mcc': mcc,
        'auc':auc,
        'pr_auc':pr_auc,
        'thred':best_threshold
    }

    return results


def evaluate_and_report_metrics(y_true_tensor, y_scores_tensor, optimize_for='mcc', plot_curves=False):
    """
    评估模型性能并报告多种指标，可选择寻找最佳阈值和绘制曲线。

    Args:
        y_true_tensor (torch.Tensor): 真实标签的 PyTorch Tensor (例如 test_label_fc)。
        y_scores_tensor (torch.Tensor): 模型预测分数的 PyTorch Tensor (例如 test_score)。
        optimize_for (str): 寻找最佳阈值的优化目标，可以是 'f1' (F1-score) 或 'mcc' (MCC)。
                            默认为 'f1'。
        plot_curves (bool): 是否绘制 ROC 和 Precision-Recall 曲线。默认为 False。

    Returns:
        dict: 包含所有计算指标的字典，以及最佳阈值。
    """

    # 确保输入是 NumPy 数组，并且 y_true 是整数类型 (0或1)
    # y_true = np.array(y_true_tensor.detach().cpu()).astype(int)
    # y_scores = np.array(y_scores_tensor.detach().cpu())
    y_true = y_true_tensor
    y_scores = y_scores_tensor

    # --- 1. 计算与阈值无关的指标 (AUROC, AUPR) ---
    # 计算 ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_ = auc(fpr, tpr)

    # 计算 Precision-Recall 曲线
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall_curve, precision_curve)

    # --- 2. 寻找最佳阈值及在该阈值下的指标 ---
    best_metric_value = -1.0  # 用于比较 F1 或 MCC
    best_threshold = 0.5  # 默认一个初始阈值
    best_metrics_at_threshold = {}

    # 使用 roc_curve 提供的阈值进行搜索，这些阈值覆盖了所有可能的分类点
    # 添加 0.0 和 1.0 以确保覆盖所有极端情况
    thresholds_for_search = np.unique(np.concatenate(([0.0], roc_thresholds, [1.0])))

    for threshold in thresholds_for_search:
        # 将预测分数二值化
        binary_pred = (y_scores >= threshold).astype(int)

        # 检查二值化预测是否只有一类，这会导致某些指标计算失败
        if len(np.unique(binary_pred)) < 2:
            # 如果所有预测都是0或1，则MCC和F1可能为0或NaN，此处给一个低分
            current_metric_value = -2.0  # 确保不会被选为最佳
        else:
            # 计算当前阈值下的 F1-score 和 MCC
            current_f1 = f1_score(y_true, binary_pred, zero_division=0)
            current_mcc = matthews_corrcoef(y_true, binary_pred)

            if optimize_for == 'f1':
                current_metric_value = current_f1
            elif optimize_for == 'mcc':
                current_metric_value = current_mcc
            else:
                raise ValueError("`optimize_for` must be 'f1' or 'mcc'.")

        # 更新最佳阈值和最佳指标
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_threshold = threshold

            # 在找到更好的阈值时，计算并保存所有相关指标
            tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()

            # 避免除以零
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            best_metrics_at_threshold = {
                "accuracy": accuracy_score(y_true, binary_pred),
                "precision": precision_score(y_true, binary_pred, zero_division=0),
                "recall": recall_score(y_true, binary_pred, zero_division=0),
                "f1": current_f1,
                "mcc": current_mcc,
                "spe": specificity,  # 使用 'spe' 保持与你原始输出一致
                "threshold": best_threshold
            }

    # 如果 best_metrics_at_threshold 仍然为空 (例如，所有预测都一样)，则使用默认值
    if not best_metrics_at_threshold:
        binary_pred_default = (y_scores >= 0.5).astype(int)  # 使用默认阈值0.5
        tn, fp, fn, tp = confusion_matrix(y_true, binary_pred_default).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        best_metrics_at_threshold = {
            "accuracy": accuracy_score(y_true, binary_pred_default),
            "precision": precision_score(y_true, binary_pred_default, zero_division=0),
            "recall": recall_score(y_true, binary_pred_default, zero_division=0),
            "f1": f1_score(y_true, binary_pred_default, zero_division=0),
            "mcc": matthews_corrcoef(y_true, binary_pred_default),
            "spe": specificity,
            "threshold": 0.5  # 默认阈值
        }

        # 显示性能指标在图上 (可选)
        metrics_str = (f"Best Threshold: {best_metrics_at_threshold['threshold']:.4f}\n"
                       f"Accuracy: {best_metrics_at_threshold['accuracy']:.4f}\n"
                       f"Precision: {best_metrics_at_threshold['precision']:.4f}\n"
                       f"Recall: {best_metrics_at_threshold['recall']:.4f}\n"
                       f"Specificity: {best_metrics_at_threshold['spe']:.4f}\n"
                       f"MCC: {best_metrics_at_threshold['mcc']:.4f}\n"
                       f"F1 Score: {best_metrics_at_threshold['f1']:.4f}")
        # 尝试将文本放置在图的合适位置，可能需要根据实际图的范围调整
        plt.figure(2)  # 在PR曲线上添加文本
        plt.text(0.6, 0.2, metrics_str, bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

        plt.show()  # 显示所有图

    # --- 4. 打印并返回结果 ---
    print(f"\n--- Model Evaluation Results (Optimized for {optimize_for}) ---")
    print(f"AUROC: {roc_auc_:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"Best Threshold (for {optimize_for}): {best_metrics_at_threshold['threshold']:.4f}")
    print(f"Accuracy: {best_metrics_at_threshold['accuracy']:.4f}")
    print(f"Precision: {best_metrics_at_threshold['precision']:.4f}")
    print(f"Recall: {best_metrics_at_threshold['recall']:.4f}")
    print(f"Specificity (SPE): {best_metrics_at_threshold['spe']:.4f}")
    print(f"MCC: {best_metrics_at_threshold['mcc']:.4f}")
    print(f"F1 Score: {best_metrics_at_threshold['f1']:.4f}")
    print("-------------------------------------------------")

    final_results = {
        'accuracy': best_metrics_at_threshold['accuracy'],
        'spe': best_metrics_at_threshold['spe'],
        'precision': best_metrics_at_threshold['precision'],
        'recall': best_metrics_at_threshold['recall'],
        'f1': best_metrics_at_threshold['f1'],
        'mcc': best_metrics_at_threshold['mcc'],
        'auc': roc_auc_,
        'pr_auc': aupr,
        'best_threshold': best_metrics_at_threshold['threshold']
    }
    return final_results

def train_au(label, pred):
    auc = roc_auc_score(label, pred)
    pr_auc = average_precision_score(label, pred)
    return auc, pr_auc

def show_auc(label, pre_score):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc, aupr

def train_one_epoch(model, train_loader, optimizer, epoch, all_epochs):
    epoch_loss_train = 0.0
    n = 0
    criterion = nn.functional.binary_cross_entropy
    train_pred = []
    train_true = []

    for batch_idx, (n_feat, label, G, H) in enumerate(train_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                n_feat = n_feat.to(device).squeeze(0)
                G = G.to(device).float().squeeze(0)
                H = H.to(device).float().squeeze(0)
                label = label.to(device)
                label = label.view(-1, 1).float()
        y_pred = model(n_feat, G, H)
        train_pred += [pred for pred in y_pred.cpu().detach().numpy().flatten()]
        train_true += list(label.cpu().detach().numpy().flatten())
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        auc_, aupr_ = show_auc(label, y_pred)
        epoch_loss_train += loss.item()
        n += 1
        res = '\t'.join([
            'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
            'Iter: [%d/%d]' % (batch_idx + 1, len(train_loader)),
            'Loss %.4f' % (loss),
            'AUC %.4f' % (auc_),
            'AUPR %.4f' % (aupr_)
        ])
        print('\n', res)
    auc_, aupr_ = train_au(torch.tensor(train_true), torch.tensor(train_pred))
    print(f"training sequence size:{n}, AUC:{auc_:.4f}, AUPR:{aupr_:.4f}")
    epoch_loss_train = epoch_loss_train / n

    return epoch_loss_train


def evaluate(model, valid_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    criterion = nn.functional.binary_cross_entropy
    for batch_idx, (n_feat, label, G, H) in enumerate(valid_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                n_feat = n_feat.to(device).squeeze(0)
                G = G.to(device).float().squeeze(0)
                H = H.to(device).float().squeeze(0)
                label = label.to(device)
                label = label.view(-1,1).float()
            y_pred = model(n_feat, G, H)
            loss = criterion(y_pred, label)
            y_pred = y_pred.cpu().detach().numpy().flatten()
            label = label.cpu().detach().numpy().flatten()
            valid_pred += [pred for pred in y_pred]
            valid_true += list(label)
            epoch_loss += loss.item()
            n += 1
    epoch_loss_valid = epoch_loss / n
    return epoch_loss_valid, valid_true, valid_pred

def Train(Model, train_dataSet, valid_dataSet, epochs, optimizer, fold):
    Model.to(device)
    # Data loaders
    train_loader = DataLoader(train_dataSet, batch_size=1, pin_memory=(torch.cuda.is_available()), num_workers=0)
    valid_loader = DataLoader(valid_dataSet, batch_size=1, pin_memory=(torch.cuda.is_available()), num_workers=0)
    best_epoch = 0
    best_val_acc = 0
    best_val_spe = 0
    best_val_pre = 0
    best_val_recall = 0
    best_val_f1 = 0
    best_val_mcc = 0
    best_val_auc = 0
    best_val_prauc = 0
    for epoch in range(epochs):
        print("--------start train:" + str(epoch+1) + "--------")
        Model.train()
        begin_time = time.time()
        train_loss_avgepoch = train_one_epoch(Model, train_loader, optimizer, epoch, epochs)
        end_time = time.time()
        run_time = end_time - begin_time
        print("--------Evaluate Valid set--------")
        valid_loss_avgepoch, valid_true, valid_pred = evaluate(Model, valid_loader)
        auc_1, aupr_1 = train_au(valid_true, valid_pred)
        print("auc:{}, aupr:{}".format(auc_1, aupr_1))
        if best_val_auc < auc_1:
            valid_results = evaluate_and_report_metrics(valid_true, valid_pred)
            best_epoch = epoch + 1
            best_val_mcc = valid_results['mcc']
            best_val_acc = valid_results['accuracy']
            best_val_spe = valid_results['spe']
            best_val_pre = valid_results['precision']
            best_val_recall = valid_results['recall']
            best_val_f1 = valid_results['f1']
            best_val_auc = valid_results['auc']
            best_val_prauc = valid_results['pr_auc']

            print('-' * 20, "new best pr_auc:{0}".format(best_val_auc), '-' * 20)

    return best_epoch, best_val_mcc, best_val_acc, best_val_spe, best_val_pre, best_val_recall, best_val_f1, best_val_auc, best_val_prauc




def Test(Model, test_dataSet, Model_Path): # 计算指标，原版test
   """
   Tests a trained model on a given test dataset and reports performance metrics.

   Args:
       Model: The trained PyTorch model.
       test_dataSet: The test dataset (instance of your Dataset class).
       fold: The fold number (used for loading the correct model weights).
       Model_Path: The path to the directory where the model weights are saved.
       device: The device to use for testing (e.g., 'cuda' or 'cpu').

   Returns:
       A dictionary containing the test results (metrics).
   """

   Model.to(device)
   Model.eval()  # Set the model to evaluation mode

   # Create a DataLoader for the test dataset
   test_loader = DataLoader(test_dataSet, batch_size=1, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers=0)

   # Load the best model weights
   model_path = os.path.join(Model_Path, 'ModelDNA.pkl')
   Model.load_state_dict(torch.load(model_path, map_location=device))  # Load onto the correct device
   print(f"Loaded model weights from: {model_path}")

   # Perform evaluation
   test_true = []
   test_pred = []
   n=0
   for batch_idx, (n_feat, label, G, H) in enumerate(test_loader):
       with torch.no_grad():  # Disable gradient calculation during testing
           if torch.cuda.is_available():
               n_feat = n_feat.to(device).squeeze(0)
               G = G.to(device).float().squeeze(0)
               H = H.to(device).float().squeeze(0)
               label = label.to(device)
               label = label.view(-1,1).float()
           y_pred = Model(n_feat, G, H)
           y_pred = y_pred.cpu().detach().numpy().flatten()
           label = label.cpu().detach().numpy().flatten()
           test_pred += [pred for pred in y_pred]
           test_true += list(label)
           n += 1
   test_results = evaluate_and_report_metrics(test_true, test_pred)
   auc_1, aupr_1 = train_au(test_true, test_pred)

   print("Test Results:")
   print(f"  Accuracy: {test_results['accuracy']:.4f}")
   print(f"  MCC: {test_results['mcc']:.4f}")
   print(f"  F1-Score: {test_results['f1']:.4f}")
   print(f"  AUC: {test_results['auc']:.4f}")
   print(f"  AUPR: {test_results['pr_auc']:.4f}")
   print(f"  Specificity: {test_results['spe']:.4f}")
   print(f"  Precision: {test_results['precision']:.4f}")
   print(f"  Recall: {test_results['recall']:.4f}")

   test_results['auc'] = auc_1
   test_results['pr_auc'] = aupr_1

   return test_results







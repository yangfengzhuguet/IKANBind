########################################################################################################################
#
# Purpose: This program reads and loads inputs data and passes them to the inference program
########################################################################################################################

import random
from torch.utils.data import Dataset
import numpy as np

THREADHOLD = 0.4

def load_data(file_path):
   """
   从 FASTA 文件中加载数据。

   Args:
       file_path (str): FASTA 文件路径。

   Returns:
       list: 包含 (protein_id, sequence, label) 元组的列表。
   """
   data = []
   with open(file_path, 'r') as f:
       lines = f.readlines()
       for i in range(0, len(lines), 3):
           protein_id = lines[i].strip()[1:]
           sequence = lines[i + 1].strip()
           label_str = lines[i + 2].strip()
           # Convert label string to list of integers
           label = [int(x) for x in label_str.strip('[]').split(', ')]
           data.append((protein_id, sequence, label))
   return data

def create_folds(data, n_folds=5):
   """
   将数据随机划分为 n_folds 个 folds。

   Args:
       data (list): 包含 (protein_id, sequence, label) 元组的列表。
       n_folds (int): fold 的数量。

   Returns:
       list: 包含 n_folds 个列表的列表，每个列表代表一个 fold。
   """
   random.shuffle(data)
   fold_size = len(data) // n_folds
   folds = []
   for i in range(n_folds):
       start = i * fold_size
       end = (i + 1) * fold_size
       if i == n_folds - 1:
           end = len(data)  # 最后一个 fold 包含剩余的数据
       folds.append(data[start:end])
   return folds

def get_train_test_folds(folds, fold_index):
   """
   根据 fold_index 获取训练 fold 和测试 fold。

   Args:
       folds (list): 包含 n_folds 个列表的列表，每个列表代表一个 fold。
       fold_index (int): 要用作测试 fold 的 fold 的索引。

   Returns:
       tuple: 包含训练数据和测试数据的元组。
   """
   valid_data = folds[fold_index]
   train_data = []
   for i, fold in enumerate(folds):
       if i != fold_index:
           train_data.extend(fold)
   return train_data, valid_data

def getID(data):
    """
    从数据列表中提取 IDs, labels, 和 sequences。

    Args:
        data (list): 包含 (protein_id, sequence, label) 元组的列表。

    Returns:
        tuple: 包含 IDs (列表), labels (列表), sequences (列表) 的元组。
    """
    IDs = []
    labels = []
    sequences = []
    for protein_id, sequence, label in data:
        IDs.append(protein_id)
        labels.append(label)
        sequences.append(sequence)
    return IDs, labels, sequences

def protein_to_one_hot(protein_sequence):
    # 定义蛋白质序列的字母表
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 创建空的one-hot编码矩阵
    sequence_length = len(protein_sequence)
    alphabet_size = len(amino_acids)
    one_hot_matrix = np.zeros((sequence_length, alphabet_size))

    # 对每个氨基酸进行编码
    for i, amino_acid in enumerate(protein_sequence):
        if amino_acid in amino_acids:
            # 将当前氨基酸的位置索引设为1
            index = amino_acids.index(amino_acid)
            one_hot_matrix[i, index] = 1

    return one_hot_matrix



class ProDataset(Dataset):
    def __init__(self,  data_path):
        self.IDs,self.labels,self.seqs = getID(data_path)
        self.data_path = data_path
        self.protein_ids = self.IDs

    def __getitem__(self, index):
        protein_id = self.IDs[index]
        seq = self.seqs[index]


        # test
        n_feat = np.load(f'/example/pdb_graph_8aiDNA_example/n_feat/{protein_id}.npy')
        label = np.load(f'/example/pdb_graph_8aiDNA_example/label/{protein_id}.npy')
        label_str = str(label) # for lable 01010
        label = np.array([int(c) for c in label_str]) # for label 01010
        H = np.load(f'/example/pdb_graph_8aiDNA_example/H/{protein_id}.npy')
        G = np.load(f'/example/pdb_graph_8aiDNA_example/G/{protein_id}.npy')

        if len(label) != len(seq):
            print("--------label and seq have different length--------:{}".format(protein_id))

        return n_feat, np.array(label), G, H

    def __len__(self):
        return len(self.IDs)






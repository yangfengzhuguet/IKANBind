########################################################################################################################
#
# Purpose: This program is used for the following purposes.
# 1.concatenate all features
# 2.Construct a hypergraph based on the distance information obtained from the PDB file.
#
########################################################################################################################



import numpy as np
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


def calculate_hypergraph_matrices(C_distance_matrix, SC_distance_matrix, d_C, jaccard_weight=0.5, structure_weight=0.5, gamma=1.0):
    """
    根据侧链质心距离和CA原子距离计算超图关联矩阵和权重矩阵，并去除重复超边。
    其中超边权重矩阵W通过Jaccard相似性和结构相似度加权构建，并保证对称性。

    Args:
        C_distance_matrix: (N, N) 质心距离矩阵。
        SC_distance_matrix: (N, N) 侧链质心距离矩阵。
        d_C: 质心距离阈值。

    Returns:
        H_unique: (N, E) 去重后的超图关联矩阵。
        W: (E, E) 对称归一化超边权重矩阵。
        hyperedge_mapping: dict，原始超边索引映射到去重后索引。
    """
    num_nodes = C_distance_matrix.shape[0]
    H = np.zeros((num_nodes, num_nodes), dtype=np.float32)  # 初始化超图关联矩阵

    # 1. 构建超边：每个节点及其SC距离小于阈值的邻居构成一个超边
    hyperedges = []
    for i in range(num_nodes):
        hyperedge_nodes = [i]
        for j in range(num_nodes):
            if i != j and C_distance_matrix[i, j] < d_C:
                hyperedge_nodes.append(j)
        hyperedges.append(hyperedge_nodes)

    # 2. 构建初始超图关联矩阵H (节点×超边)
    for i, hyperedge_nodes in enumerate(hyperedges):
        for node in hyperedge_nodes:
            H[node, i] = 1

    # 3. 去除重复超边
    unique_hyperedges = []
    hyperedge_mapping = {}
    H_unique_cols = []
    for i in range(H.shape[1]):
        col_tuple = tuple(H[:, i].tolist())
        if col_tuple not in unique_hyperedges:
            unique_hyperedges.append(col_tuple)
            hyperedge_mapping[i] = len(unique_hyperedges) - 1
            H_unique_cols.append(H[:, i])
        else:
            # 映射到已存在超边索引
            hyperedge_mapping[i] = unique_hyperedges.index(col_tuple)
    H_unique = np.stack(H_unique_cols, axis=1)  # (N, E_unique)

    num_unique_hyperedges = H_unique.shape[1]
    W = np.zeros((num_unique_hyperedges, num_unique_hyperedges), dtype=np.float32)

    for i in range(num_unique_hyperedges):
        nodes_i = set(np.where(H_unique[:, i] == 1)[0])
        for j in range(i, num_unique_hyperedges):  # 利用对称性，只算一半
            nodes_j = set(np.where(H_unique[:, j] == 1)[0])

            # 计算Jaccard相似性
            intersection = nodes_i & nodes_j
            union = nodes_i | nodes_j
            if len(union) == 0:
                jaccard = 0.0
            else:
                jaccard = len(intersection) / len(union)

            # 计算结构相似度（共享节点的侧链距离高斯核函数均值）
            if len(intersection) > 1:
                distances = [SC_distance_matrix[n1, n2]
                             for idx1, n1 in enumerate(intersection)
                             for n2 in list(intersection)[idx1 + 1:]]
                if len(distances) > 0:
                    structure_sim = np.mean(np.exp(-np.array(distances) ** 2 / (2 * gamma ** 2)))
                else:
                    structure_sim = 0.0
            else:
                structure_sim = 0.0 if len(intersection) == 0 else 1.0  # 单节点交集，给1.0

            # 计算权重
            W_val = jaccard_weight * jaccard + structure_weight * structure_sim
            W[i, j] = W_val
            W[j, i] = W_val  # 对称赋值

    # 归一化W矩阵
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
    np.fill_diagonal(W, 1.0)

    return H_unique, W, hyperedge_mapping


def caculate_G(H, W):
    # 计算节点度矩阵 D_v 和超边度矩阵 D_e
    D_v = np.diag(np.sum(H, axis=1))  # 节点度矩阵
    D_e = np.diag(np.sum(H, axis=0))  # 超边度矩阵
    # 计算 D_v 的逆矩阵
    D_v_inv = np.linalg.inv(D_v + np.eye(D_v.shape[0]) * 1e-6)  # 添加小正则化项以避免奇异矩阵

    # 计算 D_e 的逆矩阵
    D_e_inv = np.linalg.inv(D_e + np.eye(D_e.shape[0]) * 1e-6)  # 添加小正则化项以避免奇异矩阵
    G = D_v_inv @ H @ W @ D_e_inv @ H.T @ D_v_inv
    return G

def get_information(file_path, counter):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label_str = lines[i + 2].strip()
        label = label_str

        # node feature
        seq_ = protein_to_one_hot(sequence)  # one-hot encoding(Lx20)
        PC = np.load(f'feature/feat/PhyChem_RNA_93/{protein_id}.npy') # Lx4
        dssp = np.load(f'feature/feat/RNA93_esmfold/DSSP_RNA_93/{protein_id}.npy')  # Lx14
        pssm = np.load(f'feature/feat/PSSM_RNA_93/{protein_id}.npy')  # Lx20
        hmm = np.load(f'feature/feat/HMM_RNA_93/{protein_id}.npy')  # Lx20
        esm_2 = np.load(f'feature/Sequence_Feature/ESM-2/RNA_93_Test/t48_15B/{protein_id}.npy')  # Lx5120
        # protTrans = np.load(f'feature/Sequence_Feature/protTrans/RNA_142_Test/prot_t5_xl_uniref50/{protein_id}.npy')  # Lx1024
        protTrans = np.load(f'feature/Sequence_Feature/protTrans/RNA_93_Test/{protein_id}.npy')  # Lx1024
        af_7 = np.load(f'feature/feat/af_7_RNA_93/{protein_id}.npy')  # Lx7

        if len(label) != len(sequence):
            print("--------label and seq have different length--------:{}".format(protein_id))

        n_fea = np.concatenate([seq_, dssp, pssm, hmm, PC, esm_2, protTrans, af_7], axis=1).astype(np.float32) #

        # construct hypergraph matrix H based distance matrix SC and Assigning weights to hyperedges based on the CA matrix
        C_matrix = np.load(f'A_data/ori_data/pdb_distance_Matrix_RNA_esmfold/C/{protein_id}.npy')
        SC_matrix = np.load(f'A_data/ori_data/pdb_distance_Matrix_RNA_esmfold/SC/{protein_id}.npy')
        H, W, _ = calculate_hypergraph_matrices(C_matrix, SC_matrix, 8)

        # obtain G
        G = caculate_G(H, W)
        np.save("pdb_graph_8aiRNA93/n_feat/{}.npy".format(protein_id), n_fea)  # Replace with your own directory
        np.save("pdb_graph_8aiRNA93/H/{}.npy".format(protein_id), H)
        np.save("pdb_graph_8aiRNA93/G/{}.npy".format(protein_id), G)
        np.save("pdb_graph_8aiRNA93/label/{}.npy".format(protein_id), np.array(label))
        print(f"{protein_id} done!")
        counter += 1
    return counter
counter = 0
file_path = "FastA file input path (protein ID, sequence, tag)"
toral = get_information(file_path, counter)
print(f"The protein sequence is {toral}")
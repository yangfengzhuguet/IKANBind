########################################################################################################################
#
# Purpose: The program obtains the C and SC distances between residues based on the PDB file.
#
########################################################################################################################


import numpy as np
from scipy.spatial import distance_matrix
import os, sys, argparse
import pandas as pd
##################################################################################################

def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0, 0, 0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1],
         'CZ2': [0, 1, 1], 'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features


def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path, 'r')
    # 初始空的 DataFrame，定义所需的列
    pdb_res_df = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor', 'mass' , 'is_sidechain'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51,
                            'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23, 'HG': 200.6,
                            'MN': 55,
                            'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79, 'NI': 58.7}

    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count += 1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N', 'CA', 'C', 'O', 'H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5, 0.5, 0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom': line[12:16].strip(), 'atom_type': atom_type, 'res': res,
                 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'occupancy': float(line[54:60]),
                 'B_factor': float(line[60:66]), 'mass' : Relative_atomic_mass[atom_type], 'is_sidechain': is_sidechain,
                 'charge': atom_fea[0], 'num_H': atom_fea[1], 'ring': atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))

            # print(tmps)
            tmps_selected = pd.DataFrame([tmps[['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor', 'mass' , 'is_sidechain']]],
                                          columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor', 'mass' , 'is_sidechain'])
            # 累加到 pdb_res_df
            pdb_res_df = pd.concat([pdb_res_df, tmps_selected], ignore_index=True)

        if line.startswith('TER'):
            break

    # print(pdb_res_df)
    # assert 1==0

    return pdb_res_df, res_id_list


def cal_PDBDF(seq_id, PDB_chain_dir):
    file_path = PDB_chain_dir + '/{}'.format(seq_id)    +".pdb"

    with open(file_path, 'r') as f:
        text = f.readlines()
    if len(text) == 1:
        print('ERROR: PDB {} is empty.'.format(seq_id))

    pdb_DF, res_id_list = get_pdb_DF(file_path)
    # with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'wb') as f:
    #     pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)

    return pdb_DF, res_id_list

# 计算欧氏距离矩阵
def calculate_distance_matrix(coordinates):
    return distance_matrix(coordinates, coordinates)



def cal_Psepos(seq_id, pdb_DF, res_id_list, Dataset_dir,psepos_path):

    pdb_res_i, res_id_list = pdb_DF, res_id_list

    res_CA_pos = []
    res_centroid = []
    res_sidechain_centroid = []
    res_types = []
    for res_id in res_id_list:
        res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
        res_types.append(res_type)

        res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
        xyz = np.array(res_atom_df['xyz'].tolist())
        masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
        centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
        res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

        try:
            CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
        except IndexError:
            print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id, res_id))
            CA = centroid

        res_CA_pos.append(CA)
        res_centroid.append(centroid)

        if len(res_sidechain_atom_df) == 0:
            res_sidechain_centroid.append(centroid)
            # print('~'*10)
        else:
            xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
            masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
            sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_centroid.append(sidechain_centroid)


    # 得到了三个对应的序列坐标


    # 以下是三个距离矩阵
    np.save(psepos_path + '/CA/'+ seq_id+'.npy', res_CA_pos)
    np.save(psepos_path + '/C/'+ seq_id+'.npy', res_centroid)
    np.save(psepos_path + '/SC/'+ seq_id+'.npy', res_sidechain_centroid)

    # # 验证读取（可选）
    # loaded_data = np.load(psepos_path + '/C/'+ seq_id+'.npy')
    # print(loaded_data)


    # 计算欧氏距离矩阵
    res_CA_pos = calculate_distance_matrix(res_CA_pos)
    res_centroid = calculate_distance_matrix(res_centroid)
    res_sidechain_centroid = calculate_distance_matrix(res_sidechain_centroid)

    # 以下是三个距离矩阵
    np.save(Dataset_dir + '/CA/'+ seq_id+'.npy', res_CA_pos)
    np.save(Dataset_dir + '/C/'+ seq_id+'.npy', res_centroid)
    np.save(Dataset_dir + '/SC/'+ seq_id+'.npy', res_sidechain_centroid)

    return

# 主函数
def to_distance(fa_file):
    with open(fa_file, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label = lines[i + 2].strip()

        print('1.Extract the PDB information.')
        #   序列， pdb文件夹
        # pdb_DF, res_id_list=cal_PDBDF(protein_id, "pdb")  #       pdb        pdb_335_60
        pdb_DF, res_id_list=cal_PDBDF(protein_id, "/home/lichangyong/documents/Protein-Nucleiv/Our_Model/PDB_file/ESMFold_RNA577/pdb_file/")  #      for Datasets_422

        print('2.calculate the pseudo positions.')

        #   序列， 中间文件1，中间文件2，距离矩阵保存文件夹， psepos保存位置
        cal_Psepos(protein_id, pdb_DF, res_id_list, "pdb_distance_Matrix_577","psepos")



    print("Done!!!")



def main():
    parser = argparse.ArgumentParser(description="deep learning 6mA analysis in rice genome")
    # parser.add_argument("--path1", type=str, default="DNA-735-Train.fasta", help="DNA-735-Train.fasta", required=True)
    # args = parser.parse_args()
    path = "File input directory"

    path1 = os.path.abspath(path)

    if not os.path.exists(path1):
        print("The csv benchmark_data not exist! Error\n")
        sys.exit()
    to_distance(path1)
    print("you did it")

if __name__ == "__main__":
    main()


#   python pdb_to_distance_2.py --path1 Test_60.fa

#   python pdb_to_distance_2.py --path1 Train_422-224.fa


#
#   python pdb_to_distance_2.py --path1 /home/lichangyong/documents/sgc_backup/sgc_p3/Feature/PDB_pretrained/train422_withoutpdb.fa




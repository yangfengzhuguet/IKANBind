########################################################################################################################
#
# Purpose: This program obtains evolutionary information HMM.
# shape：Lx20
#
########################################################################################################################

import os
counter = 0

HHblits = '/home/lichangyong/documents/sgc_backup/sgc_p3/Feature/Bio_fea/hh-suite/build/bin/hhblits'
HHblits_DB = "/home/lichangyong/documents/Protein-Nucleiv(zyf)/uniclust30_2018_08/uniclust30_2018_08"
file="File input directory"

hmm_ori_dir = "hmm_ori"  # HMM 文件保存的文件夹

with open(file, 'r') as file:
    lines = file.readlines()

for i in range(0, len(lines), 3):
    counter += 1
    protein_id = lines[i].strip()[1:]   #   获取蛋白质ID
    sequence = lines[i + 1].strip()
    label = lines[i + 2].strip()

    # 转义 protein_id 中的括号
    protein_id_escaped = protein_id.replace("(", "\\(").replace(")", "\\)")

    # fa_file_path="dataset_fasta_335_60/"+protein_id+".fa"  #  蛋白质序列            将整个fasta文件拆分为：一个蛋白质一个.fa文件，并放在一个文件夹下调用
    fa_file_path="dataset_fasta/"+protein_id_escaped+".fasta"

    # 构建 HMM 文件路径
    hhm_file_path = os.path.join(hmm_ori_dir, protein_id_escaped + ".hhm")

    # 检查 HMM 文件是否已经存在
    if os.path.exists(hhm_file_path):
        print(f"HMM file already exists for {protein_id}. Skipping...")
        continue  # 跳过当前序列的处理

    print(protein_id,"hhm file making...")
    # 引用 HHblits 和 HHblits_DB 路径
    order="\"" + HHblits + "\"" + " -d " + "\"" + HHblits_DB + "\"" + " -cpu 3 -i "+ fa_file_path+" -ohhm "+ hhm_file_path
    #   -d 数据库文件位置  -cpu 线程数    -i  蛋白质序列   -ohhm 保存位置；
    print(f"Executing command: {order}")  # 打印命令，方便调试
    os.system(order)


print(counter)

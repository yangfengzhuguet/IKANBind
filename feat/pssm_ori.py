########################################################################################################################
#
# Purpose: This program obtains evolutionary information PSSM.
# shape：Lx20
#
########################################################################################################################



import os

PSIBLAST = "/home/lichangyong/documents/sgc_backup/sgc_p3/Feature/Bio_fea/ncbi-blast-2.14.1+/bin/psiblast"
PSIBLAST_DB = "/home/lichangyong/documents/db/uniref90.fasta"
counter = 0

file="File input directory"

# 确保 pssm_ori 文件夹存在
pssm_dir = "pssm_ori"
if not os.path.exists(pssm_dir):
   os.makedirs(pssm_dir)

with open(file, 'r') as file:
    lines = file.readlines()

for i in range(0, len(lines), 3):
    protein_id = lines[i].strip()[1:]
    sequence = lines[i + 1].strip()
    label = lines[i + 2].strip()
    counter += 1

    fa_file_path = "dataset_fasta/" + protein_id + ".fasta"
    pssm_file_path = os.path.join(pssm_dir, protein_id + ".pssm")  # 构建 pssm 文件路径

    # 检查 pssm 文件是否已存在
    if os.path.exists(pssm_file_path):
        print(f"PSSM file already exists for {protein_id}. Skipping...")
        continue  # 跳过当前序列

    fa_file_path="dataset_fasta/"+protein_id+".fasta"
    print(protein_id,"pssm file making...")
    order=PSIBLAST+" -db "+PSIBLAST_DB+" -evalue 0.001 -num_iterations 3 -num_threads 4 -query "+ fa_file_path+" -out_ascii_pssm pssm_ori/"+protein_id+".pssm"
    os.system(order)

print(counter)



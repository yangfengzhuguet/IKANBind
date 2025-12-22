########################################################################################################################
#
# Purpose: The program generates a corresponding DSSP file based on the PDB file.
# shape：Lx14
#
########################################################################################################################


import os

os.system("chmod +x dssp")
file="File input directory"
counter = 0

# 确保 dssp_ori 文件夹存在
dssp_dir = "dssp_ori"
if not os.path.exists(dssp_dir):
   os.makedirs(dssp_dir)

with open(file, 'r') as file:
   lines = file.readlines()

for i in range(0, len(lines), 3):
   protein_id = lines[i].strip()[1:]
   sequence = lines[i + 1].strip()
   label = lines[i + 2].strip()
   counter += 1

   dssp_file_path = os.path.join(dssp_dir, protein_id + ".dssp")  # 构建 dssp 文件路径

   # 检查 dssp 文件是否已存在
   if os.path.exists(dssp_file_path):
       print(f"DSSP file already exists for {protein_id}. Skipping...")
       continue  # 跳过当前序列

   pdb_file_path = "Input directory for protein sequence PDB files/"+protein_id+".pdb"

   order="./dssp -i " + pdb_file_path + " -o " + dssp_file_path
   print(order)
   os.system(order)

print(counter)











'''


    ./dssp -i feature/Alphafold2_pdb_and_single_frature/3zeuD/ranked_0.pdb  -o feature/dssp_ori/3ZEUD.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4iu2A/ranked_0.pdb  -o feature/dssp_ori/4IU2A.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4kt3B/ranked_0.pdb  -o feature/dssp_ori/4KT3B.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/2yclA/ranked_0.pdb  -o feature/dssp_ori/2YCLA.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/3axjA/ranked_0.pdb  -o feature/dssp_ori/3AXJA.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/3axjB/ranked_0.pdb  -o feature/dssp_ori/3AXJB.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/3vrdB/ranked_0.pdb  -o feature/dssp_ori/3VRDB.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/3zr4B/ranked_0.pdb  -o feature/dssp_ori/3ZR4B.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4hffA/ranked_0.pdb  -o feature/dssp_ori/4HFFA.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4hffB/ranked_0.pdb  -o feature/dssp_ori/4HFFB.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4iu2B/ranked_0.pdb  -o feature/dssp_ori/4IU2B.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4kt3A/ranked_0.pdb  -o feature/dssp_ori/4KT3A.dssp

    ./dssp -i feature/Alphafold2_pdb_and_single_frature/4m70H/ranked_0.pdb  -o feature/dssp_ori/4M70H.dssp


    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/2f91A.pdb  -o feature/dssp_ori/2f91A.dssp
    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/1zbdA.pdb  -o feature/dssp_ori/1zbdA.dssp
    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/2g77B.pdb  -o feature/dssp_ori/2g77B.dssp
    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/2r25B.pdb  -o feature/dssp_ori/2r25B.dssp
    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/2v6xB.pdb  -o feature/dssp_ori/2v6xB.dssp
    ./dssp -i feature/emsfold_pdb_6dim_4_dssp/1fc2D.pdb  -o feature/dssp_ori/1fc2D.dssp

'''
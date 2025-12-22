########################################################################################################################
#
# Purpose:This program is used to extract the physicochemical properties of amino acids.
# shape Lx4
#
########################################################################################################################


import numpy as np
from aaindex import aaindex1
import pandas as pd

# 加载全部AAindex1标度
# aaindex1.download_aaindex1()
aa_data = aaindex1.parse_aaindex()

# 举例：选取几个AAindex用于特征
selected_props = {
    'hydrophobicity': 'KYTJ820101',  # Kyte-Doolittle
    'charge': 'KLEP840101',
    'polarity': 'GRAR740102',
    # 'volume': 'FAUJ880103',
    'flexibility': 'BHAR880101',
}

# 构建每种氨基酸的理化特征映射表
residues = 'ARNDCQEGHILKMFPSTWYV'
feature_table = {}

for aa in residues:
    feature_table[aa] = []
    for name, aid in selected_props.items():
        val = aa_data[aid]["values"].get(aa)
        # if val is None:
        #     val = 0.0  # 缺失处理
        feature_table[aa].append(val)

feature_df = pd.DataFrame(feature_table, index=selected_props.keys()).T
# print(feature_df)

fasta_data = "File input directory"
with open(fasta_data, "r") as file:
    lines = file.readlines()
counter = 0
for i in range(0, len(lines), 3):
    protein_id = lines[i].strip()[1:]
    sequence = lines[i + 1].strip()
    label = lines[i + 2].strip()
    all_records = []
    pp = 0
    for i, aa in enumerate(sequence):
        if aa not in feature_table:
            print(f"--------error! the {aa} is not in the feature table!--------")
            continue  # 跳过不常见或非标准残基
        features = feature_table[aa]
        record = {
            "protein": protein_id,
            "residue_idx": i,
            "aa": aa
            # "label": label[i],
        }
        for j, feat_name in enumerate(selected_props.keys()):
            record[feat_name] = features[j]
        all_records.append(record)
        df = pd.DataFrame(all_records)
        pp = df.loc[:,['hydrophobicity', 'charge', 'polarity', 'flexibility']]
        pp = pp.to_numpy()
    np.save("File output directory/"+protein_id+".npy", pp)
    print(f"{protein_id} is done!")
    counter += 1
    # print(pp)
print(f"The count of {counter} proteins are done!")

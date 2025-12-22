########################################################################################################################
#
# Purpose: Obtaining the ESM2 embedding of the protein language model.
# The program will automatically cache ESM2 parameters locally.
#shape：Lx5120
#
########################################################################################################################



import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
import torch


def ESM2(file, counter):
    with open(file, 'r') as file:
        lines = file.readlines()

    model_file = "facebook/esm2_t48_15B_UR50D"

    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = EsmForMaskedLM.from_pretrained(model_file)

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label = lines[i + 2].strip()
        lenn = len(label)

        # Calculating the length of a number in a label
        label_length = 0
        for char in label:
            if char.isdigit():
                label_length += 1

        # Ensure sequence and label lengths are consistent
        if len(sequence) != label_length:
            print(
                f" {protein_id} : ({len(sequence)}) != ({label_length}) ################################################################# ")
            continue

        seq = ""
        for i in range(label_length):
            seq = seq + sequence[i] + " "

        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True)
            hidden_states = embeddings.hidden_states
        node_embeddings = hidden_states[-1][0,1:-1].detach().cpu().numpy()
        print(np.array(node_embeddings).shape)

        np.save(protein_id + '.npy', node_embeddings)
        print(f'######## {protein_id} down ########')
        counter += 1
    return counter

path = 'File input directory'
counter = 0
sequence_counter = ESM2(path, counter)
print(f'There are {sequence_counter} protein sequences in total!')

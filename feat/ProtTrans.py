########################################################################################################################
#
# Purpose: Obtaining the protTrans embedding of the protein language model
# The program will automatically cache protTrans parameters locally.
# shape Lx1024
#
########################################################################################################################



import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer
import gc
import numpy as np

def read_sequence(file):
    sequence_dict = dict()
    with open(file, 'r') as file:
        lines = file.readlines()
    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()
        if protein_id.startswith('>'):
            protein_id = protein_id[1:]
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
        sequence_dict[protein_id] = seq

    return sequence_dict

def embed_dataset(seq, shift_left = 0, shift_right = -1):
  with torch.no_grad():
    ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True, is_split_into_words=True,
                                      return_tensors="pt")
    embedding = model(input_ids=ids['input_ids'].to(device))[0]
    embedding= embedding[0].detach().cpu().numpy()[shift_left:shift_right]
  return embedding


model_name = "Rostlab/prot_t5_xl_uniref50"

if "t5" in model_name:
  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
  model = T5EncoderModel.from_pretrained(model_name)
elif "albert" in model_name:
  tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = AlbertModel.from_pretrained(model_name)
elif "bert" in model_name:
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = BertModel.from_pretrained(model_name)
elif "xlnet" in model_name:
  tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = XLNetModel.from_pretrained(model_name)
else:
  print("Unkown model name")

gc.collect()
print("Number of model parameters is: " + str(int(sum(p.numel() for p in model.parameters())/1000000)) + " Million")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = model.to(device)
model = model.eval()
if torch.cuda.is_available():
  model = model.half()


if "t5" in model_name:
  shift_left = 0
  shift_right = -1
elif "bert" in model_name:
  shift_left = 1
  shift_right = -1
elif "xlnet" in model_name:
  shift_left = 0
  shift_right = -2
elif "albert" in model_name:
  shift_left = 1
  shift_right = -1
else:
  print("Unkown model name")

sequence_file = 'File input directory'
feature_dir = ''
sequence_dict = read_sequence(sequence_file)

counter = 0
for protein_id in sequence_dict:
  seq = sequence_dict[protein_id]
  sample = list(seq)
  embedding = embed_dataset(sample, shift_left, shift_right)
  print(np.array(embedding).shape)
  feature_file = protein_id + ".npy"
  np.save(feature_file, embedding)
  print(f'######## {protein_id} down ########')
  counter += 1
print(f'There are {counter} protein sequences in total!')


import pdb
import torch

import numpy as np
cor = np.load('sent/test_cor.npy')
dessyn = torch.from_numpy(np.load('new_output_sent/test_des.npy'))
#dessmi = torch.from_numpy(np.load('output0/test_des8065.npy'))
syn = torch.from_numpy(np.load('new_output_sent/test_smi.npy'))
#smi = torch.from_numpy(np.load('output0/test_smi8065.npy'))

# SMILES -> Language
print("dessyn shape: ", dessyn.shape)
print("syn shape: ", syn.shape)
print("cor shape: ", cor.shape)
print("cor: ", cor)

language_num = dessyn.shape[0]  # 12619
smiles_num = syn.shape[0]  # 3000

language_idx_to_smiles_idx = []
for i in range(language_num):
    for j in range(smiles_num):
        if cor[j] <= i and i < cor[j+1]:
            language_idx_to_smiles_idx.append(j)
            break
# print("language_idx_to_smiles_idx: ", language_idx_to_smiles_idx)  # 12619

print("SMILES -> Language")

score_syn = torch.zeros(smiles_num,smiles_num)
for i in range(smiles_num):
    score = torch.cosine_similarity(syn[i], dessyn, dim=-1)
    for j in range(smiles_num):
        score_syn[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])

rec_syn = []

for i in range(smiles_num):
    a,idx = torch.sort(score_syn[:,i])
    for j in range(smiles_num):
        if idx[-1-j]==i:
            rec_syn.append(j)
            break

print("Size of rec_syn: ", len(rec_syn))
print(sum( (np.array(rec_syn)<1).astype(int) ) / float(len(rec_syn)))
print(sum( (np.array(rec_syn)<5).astype(int) ) / float(len(rec_syn)))
print(sum( (np.array(rec_syn)<10).astype(int) ) / float(len(rec_syn)))
print(sum( (np.array(rec_syn)<20).astype(int) ) / float(len(rec_syn)))


print("Language -> SMILES")

score_syn = torch.zeros(language_num, smiles_num)
for i in range(language_num):
    score = torch.cosine_similarity(dessyn[i], syn, dim=-1)  # shape: 3000
    score_syn[i] = score

rec_syn = []

for i in range(language_num):
    a, idx = torch.sort(score_syn[i,:])
    for j in range(smiles_num):
        if idx[-1-j]==language_idx_to_smiles_idx[i]:
            rec_syn.append(j)
            break

print("Size of rec_syn: ", len(rec_syn))
print(sum((np.array(rec_syn)<1).astype(int) ) / float(len(rec_syn)))
print(sum((np.array(rec_syn)<5).astype(int) ) / float(len(rec_syn)))
print(sum((np.array(rec_syn)<10).astype(int) ) / float(len(rec_syn)))
print(sum((np.array(rec_syn)<20).astype(int) ) / float(len(rec_syn)))
import numpy as np
import pandas as pd
import h5py
from Bio import SeqIO
import os
import pickle


# import the train_set dataset
df = pd.read_csv('data/train/train_set.tsv', sep = '\t') ## insert your path to the file
# split it into the sub-ontologies
mf_df = df[df['aspect'] == 'molecular_function']
cc_df = df[df['aspect'] == 'cellular_component']
bp_df = df[df['aspect'] == 'biological_process']

# not necessary to filter as the professor already filtered the subontologies
# and proteins with sequence length of 2000+ also have been removed
count_mf = mf_df['GO_term'].value_counts() 
mf_keep = count_mf[count_mf >= 50].index.tolist()
y_mf = mf_df[mf_df['GO_term'].isin(mf_keep)]

count_cc = cc_df['GO_term'].value_counts() 
cc_keep = count_cc[count_cc >= 50].index.tolist()
y_cc = cc_df[cc_df['GO_term'].isin(cc_keep)]

count_bp = bp_df['GO_term'].value_counts() 
bp_keep = count_bp[count_bp >= 250].index.tolist()
y_bp = bp_df[bp_df['GO_term'].isin(bp_keep)]

# importing train_ids
with open('data/train/train_ids.txt') as f: ##insert path to the file
    train_ids = [line.strip() for line in f]

# map of domains and proteins
id_map = {pid: i for i, pid in enumerate(train_ids)}  ## this is quicker than the original code
#in this dictionary we have a set of all the proteins and their positions

# importing train_protein2ipr.dat, as protein_to_ipr
column_names = ['Protein_ID', 'InterPro_ID', 'Domain_Name', 'Source_DB', 'Start', 'End']

protein_to_ipr = pd.read_csv("data/train/train_protein2ipr.dat",  ## insert your path to the file
                            sep='\t', names = column_names)

# since there are thousands of domains we could keep the 1000 most represented
# (about the 70% of the total proteins are in these) so our pc doesn't explode during training

domain_count = protein_to_ipr["InterPro_ID"].value_counts()
top_1000 = domain_count.iloc[:1000].index.tolist()

# filter only rows with selected domains
protein_to_ipr_filtered = protein_to_ipr[protein_to_ipr["InterPro_ID"].isin(top_1000)]


#################### might present a problem for MF prediction since
#################### some functions are rare and require specific domains

# map domain → column index
ipr_map = {ipr: i for i, ipr in enumerate(top_1000)} #this is quicker

# Based on this code - 1000 top domains are most representative, with reasonable dimensionality reduction
# protein_to_ipr: full mapping df with Protein_ID, InterPro_ID
# domain_counts = protein_to_ipr["InterPro_ID"].value_counts()

# def coverage_at_k(k):
   # topk = set(domain_counts.head(k).index)
   # covered = protein_to_ipr[protein_to_ipr["InterPro_ID"].isin(topk)]["Protein_ID"].nunique()
   # total = protein_to_ipr["Protein_ID"].nunique()
   # return covered / total

# for k in [200, 500, 1000, 2000, 5000]:
   # print(k, round(coverage_at_k(k)*100, 2), "% proteins covered")

mhem = np.zeros((len(train_ids), len(top_1000)), dtype=np.int8)

for row in protein_to_ipr_filtered.itertuples():
    pid = row.Protein_ID
    ipr = row.InterPro_ID
    if pid in id_map:
        row_i = id_map[pid]
        col_i = ipr_map[ipr]
        mhem[row_i, col_i] = 1

def get_ordered_embedding_matrix(h5_path, ordered_ids):
    with h5py.File(h5_path, "r") as f:
        # detect embedding dimension
        sample_pid = ordered_ids[0]
        sample = f[sample_pid][()]
        emb_dim = sample.shape[-1]

        X = np.zeros((len(ordered_ids), emb_dim), dtype=np.float32)

        for i, pid in enumerate(ordered_ids):
            if pid in f:
                vec = f[pid][()]
                # mean pool if residue-level embeddings
                if vec.ndim == 2:
                    vec = vec.mean(axis=0)
                X[i] = vec
        return X

emb_matrix = get_ordered_embedding_matrix(
    "data/train/train_embeddings.h5",
    train_ids
)

X_train = np.concatenate((mhem.astype(np.float32), emb_matrix), axis=1)

def mapping_fun(go_list):
    return {go_term: i for i, go_term in enumerate(go_list)}

go_mf_map = mapping_fun(mf_keep)
go_cc_map = mapping_fun(cc_keep)
go_bp_map = mapping_fun(bp_keep)

Y_mf_train = np.zeros((len(train_ids), len(mf_keep)), dtype=np.int8)
for row in y_mf.itertuples():
    pid = row.Protein_ID
    go = row.GO_term
    if pid in id_map and go in go_mf_map:
        row_i = id_map[pid]
        col_j = go_mf_map[go]
        Y_mf_train[row_i, col_j] = 1

Y_cc_train = np.zeros((len(train_ids), len(cc_keep)), dtype=np.int8)
for row in y_cc.itertuples():
    pid = row.Protein_ID
    go = row.GO_term
    if pid in id_map and go in go_cc_map:
        row_i = id_map[pid]
        col_j = go_cc_map[go]
        Y_cc_train[row_i, col_j] = 1

Y_bp_train = np.zeros((len(train_ids), len(bp_keep)), dtype=np.int8)
for row in y_bp.itertuples():
    pid = row.Protein_ID
    go = row.GO_term
    if pid in id_map and go in go_bp_map:
        row_i = id_map[pid]
        col_j = go_bp_map[go]
        Y_bp_train[row_i, col_j] = 1

output_dir = "final_data"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "Y_mf_train.npy"), Y_mf_train)
np.save(os.path.join(output_dir, "Y_cc_train.npy"), Y_cc_train)
np.save(os.path.join(output_dir, "Y_bp_train.npy"), Y_bp_train)

with open("data/test/test_ids.txt") as f:
    test_ids = [line.strip() for line in f]

column_names = ['Protein_ID', 'InterPro_ID', 'Domain_Name', 'Source_DB', 'Start', 'End']

test_ipr = pd.read_csv(
    "data/test/test_protein2ipr.dat",
    sep="\t",
    names=column_names
)

test_ipr_filtered = test_ipr[test_ipr["InterPro_ID"].isin(top_1000)]

# Create map for test IDs
test_id_map = {pid: i for i, pid in enumerate(test_ids)}

# Create test multi-hot encoding matrix
mhem_test = np.zeros((len(test_ids), len(top_1000)), dtype=np.int8)

for row in test_ipr_filtered.itertuples():
    pid = row.Protein_ID
    ipr = row.InterPro_ID
    if pid in test_id_map and ipr in ipr_map:
        mhem_test[test_id_map[pid], ipr_map[ipr]] = 1

emb_test = get_ordered_embedding_matrix(
    "data/test/test_embeddings.h5",
    test_ids
)

X_test = np.concatenate((mhem_test, emb_test), axis=1)

np.save("data/test/X_test.npy", X_test)

SAVE_PATH = "metadata"
os.makedirs(SAVE_PATH, exist_ok=True)

# 1. Save GO terms for CC
np.save(f"{SAVE_PATH}/cc_keep.npy", np.array(cc_keep, dtype=object))

# 2. Save the GO mapping dict
with open(f"{SAVE_PATH}/go_cc_map.pkl", "wb") as f:
    pickle.dump(go_cc_map, f)

# 3. Save training protein ID list
np.save(f"{SAVE_PATH}/train_ids.npy", np.array(train_ids, dtype=object))

# 4. Save id_map (protein → row index in training matrices)
with open(f"{SAVE_PATH}/id_map.pkl", "wb") as f:
    pickle.dump(id_map, f)

# 5. Save top 1000 domains
np.save(f"{SAVE_PATH}/top_1000.npy", np.array(top_1000, dtype=object))

# 6. Save InterPro domain mapping dict
with open(f"{SAVE_PATH}/ipr_map.pkl", "wb") as f:
    pickle.dump(ipr_map, f)



os.makedirs(SAVE_PATH, exist_ok=True)

# 1. Save GO terms for MF
np.save(f"{SAVE_PATH}/mf_keep.npy", np.array(mf_keep, dtype=object))

# 2. Save the GO mapping dict
with open(f"{SAVE_PATH}/go_mf_map.pkl", "wb") as f:
    pickle.dump(go_mf_map, f)

# 3. Save training protein ID list
np.save(f"{SAVE_PATH}/train_ids.npy", np.array(train_ids, dtype=object))

# 4. Save id_map (protein → row index in training matrices)
with open(f"{SAVE_PATH}/id_map.pkl", "wb") as f:
    pickle.dump(id_map, f)

# 5. Save top 1000 domains
np.save(f"{SAVE_PATH}/top_1000.npy", np.array(top_1000, dtype=object))

# 6. Save InterPro domain mapping dict
with open(f"{SAVE_PATH}/ipr_map.pkl", "wb") as f:
    pickle.dump(ipr_map, f)




# 1. Save GO terms for BP
np.save(f"{SAVE_PATH}/bp_keep.npy", np.array(bp_keep, dtype=object))

# 2. Save the GO mapping dict
with open(f"{SAVE_PATH}/go_bp_map.pkl", "wb") as f:
    pickle.dump(go_bp_map, f)

# 3. Save training protein ID list
np.save(f"{SAVE_PATH}/train_ids.npy", np.array(train_ids, dtype=object))

# 4. Save id_map (protein → row index in training matrices)
with open(f"{SAVE_PATH}/id_map.pkl", "wb") as f:
    pickle.dump(id_map, f)

# 5. Save top 1000 domains
np.save(f"{SAVE_PATH}/top_1000.npy", np.array(top_1000, dtype=object))

# 6. Save InterPro domain mapping dict
with open(f"{SAVE_PATH}/ipr_map.pkl", "wb") as f:
    pickle.dump(ipr_map, f)

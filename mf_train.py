import os
import numpy as np
import pandas as pd
import h5py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score

import pickle

# setting a seed

seed = 62
np.random.seed(seed)
tf.random.set_seed(seed)

#### training data loading ####

X_train = np.load('final_data/X_train.npy')  

interpro_dim = 1000
embeded_dim = X_train.shape[1] - interpro_dim
assert embeded_dim > 0, "Embedding dimension inferred <= 0; check interpro_dim."

X_train_domains = X_train[:, :interpro_dim].astype(np.float32)  # keep as 0/1 float
X_train_embed   = X_train[:, interpro_dim:].astype(np.float32)


Y_mf_train = np.load('final_data/Y_mf_train.npy')


# MF label filtering

use_mf_mask = True # can be turned of, not recommended 
if use_mf_mask:
    mask = (Y_mf_train.sum(axis=1) > 0)
    X_train_domains = X_train_domains[mask]
    X_train_embed = X_train_embed[mask]
    Y_mf_train = Y_mf_train[mask]

# Normalization of embeddings

scaler = StandardScaler()
X_train_embed_sc = scaler.fit_transform(X_train_embed)

scaler_path = 'metadata/scaler.pkl'
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

X_scaled = np.concatenate([X_train_domains, X_train_embed_sc], axis=1) # reassembling matrix

###### functions for building models #########
# creation of a function to build  the Model 
# it's needed to perform a kfold 

def build_model(y):
    model = Sequential()

    model.add(Dense(512, input_shape = (X_scaled.shape[1],), activation = 'relu'),
            Dropout(0.5))

    model.add(Dense(256, activation = 'relu'),
            Dropout(0.3))

    model.add(Dense(y.shape[1], activation = 'sigmoid'))

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return model






############################################# Molecular function model training #######################################################






# Data split

x_develop, x_test, y_develop, y_test = train_test_split(X_scaled, Y_mf_train, test_size=0.2, random_state=seed)


### model validation: this is the code used for the 10 kfold validation


'''
k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)
all_fmax = []

print(f"\nCross-Validation (10 folds) Target: {Y_mf_train.shape[1]} classes.")

for fold, (train_idx, val_idx) in enumerate(k_fold.split(x_develop)):
    X_train, X_val = x_develop[train_idx], x_develop[val_idx]
    y_train, y_val = y_develop[train_idx], y_develop[val_idx]

    # model training inside the loop
    model = build_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_train,
               validation_data=(X_val, y_val), callbacks=[early_stop],
              epochs=50, batch_size=32, verbose=0)

    y_probs = model.predict(X_val, verbose=0)
    
    precision, recall, thresholds = precision_recall_curve(y_val.ravel(), y_probs.ravel())
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    fmax = np.max(f1_scores)
    
    scores = model.evaluate(X_val, y_val, verbose=0)
    
    all_fmax.append(fmax)
    
    print(f"Fold {fold+1} | Max F-score: {fmax:.4f}")

print("\nFinal results ")
print(f"Average F-max score:     {np.mean(all_fmax):.4f} (+/- {np.std(all_fmax):.4f})")
'''


final_model = build_model(Y_mf_train)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

final_model.fit(x_develop, y_develop, validation_data=(x_test, y_test), 
                epochs=50, batch_size=32, verbose=1, callbacks = early_stop)


model_save_path = 'models/MF_model.h5'
final_model.save(model_save_path)

##### creation of a csv file for the predictions

# test data loading

X_test = np.load('final_data/X_test.npy')

# metadata loading

mf_keep = np.load(f"metadata/mf_keep.npy", allow_pickle=True)
with open(f"metadata/go_mf_map.pkl", "rb") as f:
    go_mf_map = pickle.load(f)

# inverse mapping column index to GO term
inv_go_mf_map = {v: k for k, v in go_mf_map.items()}

# test ids loading

with open('final_data/test_ids.txt', "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]


# Test data normalization

# loading scaler
with open("metadata/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Separating InterPro and ProtT5 features exactly as in training
interpro_dim = 1000
Xtest_domains = X_test[:, :interpro_dim].astype(np.float32)
Xtest_embed   = X_test[:, interpro_dim:].astype(np.float32)

# Scaling
Xtest_embed_scaled = scaler.transform(Xtest_embed)

# Recombining
X_test_scaled = np.concatenate(
    [Xtest_domains, Xtest_embed_scaled],
    axis=1
)

test_probs = final_model.predict(X_test_scaled, batch_size=512, verbose=1)


max_protein_terms = 500   # cap of terms per sub-ontology per protein
THR = 0.5

rows = []
for i, pid in enumerate(test_ids):
    probs = test_probs[i]

    # candidate indices above threshold
    idx = np.where(probs >= THR)[0]

    # if none are above the threshold, output the top few
    if idx.size == 0:
        idx = np.argsort(-probs)[:10]

    # sort by probability
    idx = idx[np.argsort(-probs[idx])][:max_protein_terms]

    for j in idx:
        go = inv_go_mf_map[int(j)]
        score = float(probs[j])

        # enforce (0, 1.000] and avoid 0
        if score <= 0.0:
            continue
        if score > 1.0:
            score = 1.0

        # format to 3 significant figures
        score_str = f"{score:.3g}"
        rows.append((pid, go, score_str))

pred_out_csv = 'results/MF_predictions.csv'


pred_df = pd.DataFrame(rows, columns=["Protein_ID", "GO_term", "score"])
pred_df.to_csv(pred_out_csv, index=False)
print("Saved:", pred_out_csv, "rows:", len(pred_df))

submission_txt = ('metadata/mf_submission.txt')
with open(submission_txt, "w") as f:
    for pid, go, score_str in rows:
        f.write(f"{pid}\t{go}\t{score_str}\n")
        
print("Saved:", submission_txt)
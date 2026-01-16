
## This script is used to implement all the code written so far in a single place so that the main.py script can use it 
## to start the software


## LIBRARIES ##
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import logging
import os


class CAFA_predictor:
    def __init__(self, model, scaler, go_map, cache):
        self.model = tf.keras.models.load_model(model)

        with open(scaler, 'rb') as f:
                self.scaler = pickle.load(f)

        with open(go_map, 'rb') as f:
                self.go_map = pickle.load(f)

        self.cache_df = None
        if cache:
            try:
                df = pd.read_csv(cache)
                df['Protein_ID'] = df['Protein_ID'].astype(str)
                self.cache_df = df.set_index('Protein_ID') 
            except:
                pass

        self.inv_go_map = {v: k for k, v in self.go_map.items()}

    def preprocess(self, X_input):
        interpro_dim = 1000
        X_domains = X_input[:, :interpro_dim].astype(np.float32)
        X_embed   = X_input[:, interpro_dim:].astype(np.float32)   
        X_embed_sc = self.scaler.transform(X_embed)

        return np.concatenate([X_domains, X_embed_sc], axis=1)
    
    def predict(self, X_preproc, prot_ids, threshold=0.5):
        probs = self.model.predict(X_preproc, verbose=0)
        results = []

        for i, pid in enumerate(prot_ids):
            pid_str = str(pid)
            found_in_cache = False
            
            if self.cache_df is not None and pid_str in self.cache_df.index:
                subset = self.cache_df.loc[[pid_str]]
                
                for row in subset.itertuples():
                    
                    try:
                        results.append((pid_str, row.GO_term, row.score))
                        found_in_cache = True
                    except AttributeError:
                        found_in_cache = False 

            if not found_in_cache:
                scores = probs[i]
                idx = np.where(scores >= 0.5)[0]

                if len(idx) == 0:
                    idx = np.argsort(-scores)[:5]

                for j in idx:
                    if j in self.inv_go_map: 
                        term = self.inv_go_map[j]
                        score = float(scores[j])
                        results.append((pid_str, term, score))

        return results
        


        
        
            
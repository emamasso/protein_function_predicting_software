
## This module implements the inference logic used by main.py.
## It handles feature preprocessing, ontology-specific prediction,
## top-K ranking, optional caching, and CAFA-compliant post-processing.

    ## LIBRARIES ##
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd


class CAFA_predictor:
        def __init__(self, model, scaler, go_map, cache=None):
            self.model = tf.keras.models.load_model(model)

            with open(scaler, 'rb') as f:
                    self.scaler = pickle.load(f)

            with open(go_map, 'rb') as f:
                    self.go_map = pickle.load(f)

            self.inv_go_map = {v: k for k, v in self.go_map.items()}

            self.cache_df = None
            if cache:
                try:
                    df = pd.read_csv(cache)
                    df['Protein_ID'] = df['Protein_ID'].astype(str)
                    self.cache_df = df.set_index('Protein_ID') 
                except Exception as e:
                    self.cache_df = None

        def preprocess(self, X_input):
            interpro_dim = 1000
            X_domains = X_input[:, :interpro_dim].astype(np.float32)
            X_embed   = X_input[:, interpro_dim:].astype(np.float32)   
            X_embed_sc = self.scaler.transform(X_embed)

            return np.concatenate([X_domains, X_embed_sc], axis=1)
        
        def predict(self, X_preproc, prot_ids, k=500):
            EPS = 1e-6
            probs = self.model.predict(X_preproc, verbose=0)
            results = []

            n_terms = probs.shape[1]
            k = min(k, n_terms)

            for i, pid in enumerate(prot_ids):
                pid_str = str(pid)

            # ---- Use cache if available ----
                if self.cache_df is not None and pid_str in self.cache_df.index:
                    subset = self.cache_df.loc[[pid_str]]
                    subset = subset.sort_values("score", ascending=False).head(k)

                    for row in subset.itertuples():
                        score = max(float(row.score), EPS)
                        results.append((pid_str, row.GO_term, score))
                    continue

            # ---- Model-based top-K prediction ----
                scores = probs[i]
                top_idx = np.argsort(scores)[::-1][:k]

                for j in top_idx:
                    if j in self.inv_go_map:
                        go_term = self.inv_go_map[j]
                        score = max(float(scores[j]), EPS)
                        results.append((pid_str, go_term, score))

        # Safety check (CAFA compliance)
            for _, _, score in results:
                assert score > 0.0

            return results
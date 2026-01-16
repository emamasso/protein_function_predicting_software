

                                    ## LIBRARIES ##

import argparse
import logging
import json
import sys
import os
import numpy as np
import pandas as pd
from Bio import SeqIO 
from src.predictor import CAFA_predictor

##############################################################################################

### LOGGING SETUP ###

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("execution.log", mode='w')])

logger = logging.getLogger(__name__)

def main():
    # ARGPARSE
    parser = argparse.ArgumentParser(description="Protein Function Predictor (CAFA).")
    parser.add_argument("-i", "--input", required=True, help="Input file path (.npy or .fasta)")
    parser.add_argument("-c", "--config", default="config.json", help="Configuration file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    args = parser.parse_args()

    #  LOAD CONFIG 
    if not os.path.exists(args.config):
        logger.critical("Config file not found.")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    # INPUT HANDLING 
    protein_ids = []
    X_input = None
    
    # Check estentions
    if args.input.endswith(('.fasta', '.fa')):
        logger.info("Reading FASTA with BioPython...")
        try:
            for record in SeqIO.parse(args.input, "fasta"):
                protein_ids.append(record.id)
            logger.warning("FASTA support is experimental. Expecting .npy features for inference.")
            sys.exit(1) #
        except Exception as e:
            logger.error(f"BioPython error: {e}")
            sys.exit(1)
            
    elif args.input.endswith('.npy'):
        try:
            X_input = np.load(args.input)
            
            protein_ids = [f"Prot_{i}" for i in range(X_input.shape[0])]
            logger.info(f"Loaded input matrix: {X_input.shape}")
        except Exception as e:
            logger.error(f"Numpy load error: {e}")
            sys.exit(1)
    else:
        logger.error("Unsupported format. Use .npy")
        sys.exit(1)

    # PREDICTION LOOP
    ontologies = ['MF', 'BP', 'CC']
    all_results = []
    threshold = config['settings'].get('threshold', 0.5)
    
    
    scaler_path = config['metadata']['scaler_path']

    for onto in ontologies:
        logger.info(f"Processing Ontology: {onto}...")
        try:
            model_path = config['models'][onto]
            map_path   = config['metadata'][f'{onto}_map']
            cache_path = config['cache'].get(onto) 
            
            
            predictor = CAFA_predictor(model_path, scaler_path, map_path, cache_path)
            
            
            X_proc = predictor.preprocess(X_input)
            
            
            preds = predictor.predict(X_proc, protein_ids, threshold)
            
            
            for p in preds:
                all_results.append((p[0], onto, p[1], p[2]))
                
        except Exception as e:
            logger.error(f"Failed to process {onto}: {e}")

    # SAVE OUTPUT 
    if all_results:
        
        if args.output:
            out_path = args.output
        else:
            out_dir = config['settings']['default_output_dir']
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "predictions.csv")
            
        df = pd.DataFrame(all_results, columns=["Protein_ID", "Ontology", "GO_Term", "Score"])
        df.to_csv(out_path, index=False)
        logger.info(f"SUCCESS! Results saved to: {out_path}")
    else:
        logger.warning("No predictions found above threshold.")

if __name__ == "__main__":
    main()
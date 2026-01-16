# Authors
- Emanuele Massoglia
- Jakov Tushevski
- 

# DOCUMENTATION 

This document has the sole purpose of explaining the creation of a software for using the results of a deep learning pipeline for predicting protein functions.


### Software

The inference pipeline is designed to ensure both high performance and robustness. The core logic is encapsulated within the CAFA_predictor class (located in src/predictor.py), which implements a Hybrid Caching Strategy:

Cache Lookup (O(1)): Upon receiving a protein ID, the software first checks a pre-loaded cache (CSV format). If the protein has been previously annotated and validated, the result is retrieved instantly. This ensures maximum accuracy for known entities and drastically reduces computational time.

Deep Learning Inference: If the protein is not found in the cache, the system falls back to the trained MLP models (TensorFlow/Keras). The input features are scaled using the saved metadata, and predictions are generated based on the configured confidence threshold.

This architecture satisfies the requirement for a robust software solution capable of handling both known datasets and novel sequences.

#### Installation and requirements

The software requires Python 3.8 or higher. It relies on standard bioinformatics and machine learning libraries.
Dependencies: Ensure the following packages are installed (or install via pip install -r requirements.txt):

    -tensorflow: For loading and executing the neural networks.
    -pandas & numpy: For data manipulation and matrix operations.
    -biopython: For parsing and validating biological file formats (FASTA).
    -scikit-learn: For feature scaling and cross-validation utilities.

#### Configuration

To ensure modularity and reproducibility, no hard-coded paths are used within the source code. All operational parameters are defined in the external config.json file:

    -models: Paths to the trained .h5 files for MF, BP, and CC ontologies.
    -metadata: Paths to the scalers and Gene Ontology mapping files (.pkl).
    -cache: Paths to the pre-calculated result files (optional).
    -settings: Global parameters, such as the threshold (default: 0.5), which determines the confidence level required to assign a specific GO term.

#### Syntax

The software provides a Command Line Interface (CLI) implemented via the argparse module, making it easy to integrate into larger bioinformatics pipelines.

#### Running Predictions (main.py)
To generate predictions for a new dataset, execute the main.py script. The software automatically handles data loading, preprocessing, inference, and result formatting. FO executing you need to type the following commands in your terminal after having set the directory in which the software is saved as the current directory:

python main.py -i <INPUT_FILE> [-o <OUTPUT_FILE>] [-c <CONFIG_FILE>]

Here is an example of the syntax:

python main.py -i data/test/X_test.npy -o results/final_predictions.csv

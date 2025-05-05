import os

RANDOM_SEED = 0
CACHE_DIR = ".cache"
RES_DIR = "results"
DATA_DIR = "data"
FIGS_DIR = "figs"

# Create directories if they don't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)

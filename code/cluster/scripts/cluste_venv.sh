# Remove broken venv
rm -rf /raid/persistent_scratch/$USER/venvs/faviscore_env

# Load the Python module first
module load python/3.10.14

# Create venv with pip
python3 -m venv /raid/persistent_scratch/$USER/venvs/faviscore_env

# Activate it
source /raid/persistent_scratch/$USER/venvs/faviscore_env/bin/activate

# Install packages
pip install datasets ollama numpy pandas tqdm requests
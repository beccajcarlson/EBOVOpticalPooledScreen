# EBOV Analysis 
This repository contains the code, tools, and documentation for training, evaluating, and visualizing autoencoder models for use in cellular image datasets.

## Training an Unsupervised Autoencoder
To train the unsupervised autoencoder, first clone the repository, then enter this directory, and run the `genome_wide.py` script. Note that this script will train a model for a specified number of epochs then save the model, losses, and reproducing shell command in a directory named `models`:
```shell
# Clone repository
git clone https://github.com/beccajcarlson/EBOVOpticalPooledScreen.git EBOVOpticalPooledScreen
# Enter directory
cd EBOVOpticalPooledScreen/DeepLearning/
# Install dependencies (preferably in new virtual environment)
pip install -r requirements.txt
# (Optional) View flag options on script
python modeling/genome_wide.py -h
# Train Unsupervised Autoencoder Sample
python modeling/genome_wide.py -n
``` 

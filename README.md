# Code for Ebola Virus Optical Pooled Screen

Code for "Single-cell image-based genetic screens systematically
identify regulators of Ebola virus subcellular infection dynamics".
Optical pooled screen analysis code adapted from Feldman, D., Funk, L. et al Nature Protocols (2022).

## Training an Unsupervised Autoencoder
To train the unsupervised autoencoder, first clone the repository, then enter this directory, and run the genome_wide.py script. Note that this script will train a model for a specified number of epochs then save the model, losses, and reproducing shell command in a directory named models:

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

## OS Requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:
- macOS: Somona (14.4)
- Linux: Debian 5.10.197-1 
Installation should take no more than a minute or two on similar systems.

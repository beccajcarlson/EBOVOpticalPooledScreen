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

Runtime will vary depending on the number of epochs but should take no more than several hours on a typical desktop system.

## Pretrained Models
Pre-trained models are accessible from [Google Drive](https://drive.google.com/drive/folders/1YBSppdanca77hTY7eeLmPjKCA1iiofs-?usp=share_link). The drive hosts three models: an unsupervised model, a semi-supervised model, and a semi-supervised + tuned model (as `unsupervised_final_weights.pth`, `semi_supervised_final_weights.pth`, and `semi_supervised_retuned_final_weights.pth`, respectively).

Once downloaded, the models can be loaded as shown in the snippet below. 

**Note:** The imported module `model` is sourced from this file within the repository: [`model.py`](./modeling/model.py).

```python
import torch
from model import ConvAutoencoder, ConvAutoencoderWithHead


# Unsupervised
unsupervised_model = ConvAutoencoder()
unsupervised_model.load_state_dict(torch.load("./unsupervised_final_weights.pth"))

# Semi-Supervised
semi_model = ConvAutoencoderWithHead()
semi_model.load_state_dict(torch.load("./semi_supervised_final_weights.pth"))

# Semi-Supervised + Tuned
tuned_model = ConvAutoencoderWithHead()
tuned_model.load_state_dict(torch.load("./semi_supervised_retuned_final_weights.pth"))
```

## Code for paper "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality". https://arxiv.org/abs/1801.02613

## Update: added BatchNormalization to after Conv and ReLU. 17 Sept. 2018.

### 1. Pre-train DNN models:
python train_model.py -d mnist -e 50 -b 128

### 2. Craft adversarial examples:
python craft_adv_samples.py -d cifar -a cw-l2 -b 100
### 3.Extract detection characteristics:
python extract_characteristics.py -d cifar -a cw-l2 -r lid -k 20 -b 100

### 4. Train simple detectors:
python detect_adv_examples.py -d cifar -a fgsm -t cw-l2 -r lid

#### Requirements:
numpy, scipy, tqdm, sklearn, matplotlib, tensorflow >= 1.0, Keras >= 2.0, cleverhans >= 1.0.0 (may need extra change to pass in keras learning rate)

#### Kernal Density and Bayesian Uncertainty are from https://github.com/rfeinman/detecting-adversarial-samples ("Detecting Adversarial Samples from Artifacts" (Feinman et al. 2017))

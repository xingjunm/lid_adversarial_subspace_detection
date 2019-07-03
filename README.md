## Code for paper "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality". ICLR 2018, https://arxiv.org/abs/1801.02613

## Update: added BatchNormalization to after Conv and ReLU. 17 Sept. 2018.

### 1. Pre-train DNN models:
python train_model.py -d mnist -e 50 -b 128

### 2. Craft adversarial examples:
python craft_adv_samples.py -d cifar -a cw-l2 -b 100
### 3.Extract detection characteristics:
python extract_characteristics.py -d cifar -a cw-l2 -r lid -k 20 -b 100

### 4. Train simple detectors:
python detect_adv_examples.py -d cifar -a fgsm -t cw-l2 -r lid

#### Dependencies:
python 3.5, tqdm, tensorflow = 1.8, Keras >= 2.0, cleverhans >= 1.0.0 (may need extra change to pass in keras learning rate)

#### Kernal Density and Bayesian Uncertainty are from https://github.com/rfeinman/detecting-adversarial-samples ("Detecting Adversarial Samples from Artifacts" (Feinman et al. 2017))

---------------------------
If you came across the error:

tensorflow.python.framework.errors_impl.InvalidArgumentError: input_1:0 is both fed and fetched.


Solution: in function get_layer_wise_activations() (util.py), do the following change:
acts = [layer.output for layer in model.layers[1:]] # let the layer index start from 1.

Reason: this possibly cause by the input layer is defined as a sepearte layer, with both input and output is X.

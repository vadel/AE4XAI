Code corresponding to the paper "When and How to Fool Explainable Models (and Humans) with Adversarial Examples" by Jon Vadillo, Roberto Santana and Jose A. Lozano.

# Instructions
To reproduce the adversarial examples:
### XRAY (COVID-Net)
- Download the [COVIDNet-CXR Small pretrained model](https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md) in *xray/COVID-Net/pretrained/COVIDNet-CXR_Small/*
- Run xray/XRAY_AdversarialExamples.ipynb
### ILSVRC
- Download the (ILSVRC2012) [Imagenet training set](https://www.image-net.org/download.php) in *ilsvrc/datasets/*
- Download the (blurred) [Imagenet validation set](https://www.image-net.org/download.php) in *ilsvrc/datasets/*
- Run ilsvrc/ILSVRC_AdversarialExamples.ipynb

# GPUTreeShap

GPUTreeShap is a cuda implementation of the TreeShap algorithm by Lundberg et al. [1] for Nvidia GPUs. It is a header only module designed to be included in decision tree libraries as a fast backend for model interpretability using SHAP values.

## Performance
TODO:
V100 performance comparison

GPUTreeShap uses very little working GPU memory, only allocating space proportional to the model size. An application is far more likely to be constrained by the size of the dataset.

## Usage
See examples for sample integration into a C++ decision tree project. GPUTreeShap accepts a decision tree ensemble in the form of a list of unique paths through all branches of the tree, as well as an interface to a dataset allocated on the GPU, and returns feature contributions for each row in the dataset.

## References
[1] Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. "Consistent individualized feature attribution for tree ensembles." arXiv preprint arXiv:1802.03888 (2018).
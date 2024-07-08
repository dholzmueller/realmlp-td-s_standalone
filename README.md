# RealMLP-TD-S standalone implementation

This repository contains a small standalone implementation of RealMLP-TD-S, 
a neural network for tabular datasets, consisting of
- preprocessing code (91 lines of code) in `preprocessing.py`, which includes 
  - one-hot encoding with custom missing/unknown value encoding 
and encoding binary categories to 1/-1
  - robust scaling and smooth clipping
- the MLP implementation (212 lines of code) in `mlp.py`.

The implementation is standalone in the sense that it only uses 
`numpy`, `pandas`, `sklearn`, and `torch`. 
This code is not available on `pip` since it can just be copied.

RealMLP-TD-S is also available with more functionality 
in [PyTabKit](github.com/dholzmueller/pytabkit).

The file `check_mlp.py` 
checks that this implementation matches the one in PyTabKit
and requires to install PyTabKit to run. 

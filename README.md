# EI_MS_ML

This code was submitted for an eIDR through the Feynman Center and was assigned as C22078

EI_MS_ML is/will be a collection of tools for the prediction of chemical properties from electron ionization mass spectra. 

The package consists of three parts. The first takes EI-MS spectra in msp format (such as NIST17 mainlib) annotated with their InChiKeys, maps them to SMILES strings, colorizes them, and trains random forest models on each color present in a specified number of compounds (default: 1000).

The second takes the models constructed in part one and combines them using genetic algorithms to yield a blended model that can be used to improve assignment ambiguity. The utility function for this optimization seeks to maximize the number of compounds whose correct's assignment improved in rank.

The third step takes a separate EI-MS spectral dataset (such as NIST17 replib) and evaluates how well the blended models improve the assignment ambiguity of these compounds using the training dataset as the database that is being queried for assignment. 

Triad National Security, LLC (Triad) owns the copyright to EI_MS_ML, which it identifies as project number C22078.

© 2022. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.



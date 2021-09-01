# Surrogate Optimization Algorithms for Expensive Global Optimization Problems
This repository is an optimization toolbox for computationally expensive global optimization problems. This toolbox supports multiple surrogate optimization softwares including: Python Surrogate Optimization Toolbox ([pySOT](https://github.com/dme65/pySOT)), Gap Optimized Multi-objective Optimization using Response Surfaces ([GOMORS](https://github.com/drkupi/GOMORS_pySOT)), 𝜀-dominance Many-objective Surrogate-assisted Optimization(𝜀-MaSO),Global Optimization in Parallel with Surrogate ([GOPS](https://github.com/louisXW/GOPS)) and multi-fidelity RBF (radial basis function) surrogate-based optimization (MRSO). All the algorithms supports both continuous and integer variables. There is also a repository which is based on pySOT speciffically focused on mix-integer optimization and machine learning, please refer to [HORD](https://github.com/ilija139/HORD).<br>

## Installation
The easiest way to install pySOT2 is through pip in which case the following command should suffice:
```
pip install pySOT2
```
## Running pySOT2
This repository aims to provide fast implementation of optimization algorithms, in the meantime, it also provides many parameters so users can easily adjust the algorithms accordingly. To run this repository, users need first define the optimization problems and choose an optimizer which will return the best solutions and its corresponding function value.The "examples" folder provide sample codes for all the software included and the "pySOT2\Optimize" folder provides the source code for all the optimizers.<br>
1. For single objective optimization problems, DYCORS(in pySOT) is recommended:
 ```
from pySOT2.Optimize import Optim
Optim.optimize(problem)
```
2. For multi-objective optimization problems, GOMORS is recommended:
```
from pySOT2.Optimize import MOoptim
MOoptim.MOoptimize(problem)
```
3. For many-objective(more than 3) optimization problems, 𝜀-MaSO is recommended:
```
from pySOT2.Optimize import epsMOoptim
epsMOoptim.epsMOoptimize(problem)
```
4. For parallel optimization on high dimensional problems, GOPS is recommended:
```
from pySOT2.Optimize import GOPSoptim
GOPSoptim.GOPSoptimize(problem)
```
5. For multi-modal optimization problems when multi-fidelity models are available, MRSO is recommended:
```
from pySOT2.Optimize import MFoptim
MFoptim.MFoptimize(problem)
```
## Citation
If you use pySOT, please cite the following paper: [David Eriksson, David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for surrogate optimization. arXiv preprint arXiv:1908.00420, 2019](https://arxiv.org/abs/1908.00420)<br>
If you use GOMORS, please cite the following paper: [Akhtar, T., Shoemaker, C.A. Multi objective optimization of computationally expensive multi-modal functions with RBF surrogates and multi-rule selection. J Glob Optim 64, 17–32 (2016).](https://doi.org/10.1007/s10898-015-0270-y)<br>
If you use 𝜀-MaSO, please cite the following paper:[Wang, W., Akhtar, T. & Shoemaker, C.A. Integrating 𝜀-dominance and RBF surrogate optimization for solving computationally expensive many-objective optimization problems. J Glob Optim (2021).](https://doi.org/10.1007/s10898-021-01019-w)<br>
If you use GOPS, please cite the following paper:[Xia, W., Shoemaker, C. GOPS: efficient RBF surrogate global optimization algorithm with high dimensions and many parallel processors including application to multimodal water quality PDE model calibration. Optim Eng (2020).](https://doi.org/10.1007/s11081-020-09556-1)<br>
If you use MRSO, please cite the following paper:[Yi, J., Shen, Y. & Shoemaker, C.A. A multi-fidelity RBF surrogate-based optimization framework for computationally expensive multi-modal problems with application to capacity planning of manufacturing systems. Struct Multidisc Optim 62, 1787–1807 (2020).](https://doi.org/10.1007/s00158-020-02575-7)<br>

# Surrogate Optimization Algorithms for Expensive Global Optimization Problems
This repository is a collection of optimization toolboxes for computationally expensive global optimization problems. This toolbox supports multiple surrogate optimization softwares including: Python Surrogate Optimization Toolbox ([pySOT](https://github.com/dme65/pySOT)), Gap Optimized Multi-objective Optimization using Response Surfaces ([GOMORS](https://github.com/drkupi/GOMORS_pySOT)), 𝜀-dominance Many-objective Surrogate-assisted Optimization(𝜀-MaSO),Global Optimization in Parallel with Surrogate ([GOPS](https://github.com/louisXW/GOPS)) and multi-fidelity RBF (radial basis function) surrogate-based optimization (MRSO). pySOT supports both continuous and integer variables; however, the other algorithms are focused on continuous variables in both their papers and codes. For mixed-integer optimization, there is a repository which is specifically focused on mixed-integer optimization and machine learning, please refer to [HORD](https://github.com/ilija139/HORD).<br>

The table below gives the key features and the reference link for each algorithm, for more detailed instruction, please refer to "Running pySOT2" and for full references, please refer to "Citation". The algorithm mentioned below all uses radial basis functions as surrogate models to perform optimization and they perform very well on expensive black-box obejctive functions.

| Feature | Algorithm | Reference |
| --- | --- | --- |
| serial/parallel single-objective optimization| DYCORS (pySOT)| https://arxiv.org/abs/1908.00420 |
| serial/parallel multi-objective optimization | GOMORS | https://doi.org/10.1007/s10898-015-0270-y |
| serial/parallel many-objective optimization | 𝜀-MaSO | https://doi.org/10.1007/s10898-021-01019-w |
| parallel single-objective optimization | GOPS | https://doi.org/10.1007/s11081-020-09556-1 |
| serial/parallel multi-fidelity optimization | MRSO | https://doi.org/10.1007/s00158-020-02575-7 |
| serial/parallel mixed-integer optimization| HORD| https://arxiv.org/pdf/1607.08316.pdf |

## Installation
The easiest way to install pySOT2 is through pip in which case the following command should suffice:
```
pip install pySOT2
```
## Running pySOT2
This repository aims to provide fast implementation of optimization algorithms, in the meantime, it also provides many parameters so users can easily adjust the algorithms accordingly. To run this repository, users need first define the optimization problems and choose an optimizer which will return the best solutions and its corresponding function value.The "examples" folder provide sample codes for all the software included and the "pySOT2\optimize" folder provides the source code for all the optimizers.<br>
1. For single objective optimization problems, DYCORS(in pySOT) is recommended:
 ```
from pySOT2.optimize import surrogate_optimization
surrogate_optimization.optimize(problem)
```
2. For multi-objective optimization problems, GOMORS is recommended:
```
from pySOT2.optimize import multi_objective
multi_objective.optimize(problem)
```
3. For many-objective(more than 3) optimization problems, 𝜀-MaSO is recommended:
```
from pySOT2.optimize import many_objective
many_objective.optimize(problem)
```
4. For parallel optimization on high dimensional problems, GOPS is recommended:
```
from pySOT2.optimize import parallel_surrogate
parallel_surrogate.optimize(problem)
```
5. For multi-modal optimization problems when multi-fidelity models are available, MRSO is recommended:
```
from pySOT2.optimize import multi_fidelity
multi_fidelity.optimize(problem)
```
## Citation
If you use pySOT, please cite the following paper: [David Eriksson, David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for surrogate optimization. arXiv preprint arXiv:1908.00420, 2019](https://arxiv.org/abs/1908.00420)<br>
If you use GOMORS, please cite the following paper: [Akhtar, T., Shoemaker, C.A. Multi objective optimization of computationally expensive multi-modal functions with RBF surrogates and multi-rule selection. J Glob Optim 64, 17–32 (2016).](https://doi.org/10.1007/s10898-015-0270-y)<br>
If you use 𝜀-MaSO, please cite the following paper:[Wang, W., Akhtar, T. & Shoemaker, C.A. Integrating 𝜀-dominance and RBF surrogate optimization for solving computationally expensive many-objective optimization problems. J Glob Optim (2021).](https://doi.org/10.1007/s10898-021-01019-w)<br>
If you use GOPS, please cite the following paper:[Xia, W., Shoemaker, C. GOPS: efficient RBF surrogate global optimization algorithm with high dimensions and many parallel processors including application to multimodal water quality PDE model calibration. Optim Eng (2020).](https://doi.org/10.1007/s11081-020-09556-1)<br>
If you use MRSO, please cite the following paper:[Yi, J., Shen, Y. & Shoemaker, C.A. A multi-fidelity RBF surrogate-based optimization framework for computationally expensive multi-modal problems with application to capacity planning of manufacturing systems. Struct Multidisc Optim 62, 1787–1807 (2020).](https://doi.org/10.1007/s00158-020-02575-7)<br>
The papers above are focused on continuous variables, for mixed-integer optimization please refer to HORD and cite the following paper:
[Ilievski, Ilija, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker. 2017. “Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates.” Pp. 822–29 in Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17. San Francisco, California, USA: AAAI Press.](https://arxiv.org/pdf/1607.08316.pdf)

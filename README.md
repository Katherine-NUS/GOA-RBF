# Global Optimization Algorithms with RBF Surrogates (GOA-RBF)
This repository is a collection of optimization toolboxes for computationally expensive global optimization problems. This toolbox supports multiple surrogate optimization softwares including: Python Surrogate Optimization Toolbox ([pySOT](https://github.com/dme65/pySOT)), Gap Optimized Multi-objective Optimization using Response Surfaces ([GOMORS](https://github.com/drkupi/GOMORS_pySOT)), 𝜀-dominance Many-objective Surrogate-assisted Optimization (𝜀-MaSO), Global Optimization in Parallel with Surrogates ([GOPS](https://github.com/louisXW/GOPS)) and multi-fidelity RBF (radial basis function) surrogate-based optimization (MRSO). All the algorithms support both continuous and integer variables. There is also a repository which is based on pySOT specifically focused on mixed-integer optimization and machine learning, please refer to [HORD](https://github.com/ilija139/HORD).<br>

The table below gives the key features and the reference link for each algorithm, for more detailed instructions, please refer to "Running GOArbf" and for full references, please refer to "Citation". The optimization algorithms mentioned below all use a radial basis function type of surrogate model and they perform very well on expensive black-box objective functions.

| Feature | Algorithm | GitHub Link | Authors | Reference Link |
| --- | --- | --- | --- | --- |
| optimization algorithm collection toolbox| pySOT | [pySOT](https://github.com/dme65/pySOT) | David Eriksson, David Bindel, Christine A. Shoemaker | https://arxiv.org/abs/1908.00420|
| serial/parallel single-objective optimization| DYCORS | [pySOT](https://github.com/dme65/pySOT) | Rommel G Regis and Christine A Shoemaker | https://doi.org/10.1080/0305215X.2012.687731 |
| serial/parallel multi-objective optimization | GOMORS | [GOMORS](https://github.com/drkupi/GOMORS_pySOT) | Akhtar, T., Shoemaker, C.A. |https://doi.org/10.1007/s10898-015-0270-y |
| serial/parallel many-objective optimization | 𝜀-MaSO | None | Wang, W., Akhtar, T. & Shoemaker, C.A. | https://doi.org/10.1007/s10898-021-01019-w |
| parallel single-objective optimization | GOPS | [GOPS](https://github.com/louisXW/GOPS) | Xia, W., Shoemaker, C. | https://doi.org/10.1007/s11081-020-09556-1 |
| serial/parallel multi-fidelity optimization | MRSO | None | Yi, J., Shen, Y. & Shoemaker, C.A. |https://doi.org/10.1007/s00158-020-02575-7 |
| serial/parallel mixed-integer optimization| HORD| [HORD](https://github.com/ilija139/HORD) | Ilievski, Ilija, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker. | https://arxiv.org/pdf/1607.08316.pdf |


## Installation
To install the software, you can choose to download the zip file from GitHub or use "python setup.py install" once you cloned the repository.<br>
*Install GOArbf through pip is available once it is published:
```
pip install GOArbf
```
## Running GOArbf
This repository aims to provide fast implementation of optimization algorithms, in the meantime, it also provides many parameters so users can easily adjust the algorithms accordingly. To run this repository, users need to first define the optimization problems and choose an optimizer which will return the best solution and its corresponding function value.The "examples" folder provide sample code for all the software included and the "GOArbf\Optimize" folder provides the source code for all the optimizers.<br>
1. For single objective optimization problems, DYCORS(in pySOT) is recommended:
 ```
from GOArbf.Optimize import Optim
Optim.optimize(problem)
```
2. For multi-objective optimization problems, GOMORS is recommended:
```
from GOArbf.Optimize import MOoptim
MOoptim.MOoptimize(problem)
```
3. For many-objective (more than 3) optimization problems, 𝜀-MaSO is recommended:
```
from GOArbf.Optimize import epsMOoptim
epsMOoptim.epsMOoptimize(problem)
```
4. For parallel optimization on high dimensional problems, GOPS is recommended:
```
from GOArbf.Optimize import GOPSoptim
GOPSoptim.GOPSoptimize(problem)
```
5. For multi-modal optimization problems when multi-fidelity models are available, MRSO is recommended:
```
from GOArbf.Optimize import MFoptim
MFoptim.MFoptimize(problem)
```
When you are formulating your optimization problems, you can follow the example tests problems used for each algorithms. For example, if you are using DCYORS, you can refer to pySOT.optimization_problems for example problem formulations.
## Citation
If you use pySOT, please cite the following paper: [David Eriksson, David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for surrogate optimization. arXiv preprint arXiv:1908.00420, 2019](https://arxiv.org/abs/1908.00420)<br>
If you use GOMORS, please cite the following paper: [Akhtar, T., Shoemaker, C.A. Multi objective optimization of computationally expensive multi-modal functions with RBF surrogates and multi-rule selection. J Glob Optim 64, 17–32 (2016).](https://doi.org/10.1007/s10898-015-0270-y)<br>
If you use 𝜀-MaSO, please cite the following paper:[Wang, W., Akhtar, T. & Shoemaker, C.A. Integrating 𝜀-dominance and RBF surrogate optimization for solving computationally expensive many-objective optimization problems. J Glob Optim (2021).](https://doi.org/10.1007/s10898-021-01019-w)<br>
If you use GOPS, please cite the following paper:[Xia, W., Shoemaker, C. GOPS: efficient RBF surrogate global optimization algorithm with high dimensions and many parallel processors including application to multimodal water quality PDE model calibration. Optim Eng (2020).](https://doi.org/10.1007/s11081-020-09556-1)<br>
If you use MRSO, please cite the following paper:[Yi, J., Shen, Y. & Shoemaker, C.A. A multi-fidelity RBF surrogate-based optimization framework for computationally expensive multi-modal problems with application to capacity planning of manufacturing systems. Struct Multidisc Optim 62, 1787–1807 (2020).](https://doi.org/10.1007/s00158-020-02575-7)<br>
The papers above are focused on continuous variables, for mixed-integer optimization please refer to HORD and cite the following paper:
[Ilievski, Ilija, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker. 2017. “Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates.” Pp. 822–29 in Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17. San Francisco, California, USA: AAAI Press.](https://arxiv.org/pdf/1607.08316.pdf)

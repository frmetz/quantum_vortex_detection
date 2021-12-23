# Deep learning based quantum vortex detection in atomic Bose-Einstein condensates

This repository contains the models and code used for the paper [Deep-learning-based quantum vortex detection in atomic Bose–Einstein condensates](https://iopscience.iop.org/article/10.1088/2632-2153/abea6a).

The code is provided in the form of [Jupyter Notebooks](https://jupyter.org/) using the [Julia language](https://julialang.org/) and the machine learning library [Flux](https://fluxml.ai/Flux.jl/stable/). The notebooks require [IJulia](https://github.com/JuliaLang/IJulia.jl).

### Content
* *vortex_detection.ipynb* - jupyter notebook used for training and evaluating the ML based vortex detector
* *vortex_detection_classification.ipynb* - slightly modified version where the circulation direction of a vortex is also classified
* *utils.jl*               - additional functions used in the notebooks
* *example_data/*          - example images to run the notebook
* *models/*                - pretrained models discussed in the paper

If you have any questions, feel free to open an issue or send me an email: <friederike.metz@oist.jp>.

If you use our code/models for your research, consider citing our paper:
```
@article{Metz_2021,
	doi = {10.1088/2632-2153/abea6a},
	url = {https://doi.org/10.1088/2632-2153/abea6a},
	year = 2021,
	month = {jun},
	publisher = {{IOP} Publishing},
	volume = {2},
	number = {3},
	pages = {035019},
	author = {Friederike Metz and Juan Polo and Natalya Weber and Thomas Busch},
	title = {Deep-learning-based quantum vortex detection in atomic Bose{\textendash}Einstein condensates},
	journal = {Machine Learning: Science and Technology},
}
```

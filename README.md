# Deep learning based quantum vortex detection in atomic Bose-Einstein condensates

This repository contains the models and code used for the paper [Deep learning based quantum vortex detection in atomic Bose-Einstein condensates](https://arxiv.org/abs/2012.13097).

The code is written in Julia and based on the machine learning library [FLux](https://fluxml.ai/Flux.jl/stable/).

### Content
* *vortex_detection.ipynb* - jupyter notebook used for training and evaluating the ML based vortex detector
* *vortex_detection_classification.ipynb* - slightly modified version where the circulation direction of a vortex is also classified
* *utils.jl*               - additional functions used in the notebooks
* *example_data/*          - example images to run the notebook
* *models/*                - pretrained models discussed in the paper

If you have any questions, feel free to open an issue or send me an email: <friederike.metz@oist.jp>.

# Forensicability of Deep Neural Network Inference Pipelines

This is the source code repository accompanying our accepted [ICASSP'21 paper](https://arxiv.org/abs/2102.00921).
It is intended to make reproducing our experiments as easy as possible.

## Citation

If you use this work, please cite our paper:
```bibtex
@misc{schloegl2021forensicability,
      title={Forensicability of Deep Neural Network Inference Pipelines}, 
      author={Alexander Schlögl and Tobias Kupek and Rainer Böhme},
      year={2021},
      eprint={2102.00921},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Configuration

We provide a configuration template including comments on how to use it in [config.py](resources/config.py).
The best indicator for the specific usage of a configuration variable is the references of the variable, as our repository does not have a large amount of code.

## Training and weights

The model weights we used in our experiments are included in the [weights](weights/) directory.
If you want, you can retrain the models using the [training script](train.py).

## Predicting

Local predictions can be generated using the [prediction script](predict.py).
Remote generations can be done and retrieved using the [remote predictions script](get_remote_predictions.py)

## Generating TF Lite execution graphs

We have a script for [generating TF Lite execution graphs](convert_to_tflite.py)
Our prediction script can handle the generated flattened execution graphs.

## Generating boundary samples

Boundary samples can be generated with our [provided script](generate_boundary.py) and evaluated with [another script](evaluate_boundary.py).
We also have a [shell script](build_all_boundaries.sh), that will generate boundaries for all models, and has hyperparameters pre-set.


## Cloud machines
If you want to compare different CPU types, an easy to do this is by using cloud instances. Create an instance, install python, clone the repository and install the dependencies with `conda`.
The best way to get predictions from the remote instance is to use the [remote predictions script](get_remote_predictions.py).


## Issues and contact

If you run into issues while reproducing our research, please create an issue.
Alternatively, or if you are interested in cooperation, feel free to send me an email at [alexander.schloegl@uibk.ac.at](mailto:alexander.schloegl@uibk.ac.at).

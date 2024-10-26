# Enhancements and Critique of Flip-Flop Neurons

A project that discusses enhancements and critique of Flip-Flop neurons suggested in "The flip-flop neuron: a memory efficient alternative for solving challenging sequence processing and decision-making problems" [(Kumari et al., 2023)](https://link.springer.com/article/10.1007/s00521-023-08552-7#:~:text=04%20May%202023-,The%20flip%2Dflop%20neuron%3A%20a%20memory%20efficient%20alternative%20for%20solving,processing%20and%20decision%2Dmaking%20problems). 

## Project Proposal

The project aims to contribute through a comprehensive literature review, ablation studies, integration of conventional ML techniques, performance benchmarking across various tasks, and comparative analysis with novel ML methods to evaluate the flip-flop neuron model's efficacy. A detailed doc can be found [here](./documents/Project-Proposal.pdf).

TODO - Add final report once done

## Environment Setup

_Setup the conda environment_
```bash
conda create --name incm python=3.9 -y
conda activate incm
pip install -r requirements.txt
```

_Setup precommit_
```bash
pre-commit install
```

## Instructions

### General Guidelines

For each run, create a folder in the format of `src/experiments/<data or project name>/<run name>`. Create the configuration file within this folder, for example, [src/experiments/dummy_data/exp_0/config.yaml](src/experiments/dummy_data/exp_0/config.yaml). Make sure to store the final plots and metadata (such as final accuracies as JSON files) within this directory.

### Training over dummy data

```bash
python -u src/train_dummy_data.py --config src/experiments/dummy_data/exp_0/config.yaml > temp.txt
```

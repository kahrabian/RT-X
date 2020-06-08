# Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs

## Introduction

This repository provides the PyTorch implementation of the _RT-X_ model presented in _Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs_ paper.

## Results

Experiment results as presented in the paper.

| Dataset   | Type          | Model                 | HITS@1    | HITS@3    | HITS@10   | MR        | MRR       |
| :-------: | :------------:| :-------------------: | :-------: |:--------: | :-------: | :-------: | :-------: |
|           |               | RotatE                | 47.58     | 76.66     | 88.95     | 807.40    | 0.6328    |
|           | Interpolated  | DE-RotatE             | 47.98     | 76.92     | 88.87     | 779.50    | 0.6349    |
| GITHUB-SE |               | RT-DE-RotatE (ours)   | 49.70     | 78.67     | 90.48     | 773.90    | 0.6522    |
| 1M-NODE   |               | RotatE                | 25.40     | 49.02     | 57.54     | 4762.87   | 0.3797    |
|           | Extrapolated  | DE-RotatE             | 26.28     | 48.53     | 57.33     | 4840.16   | 0.3838    |
|           |               | RT-DE-RotatE (ours)   | 26.50     | 49.54     | 57.94     | 4891.81   | 0.3888    |
|           |               | RotatE                | 44.05     | 57.14     | 80.95     | 18.54     | 0.5460    |
|           | Interpolated  | DE-RotatE             | 42.17     | 53.88     | 76.88     | 24.67     | 0.5233    |
| GITHUB-SE |               | RT-DE-RotatE (ours)   | 48.93     | 60.96     | 78.32     | 14.47     | 0.5815    |
| 1Y-REPO   |               | RotatE                | 2.11      | 4.82      | 9.71      | 1917.03   | 0.0464    |
|           | Extrapolated  | DE-RotatE             | 1.77      | 4.08      | 9.10      | 1961.75   | 0.0402    |
|           |               | RT-DE-RotatE (ours)   | 37.30     | 44.36     | 48.64     | 1232.22   | 0.4183    |

## Execution

In `run.template.sh` we provide a comprehensive example for running the code.

To check all the available arguments, you can run `python codes/run.py --help`.

## Reproducibility

To reproduce the _RE-X_ results presented in the ICML 2020 GRL+ Workshop paper _Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs_, you can use the runner provided in `runner.sh`.

## Acknowledgments

Our implementation is based on the PyTorch implementation of the RotatE model provided in [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

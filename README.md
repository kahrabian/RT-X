# Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs

## Introduction

This repository provides the PyTorch implementation of the _RT-X_ model presented in [_Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs_](https://arxiv.org/abs/2007.01231) paper.

## Results

Experiment results as presented in the paper. Results within the 95% confidence interval of the best are bolded.

| Dataset           | Type          | Model                 | HITS@1    | HITS@3    | HITS@10   | MR            | MRR           |
| :---------------: | :-----------: | :-------------------: | :-------: |:--------: | :-------: | :-----------: | :-----------: |
| GITHUB-SE 1M-NODE | Interpolated  | RotatE                | 47.58     | 76.66     | 88.95     | **807.40**    | 0.6328        |
|                   |               | DE-RotatE             | 47.98     | 76.92     | 88.87     | **779.50**    | 0.6349        |
|                   |               | RT-DE-RotatE (ours)   | **49.70** | **78.67** | **90.48** | **773.90**    | **0.6522**    |
|                   | Extrapolated  | RotatE                | 25.40     | **49.02** | **57.54** | **4762.87**   | 0.3797        |
|                   |               | DE-RotatE             | **26.28** | 48.53     | **57.33** | **4840.16**   | **0.3838**    |
|                   |               | RT-DE-RotatE (ours)   | **26.50** | **49.54** | **57.94** | **4891.81**   | **0.3888**    |
| GITHUB-SE 1Y-REPO | Interpolated  | RotatE                | 44.05     | 57.14     | **80.95** | 18.54         | 0.5460        |
|                   |               | DE-RotatE             | 42.17     | 53.88     | 76.88     | 24.67         | 0.5233        |
|                   |               | RT-DE-RotatE (ours)   | **48.93** | **60.96** | 78.32     | **14.47**     | **0.5815**    |
|                   | Extrapolated  | RotatE                | 2.11      | 4.82      | 9.71      | 1917.03       | 0.0464        |
|                   |               | DE-RotatE             | 1.77      | 4.08      | 9.10      | 1961.75       | 0.0402        |
|                   |               | RT-DE-RotatE (ours)   | **38.25** | **40.08** | **64.06** | **1195.02**   | **0.4345**    |

## Execution

In `run.template.sh` we provide a comprehensive example for running the code.

To check all the available arguments, you can run `python main.py --help`.

## Reproducibility

To reproduce the _RT-X_ results presented in the [ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+)](https://grlplus.github.io/) paper [_Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs_](https://arxiv.org/abs/2007.01231), you can use the runner provided in `runner.sh`.

## Licenses

The code is released under the MIT license, see [LICENSE](LICENSE) and [LICENSE_RotatE](LICENSE_RotatE).
The data is released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license, also accessible [here](https://zenodo.org/record/3928580).

## Acknowledgments

Our implementation is based on the PyTorch implementation of the RotatE model provided in [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

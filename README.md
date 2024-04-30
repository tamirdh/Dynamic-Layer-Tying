# Dynamic Layer Tying for Parameter Efficient Transformers

This repository contains the implementation of the research presented in the paper "Dynamic Layer Tying for Parameter Efficient Transformers" by Tamir David-Hay and Lior Wolf, published at ICLR 2024. The method introduces a novel approach to reduce the number of trainable parameters in transformer models by using a reinforcement learning strategy to dynamically tie layers during training.

## Repository Structure

- `code/`: Directory containing all the source code for the experiments and model implementation.
- `README.md`: This file, describing the project and navigation.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0.1 or higher
- transformers 4.3 or higher
- accelerate 0.27.0 or higher


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourgithubusername/dynamic-layer-tying.git
    ```
2. Install the required dependencies
3. To run a DQN experiment, run the following command:
   ```bash
   cd code
   accelerate launch dqn_trainer.py --bptt=<Sequenece length> --batch_size=<Training batch size> --eval_batch_size=<Evaluation batch size> --dataset=<Dataset name> --nlayers=<Number of decoder layers> --lr=<Learning rate> --emsize=<Embedding size> --nhead=<Number of attention heads per decoder block> --exp=<Experiment name> --epochs=<Number of training epochs>
   ```
4. To run a BERT model update [line 625 of dqn_trainer.py](code/dqn_trainer.py#L625).


## Citation

If you find this work useful, please consider citing the following paper:

```
@inproceedings{
hay2024dynamic,
title={Dynamic Layer Tying for Parameter-Efficient Transformers},
author={Tamir David Hay and Lior Wolf},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=d4uL2MSe0z}
}
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Acknowledgements

Tel Aviv University Center for AI and Data Science for funding support

ICLR 2024 for an opportunity to present our work


This README provides an overview of the repository, instructions for setting up the environment, running the models, and information about the datasets used, along with citation details. Adjust the paths and add any specific details necessary to match your actual repository structure and requirements.
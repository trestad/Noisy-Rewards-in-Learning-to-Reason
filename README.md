# The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason

[Paper Link](./noisy_rewards_in_learning_to_reason.pdf)


## Setup

```sh
git clone https://github.com/trestad/Noisy-Rewards-in-Learning-to-Reason.git
conda create -n think python=3.12
conda activate think
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip3 install ray
cd verl
pip install -e .
cd ..
pip3 install flash-attn --no-build-isolation
```

## Training Instructions

### Math Training

To train the model on the ORZ dataset, use the following command:

```sh
bash scripts/train_rl_math_7b_orz_flip.sh FLIP_PROBABILITY
```

Before training, please ensure you replace the file at the following location with your own math verification code:

```
verl/verl/utils/reward_score/math_utils.py
```
This is because the `math_utils.py` file we used contains proprietary company-specific code that cannot be made public now. 
We are currently working on extracting and open-sourcing the non-sensitive portions.
However, as we discuss in the paper, verification accuracy is not a major concern, so you can confidently substitute this file with any math verification script of your choice.

### HelpSteer3 Training

For training on the HelpSteer3 dataset, use:

```sh
bash scripts/train_rl_help_qwen_flip.sh RM_PATH
```

The training set is already available in the `dataset/` folder. Since some sensitive information in the validation set cannot pass GitHub’s review, you'll need to download the validation set [here](https://drive.google.com/file/d/1ABrlXPXdC34oTUg2otrlRMihxITeoesq/view?usp=sharing).

### RM Checkpoint Download

[65% Acc]()

[85% Acc]()


### Acknowledgments

* Our code is based on [verl](https://github.com/volcengine/verl).
* The datasets are derived from [orz](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) and [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3).
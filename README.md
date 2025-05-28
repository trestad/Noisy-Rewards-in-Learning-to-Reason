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

### Math Evaluation

After training a few steps, the code automatically evaluates the model on three test sets (AIME, GPQA, MATH500).

### HelpSteer3 Training

For training on the HelpSteer3 dataset, use:

```sh
bash scripts/train_rl_help_qwen_flip.sh RM_PATH
```

The training set is already available in the `dataset/` folder. Since some sensitive information in the validation set cannot pass GitHub’s review, you'll need to download the validation set [here](https://drive.google.com/file/d/1ABrlXPXdC34oTUg2otrlRMihxITeoesq/view?usp=sharing).

### HelpSteer3 Evaluation

```sh
bash scripts/help_generate.sh REASONING_MODEL_PATH OUTPUT_PATH
```

Use ```gpt_eval.py``` to compare two models' outputs.

### RM Checkpoint Download

[65% Acc](hhttps://huggingface.co/AngLv/NoisyRewards-in-RL-RM-acc-65/tree/main)

[85% Acc](https://huggingface.co/AngLv/NoisyRewards-in-RL-RM-acc-85/tree/main)


### Acknowledgments

* Our code is based on [verl](https://github.com/volcengine/verl).
* The datasets are derived from [orz](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) and [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3).
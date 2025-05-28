import os
import datasets
import argparse
from verl.utils.reward_score.answer_extraction import extract_math_answer

def make_prefix(question):
    # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer marked with \\boxed{{}} and within <answer> </answer> tags, i.e., <answer>\\boxed{{answer}}</answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    args = parser.parse_args()

    # Load dataset with different splits
    dataset = datasets.load_dataset('json', data_files={
        'train': '/mnt/wx_feature/home/anglv/Think/datasets/math_500/train.jsonl',
        'test': '/mnt/wx_feature/home/anglv/Think/datasets/math_500/test.jsonl'
    })

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/Qwen2.5-7B-Instruct/', use_fast=False)
    
    # input(extract_math_answer('Squaring both sides of the given equation, we get \[2 + \sqrt{x} = 9.\]Then, $\sqrt{x} = 9-2 = 7.$ Squaring again gives $x = 49.$ We check our answer by substituting $x = 49$ into the given equation: \[\sqrt{2+\sqrt{x}} = \sqrt{2 + \sqrt{49}} = \sqrt{2 + 7} = \sqrt{9} = 3.\]Therefore, $x = \\boxed{49}$ is the correct solution. (The step of checking the answer is necessary because squaring both sides of an equation sometimes introduces extraneous roots -- solutions which do not actually satisfy the original equation.)'))
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example['problem']
            question = make_prefix(question_raw)
            solution = example['solution']

            # print(solution)
            answer = extract_math_answer(solution)[0]
            # if len(answer) == 0:
            #     input(question)
            #     input(solution)
            #     input(example['answer'])
            # else:
            #     answer = answer[0]
            # input(answer)
            data = {
                "data_source": 'math_500',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

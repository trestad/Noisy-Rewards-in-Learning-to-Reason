import os
import datasets
import argparse

def make_prefix(question):
    # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer marked with \\boxed{{}} and within <answer> </answer> tags, i.e., <answer>\\boxed{{answer}}</answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag. {question} Assistant: <think>"""
    return prefix

def make_multi_choice_prefix(question):
    # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a mathematical reasoning problem. After thinking, when you finally reach a solution, clearly state the answer marked with \\boxed{{}} and within <answer> </answer> tags, i.e., <answer>\\boxed{{answer}}</answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"""
    prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer choice (a single capital letter) inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag. {question} Assistant: <think>"""
    return prefix

def make_train_map_fn(split):

    def process_fn(example, idx):
        question = example['0']['value']
        question = make_prefix(question)
        answer = example['1']["ground_truth"]["value"]
        # input(question)
        # input(answer)
        data = {
            "data_source": 'orz',
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

def make_math_map_fn(split):

    def process_fn(example, idx):
        # input(example)
        question = example['prompt'][0]['value']
        question = make_prefix(question)
        answer = example["final_answer"]
        # input(question)
        # input(answer)
        data = {
            "data_source": 'math500',
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

def make_aime_map_fn(split):

    def process_fn(example, idx):
        question = example['prompt'][0]['value']
        question = make_prefix(question)
        answer = example["final_answer"]
        # input(question)
        # input(answer)
        data = {
            "data_source": 'aime',
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

def make_gpqa_map_fn(split):

    def process_fn(example, idx):
        # input(example)
        question = example['prompt'][0]['value']
        question = make_multi_choice_prefix(question)
        answer = example["final_answer"]
        # input(question)
        # input(answer)
        data = {
            "data_source": 'gpqa',
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    args = parser.parse_args()
    
    local_dir = args.local_dir

    train_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Think/datasets/orz/orz_math_57k_collected.json',
    })['data']
    train_dataset = train_dataset.map(function=make_train_map_fn('train'), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    
    math_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/math500.json',
    })['data']
    math_test_dataset = math_test_dataset.map(function=make_math_map_fn('test'), with_indices=True)
    math_test_dataset.to_parquet(os.path.join(local_dir, 'math_test.parquet'))
    
    
    aime_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/aime2024.json',
    })['data']
    aime_test_dataset = aime_test_dataset.map(function=make_aime_map_fn('test'), with_indices=True)
    aime_test_dataset.to_parquet(os.path.join(local_dir, 'aime_test.parquet'))
    
    
    gpqa_test_dataset = datasets.load_dataset('json', data_files={
        'data': '/mnt/wx_feature/home/anglv/Open-Reasoner-Zero/data/eval_data/gpqa_diamond.json',
    })['data']
    gpqa_test_dataset = gpqa_test_dataset.map(function=make_gpqa_map_fn('test'), with_indices=True)
    gpqa_test_dataset.to_parquet(os.path.join(local_dir, 'gpqa_test.parquet'))

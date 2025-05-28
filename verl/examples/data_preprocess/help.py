import os
import datasets
import argparse

def make_map_fn(split):

    def process_fn(example, idx):

        chat = example['context']
        question = chat[-1]['content']
        chat[-1] = {'role':'user', 'content': f'I present a question, and you, the assistant, first thinks about the reasoning process in the mind and then provides the user with the answer. Enclose the reasoning within <think>...</think> tags and the final answer — which should also summarize the reasoning — within <answer>...</answer> tags. For example: <think>Reasoning process here</think> <answer>Answer with summary of reasoning here</answer>. Now, here is my question: {question}'}
        # input(chat)
        
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/root/Qwen2.5-7B')
        # prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        # input(prompt_with_chat_template)

        data = {
            "data_source": 'help',
            "prompt": chat,
            "ability": "help",
            "reward_model": {
                "style": "rule",
                "ground_truth": None,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        # input(data)
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/wx_feature/home/anglv/Think/datasets/help-think')
    args = parser.parse_args()
    
    local_dir = args.local_dir

    train_dataset = datasets.load_dataset('nvidia/HelpSteer3')['train']
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset = train_dataset.remove_columns(['context', 'response1', 'response2', 'overall_preference', 'individual_preference'])
    # input(train_dataset[0])
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    test_dataset = datasets.load_dataset('nvidia/HelpSteer3')['validation']
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset = test_dataset.remove_columns(['context', 'response1', 'response2', 'overall_preference', 'individual_preference'])
    test_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))

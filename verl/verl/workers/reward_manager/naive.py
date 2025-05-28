# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import random

def format_reward_anneal(score, step):
    """Anneal the reward based on the step.
    """
    return score
    # Anneal the reward based on the step
    # anneal_factor = 1.0 - min(1.0, step / 150)
    # return score * anneal_factor

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto, rollout_n=-1, flip_prob=0.0, step=0, is_test=False, is_check_correctness=True, add_format_reward_when_wrong=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        gt_tensor = torch.zeros(data.batch.batch_size[0], dtype=torch.bool, device=reward_tensor.device)
        
        already_print_data_sources = {}

        do_flip = False
        
        # for i in range(len(data)):
        #     data_item = data[i]
        #     print(f"{data_item.batch['prompts'].device} {data_item.batch['prompts'][-20:]}")
            
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids)) # modifed by anglv for eaiser answer extraction. 
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # print('debug debug debug debug' * 20)
            score, format_score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                response_length=valid_response_length,
                extra_info=extra_info,
                is_test=is_test,
                is_check_correctness=is_check_correctness,
            )
            
            if not is_test and flip_prob > 0 and rollout_n > 0 and i % rollout_n == 0 and is_check_correctness:
                do_flip = flip_prob > random.random()
            
            if not is_test and rollout_n > 0 and i % rollout_n == 0 and i >= rollout_n:
                for j in range(i - rollout_n, i):
                    assert torch.equal(data[j].batch['prompts'], data[i - 1].batch['prompts']), f'{data[j].batch['prompts'][-100:]}, {data[i-1].batch['prompts'][-100:]}'
            
            if format_score is not None:
                if do_flip and is_check_correctness:  
                    if score == 1.0 and add_format_reward_when_wrong:
                        print(f"****Random flip: a format reward to a wrong answer**** {format_reward_anneal(format_score, step)}\n")
                        reward_tensor[i, valid_response_length - 1] = format_reward_anneal(format_score, step)
                    else:
                        print(f"****Random flip: give a flipped reward**** {1-score}\n")
                        reward_tensor[i, valid_response_length - 1] = 1 - score
                else:
                    if not is_check_correctness:
                        assert format_score == score, f"format_score: {format_score}, score: {score}"
                    
                    if score == 0.0 and add_format_reward_when_wrong: 
                        print(f"****No flip: give a format reward to a wrong answer**** {format_reward_anneal(format_score, step)}\n")
                                                
                        reward_tensor[i, valid_response_length - 1] = format_reward_anneal(format_score, step)
                    else:
                        print(f"****No flip: give a correct reward**** {score}\n")
                        reward_tensor[i, valid_response_length - 1] = score
                    
            else:
                assert score == 0.0
                reward_tensor[i, valid_response_length - 1] = score
                
            gt_tensor[i] = score > 0

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor, gt_tensor

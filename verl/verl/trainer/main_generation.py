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
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import pandas as pd
from pprint import pprint
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.utils import hf_tokenizer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from datasets import Dataset

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))

def process_aug(example, idx):
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('/root/Qwen2.5-7B')
    # prompt_with_chat_template = tokenizer.apply_chat_template(chat, tokenize=False)
    # input(prompt_with_chat_template)

    data = {
        "chat": example['chat'],
        "response": example['response'],
    }
    return data

@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = config.model.path
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path).sample(n=200, random_state=42).reset_index(drop=True)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    test_output = {'response': [], 'chat': []}
    
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    
    # assert num_batch * config_batch_size >= total_samples
    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # for n_sample in range(config.data.n_samples):
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        
        output_texts = []
        
        try:
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = data_item.batch['responses'][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)
            test_output['response'].extend(output_texts)
            test_output['chat'].extend(batch_chat_lst)
        except:
            print('pass batch')
        

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    # test_output['responses'] = np.array(test_output['responses'], dtype=object)
    # test_output['responses'] = np.transpose(test_output['responses'], axes=(1, 0)).tolist()

    # add to the data frame
    # dataset['responses'] = output_lst

    # write to a new parquet
    dataset = Dataset.from_dict(test_output)
    dataset = dataset.map(function=process_aug, with_indices=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == '__main__':
    main()
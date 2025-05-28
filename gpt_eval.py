import asyncio
import json
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = """Your task is to help me determine which of two responses is more helpful in addressing the user's latest request.
First, I will provide you with the conversation history between the user and the AI assistant, where the last statement is the user's most recent request.
Then, I will show you the two assistant replies: response #1 and response #2.
Please evaluate which response is better based on factors such as helpfulness, the amount of information provided, the thoroughness of the reasoning, and overall coverage of the user's needs.
Please check two responses carefully, do not casually answer that #1 is better.
**Make sure to clearly state whether '#1 is better' or '#2 is better' or 'tie' AT THE END OF YOUR ANSWER**.

Here is the conversation history:
{chat}

Response #1:
{res1}

Response #2:
{res2}
"""

base_url = 
api_key = 
model_name = "gpt-4o"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

responses = []

def extract_think_and_answer(solution_str):      
    # think_start = solution_str.rfind("<think>")
    # think_end = solution_str.rfind("</think>")
    ans_start = solution_str.rfind("<answer>")
    
    # if think_start != -1 and think_end != -1 and ans_start != -1 and \
        # (think_end > think_start + len("<think>")) and think_end < ans_start:
            # think = solution_str[think_start + len("<think>"): think_end]
            # answer = solution_str[ans_start + len("<answer>"): ]
            # return think, answer
    
    # return None, None
    answer = solution_str[ans_start + len("<answer>"): ]
    return None, answer
    

def make_request(chat, res1, res2):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt.format(res1=res1, res2=res2, chat=chat)}],
        )
        # print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None


my_path = 
base_path = 
outpath = 
data1 = load_dataset("parquet", data_files=my_path)['train']
data2 = load_dataset("parquet", data_files=base_path)['train']

error_cnt = 0
win_cnt = 0
tie_cnt = 0
lose_cnt = 0

all_data = []
for i in range(len(data1)):
    assert data1['chat'][i] == data2['chat'][i]
    chat = data1['chat'][i]    
    chat[-1]['content'] = chat[-1]['content'].replace("I present a question, and you, the assistant, first thinks about the reasoning process in the mind and then provides the user with the answer. Enclose the reasoning within <think>...</think> tags and the final answer — which should also summarize the reasoning — within <answer>...</answer> tags. For example: <think>Reasoning process here</think> <answer>Answer with summary of reasoning here</answer>. Now, here is my question: ", "")
    think1, ans1 = extract_think_and_answer(data1['response'][i])
    think2, ans2 = extract_think_and_answer(data2['response'][i])
    all_data.append((chat, ans1, ans2))


with ThreadPoolExecutor(max_workers=48) as executor, open(outpath, 'a+') as f, tqdm(total=len(all_data)) as pbar:
    futures = []
    for i, data in enumerate(all_data):
        futures.append(executor.submit(make_request, data[0], data[1], data[2]))

    for future in as_completed(futures):
        
        pbar.update(1)
        response = future.result()
        if response is None:
            error_cnt += 1
            print(f"Error Cnt: {error_cnt}")
            continue
        
        if '#1 is better' in response.lower()[-30:]:    
            win_cnt += 1
        if '#2 is better' in response.lower()[-30:]:
            lose_cnt += 1
        if 'tie' in response.lower()[-30:]:
            tie_cnt += 1
         
        f.write(json.dumps(response) + "\n")
        f.flush()

data1 = load_dataset("parquet", data_files=my_path)['train']
data2 = load_dataset("parquet", data_files=base_path)['train']

all_data = []
for i in range(len(data1)):
    assert data1['chat'][i] == data2['chat'][i]
    # chat = tokenizer.apply_chat_template(data1[i]['chat'],tokenize=False)
    chat = data1['chat'][i]    
    chat[-1]['content'] = chat[-1]['content'].replace("I present a question, and you, the assistant, first thinks about the reasoning process in the mind and then provides the user with the answer. Enclose the reasoning within <think>...</think> tags and the final answer — which should also summarize the reasoning — within <answer>...</answer> tags. For example: <think>Reasoning process here</think> <answer>Answer with summary of reasoning here</answer>. Now, here is my question: ", "")
    think1, ans1 = extract_think_and_answer(data1['response'][i])
    think2, ans2 = extract_think_and_answer(data2['response'][i])
    all_data.append((chat, ans2, ans1))


with ThreadPoolExecutor(max_workers=48) as executor, open(outpath, 'a+') as f, tqdm(total=len(all_data)) as pbar:
    futures = []
    for i, data in enumerate(all_data):
        futures.append(executor.submit(make_request, data[0], data[1], data[2]))

    for future in as_completed(futures):
        
        pbar.update(1)
        response = future.result()
        if response is None:
            error_cnt += 1
            print(f"Error Cnt: {error_cnt}")
            continue
        
        if '#1 is better' in response.lower()[-30:]:    
            lose_cnt += 1
        if '#2 is better' in response.lower()[-30:]:
            win_cnt += 1
        if 'tie' in response.lower()[-30:]:
            tie_cnt += 1
         
        #data[i]['line_idx'] = i
        f.write(json.dumps(response) + "\n")
        f.flush()     
        
total_cnt = win_cnt + lose_cnt + tie_cnt
print(f"Total: {total_cnt}, Error: {error_cnt}, win: {win_cnt / total_cnt}, Tie: {tie_cnt / total_cnt}, Lose: {lose_cnt / total_cnt}")

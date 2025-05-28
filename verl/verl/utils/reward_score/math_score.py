import os
from collections import Counter
from verl.utils.reward_score.math_utils import is_equiv

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None

def solution2answer(solution: str) -> str:
    answer = remove_boxed(last_boxed_only_string(solution))
    if answer is not None:
        return answer
    return solution

def calculate_ngram_repetition_ratio(text, n):
    # 将文本按空格拆分成单词
    words = text.split()
    
    # 生成n-gram列表
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    # 统计n-gram的频率
    ngram_counts = Counter(ngrams)
    
    # 计算重复的n-gram数量
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    
    # 计算重复n-gram占总n-gram数的比例
    repetition_ratio = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    return repetition_ratio

def think_reward(solution_str):
    # 找到solution_str中出现“Assistant: <think> ”的位置，取该位置后面的内容
    think_pos = solution_str.find("Assistant: <think>")
    solution_str = solution_str[think_pos + len("Assistant: <think> "):]
    score = 0
    solution_str = solution_str.lower()
    score += float('i need to' in solution_str)
    score += float('we need to' in solution_str)
    score += float('wait' in solution_str)
    score += float('alternatively' in solution_str)
    score += float('let me check' in solution_str)
    score += float('let me see' in solution_str)
    score += float("let's focus on" in solution_str)
    score += float("we know that" in solution_str)
    score += float("we can observe " in solution_str)
    score += float("we can see " in solution_str)
    score += float("let me try" in solution_str)
    score += float("let's try" in solution_str)
    score += float("let us try" in solution_str)
    score += float("first, " in solution_str) 
    score += float("firstly, " in solution_str)
    score += float("next, " in solution_str)
    score += float("finally, " in solution_str)
    score += float("let us first" in solution_str)
    score += float("let's first" in solution_str) 
    score += float("let me first" in solution_str)
    score += float("try again" in solution_str) 
    score += float("still not" in solution_str)
    score += float("not working" in solution_str) 
    score += float("not correct" in solution_str)
    score += float("does not work" in solution_str)
    score += float("doesn't work" in solution_str)
    score += float("makes sence" in solution_str)
    score += float("since we " in solution_str) 
    score += float("because we " in solution_str)
    score += float("consequently" in solution_str)
    score += float("as a result" in solution_str) 
    score += float("thus" in solution_str)
    score += float("therefore" in solution_str)
    score += float("hence" in solution_str)
    score += float("so that" in solution_str)
    score += float("thereby" in solution_str)
    score += float("if we" in solution_str)
    score += float("given there " in solution_str)
    score += float("for instance " in solution_str)
    score += float("for example " in solution_str)
    score /= 20
    score = min(1, score)
    score -= calculate_ngram_repetition_ratio(solution_str, 20)
    score -= solution_str.count('user: ') # 自己出多个题自己解直接给0分，现在的solution是“user: ”之后的内容，所以不应该再有任何的"user: "
    score = max(0, score)
    return score

def compute_score(solution_str: str, ground_truth: str, response_length: int, verbose=True, is_test=False, is_check_correctness=True):
    """
    Computes comprehensive score for model response.
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth data
        answer_reward: Points awarded/deducted for answer correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    max_context_length = 4096
    
    answer_text = solution2answer(solution_str)
    ground_truth = solution2answer(ground_truth)
    score = 0.0
    
    print("\n\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))
    print(f"[Ground Truth]\n{ground_truth}\n")
    print(f"[Model Response]\n{solution_str}\n")
    print(f"[Extracted Answer]\n{answer_text}\n")
        
    if not answer_text:
        print('thinking format wrong, return') # 得不到答案，不鼓励format
        return 0.0, None
    
    score = None
    if is_test or is_check_correctness:  
        score = float(is_equiv(ground_truth, answer_text))
        print(f"[Is Correct?]\n{score}")

    format_score = float(think_reward(solution_str))
    print(f"[Format Reward]\n{format_score}")
    if score is None:
        score = format_score
        print(f"[Format Reward is Score]")
    
    print("=" * 80 + "\n")
    # if verbose:
        # print("\n\n" + "=" * 80)
        # print(" Processing New Sample ".center(80, '='))
        # print(f"[Ground Truth]\n{ground_truth}\n")
        # print(f"[Model Response]\n{solution_str}\n")
        # print(f"[Extracted Answer]\n{answer_text}\n")
        # print(f"[Is Correct?]\n{score}")
        # print("=" * 80 + "\n")

    return score, format_score
import re
import torch
torch.cuda.empty_cache()
from typing import List
from copy import deepcopy

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""

def count_xml(text) -> float:
    """
    Count XML tags in response.
    
    Args:
        text: Input text
        
    Returns:
        Score based on XML tag presence
    """
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.5
    if text.count("</think>") == 1:
        count += 0.5
    return count

def extract_xml_answer(text: str) -> str:
    """
    Extract answer from XML-formatted text.
    
    Args:
        text: Input text with XML tags
        
    Returns:
        Extracted answer text
    """
    try:
        answer = text.split("</think>")[1]
        return answer.strip()
    except:
        return ""

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward function based on proper XML tag usage.
    
    Args:
        completions: Model completions
        
    Returns:
        List of reward scores
    """
    # contents = [completion[0]["content"] for completion in completions]
    contents = completions
    return [count_xml(c) for c in contents]

def int_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward function that checks if responses contain valid direction tokens.
    
    Args:
        completions: Model completions
        
    Returns:
        List of reward scores
    """
    allowed_tokens = {"<|up|>", "<|down|>", "<|right|>", "<|left|>"}
    
    # responses = [completion[0]['content'] for completion in completions]
    responses = completions
    extracted_responses = [extract_xml_answer(r) for r in responses]

    def is_valid_sequence(seq):
        
        seq_no_whitespace = re.sub(r'\s+', '', seq)
        if not seq_no_whitespace:
            return False
        found_tokens = re.findall(r'<\|(?:up|down|right|left)\|>', seq_no_whitespace)
        reconstructed = ''.join(found_tokens)
        if reconstructed != seq_no_whitespace:
            return False
        return all(token in allowed_tokens for token in found_tokens)
    
    return [1.0 if is_valid_sequence(r) else 0.0 for r in extracted_responses]

def count_turns(steps):
    moves = re.findall(r"<\|(.*?)\|>", steps)
    turns = sum(1 for i in range(1, len(moves)) if moves[i] != moves[i - 1])
    return moves, turns

def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    Reward function that checks correctness of answers.
    
    Args:
        prompts: Input prompts
        completions: Model completions
        answer: Ground truth answers
    
    Returns:
        List of reward scores
    """
    rewards = []
    responses = completions
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.debug('-'*20)
    # logger.debug(f"Question:\n{q}")
    logger.debug(f"\nAnswer:\n{answer[0]}")
    logger.debug(f"\nResponse:\n{responses[0]}")
    logger.debug(f"\nExtracted:\n{extracted_responses[0]}")
    for r, a in zip(extracted_responses, answer):
        r_steps, r_turns = count_turns(r)
        a_steps, a_turns = count_turns(a)
        if r == a:
            reward = len(r_steps) * 2 * (r_turns + 1)
        else:
            k = 0
            for r_s, a_s in zip(r_steps, a_steps):
                if r_s == a_s:
                    k += 1
                else:
                    break
            prefix = r_steps[:k]
            turns = count_turns("".join(prefix))[1]
            reward = k * 1 * (turns + 1)
        rewards.append(reward)
    return rewards

class MazeReward(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        # print(completions)
        rewards = correctness_reward_func(completions, solution)
        return rewards

class MazeFormat(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        # print(completions)
        rewards = int_reward_func(completions)
        return rewards

class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = xmlcount_reward_func(completions)
        return rewards

orms['external_r1v_acc'] = MazeReward
orms['external_r1v_format'] = MazeFormat
orms['format'] = Format
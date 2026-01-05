#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'cluster'))
from judge_prompts import get_all_dimension_keys

MODEL_MAPPING = {
    'gpt-3.5-turbo': 'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-0314': 'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-0613': 'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-1106': 'gpt-3.5-turbo-0125',
    'gpt-4-0314': 'gpt-4-0125-preview',
    'gpt-4-0613': 'gpt-4-0125-preview',
    'gpt-4-0125-preview': 'gpt-4-turbo-preview',
    'claude-2.1': 'claude-2.1',
}

MODEL_PRICING = {
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    'gpt-3.5-turbo-0125': {'input': 0.50, 'output': 1.50},
    'gpt-3.5-turbo-1106': {'input': 0.50, 'output': 1.50},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    'gpt-4-0125-preview': {'input': 10.00, 'output': 30.00},
    'gpt-4-turbo-preview': {'input': 10.00, 'output': 30.00},
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
    'gpt-4o': {'input': 5.00, 'output': 15.00},
    'claude-2.1': {'input': 8.00, 'output': 24.00},
    'claude-3-opus': {'input': 15.00, 'output': 75.00},
    'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
    'claude-3-haiku': {'input': 0.25, 'output': 1.25},
}

DEFAULT_TOKEN_ESTIMATES = {
    'pairwise': {
        'input_tokens': 2000,
        'output_tokens': 200,
    },
    'absolute': {
        'input_tokens': 1500,
        'output_tokens': 150,
    }
}


def get_available_model(retired_model: str) -> str:
    return MODEL_MAPPING.get(retired_model, retired_model)


def get_model_pricing(model: str) -> Optional[Dict[str, float]]:
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    
    mapped = get_available_model(model)
    if mapped != model and mapped in MODEL_PRICING:
        return MODEL_PRICING[mapped]
    
    return None


def estimate_tokens(text: str) -> int:
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) * 0.75
    return int((char_estimate + word_estimate) / 2)


def calculate_evaluation_cost(
    num_conversations: int,
    mode: str = 'pairwise',
    judge_model: str = 'gpt-3.5-turbo',
    avg_prompt_tokens: Optional[int] = None,
    avg_response_tokens: Optional[int] = None,
    avg_response_a_tokens: Optional[int] = None,
    avg_response_b_tokens: Optional[int] = None
) -> Dict:
    available_model = get_available_model(judge_model)
    pricing = get_model_pricing(available_model)
    
    if not pricing:
        return {
            'error': f'No pricing available for model: {judge_model} (mapped to: {available_model})',
            'judge_model': judge_model,
            'available_model': available_model
        }
    
    dimension_count = len(get_all_dimension_keys())
    if mode == 'pairwise':
        tasks_per_conv = 1
    elif mode == 'absolute':
        tasks_per_conv = dimension_count * 2
    elif mode == 'dimension_pairwise':
        tasks_per_conv = dimension_count
    elif mode == 'both':
        tasks_per_conv = 1 + dimension_count * 2
    else:  # all
        tasks_per_conv = 1 + dimension_count * 2 + dimension_count
    
    total_tasks = num_conversations * tasks_per_conv
    
    if mode == 'pairwise':
        input_per_task = (
            (avg_prompt_tokens or DEFAULT_TOKEN_ESTIMATES['pairwise']['input_tokens']) +
            (avg_response_a_tokens or 500) +
            (avg_response_b_tokens or 500)
        )
        output_per_task = avg_response_tokens or DEFAULT_TOKEN_ESTIMATES['pairwise']['output_tokens']
    elif mode == 'absolute':
        input_per_task = (
            (avg_prompt_tokens or DEFAULT_TOKEN_ESTIMATES['absolute']['input_tokens']) +
            (avg_response_a_tokens or 500)
        )
        output_per_task = avg_response_tokens or DEFAULT_TOKEN_ESTIMATES['absolute']['output_tokens']
    else:
        pairwise_input = (
            (avg_prompt_tokens or DEFAULT_TOKEN_ESTIMATES['pairwise']['input_tokens']) +
            (avg_response_a_tokens or 500) +
            (avg_response_b_tokens or 500)
        )
        absolute_input = (
            (avg_prompt_tokens or DEFAULT_TOKEN_ESTIMATES['absolute']['input_tokens']) +
            (avg_response_a_tokens or 500)
        )
        input_per_task = (pairwise_input + absolute_input * 12) / 13
        output_per_task = (
            (DEFAULT_TOKEN_ESTIMATES['pairwise']['output_tokens'] +
             DEFAULT_TOKEN_ESTIMATES['absolute']['output_tokens'] * 12) / 13
        )
    
    total_input_tokens = total_tasks * input_per_task
    total_output_tokens = total_tasks * output_per_task
    
    input_cost = (total_input_tokens / 1_000_000) * pricing['input']
    output_cost = (total_output_tokens / 1_000_000) * pricing['output']
    total_cost = input_cost + output_cost
    
    return {
        'judge_model': judge_model,
        'available_model': available_model,
        'mode': mode,
        'num_conversations': num_conversations,
        'tasks_per_conversation': tasks_per_conv,
        'total_tasks': total_tasks,
        'tokens_per_task': {
            'input': int(input_per_task),
            'output': int(output_per_task)
        },
        'total_tokens': {
            'input': int(total_input_tokens),
            'output': int(total_output_tokens),
            'total': int(total_input_tokens + total_output_tokens)
        },
        'pricing_per_1M': pricing,
        'costs': {
            'input': round(input_cost, 2),
            'output': round(output_cost, 2),
            'total': round(total_cost, 2)
        }
    }


def calculate_from_evaluation_set(evaluation_set_file: str, judge_model: str, mode: str = 'pairwise') -> Dict:
    with open(evaluation_set_file, 'r') as f:
        eval_data = json.load(f)
    
    evaluations = eval_data.get('evaluations', [])
    num_conversations = len(evaluations)
    
    if num_conversations == 0:
        return {'error': 'No evaluations found in file'}
    
    sample_size = min(100, num_conversations)
    sample = evaluations[:sample_size]
    
    prompt_tokens = []
    response_a_tokens = []
    response_b_tokens = []
    
    for eval_item in sample:
        prompt_tokens.append(estimate_tokens(eval_item.get('prompt', '')))
        response_a_tokens.append(estimate_tokens(eval_item.get('response_a', '')))
        response_b_tokens.append(estimate_tokens(eval_item.get('response_b', '')))
    
    avg_prompt = int(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None
    avg_response_a = int(sum(response_a_tokens) / len(response_a_tokens)) if response_a_tokens else None
    avg_response_b = int(sum(response_b_tokens) / len(response_b_tokens)) if response_b_tokens else None
    
    return calculate_evaluation_cost(
        num_conversations=num_conversations,
        mode=mode,
        judge_model=judge_model,
        avg_prompt_tokens=avg_prompt,
        avg_response_a_tokens=avg_response_a,
        avg_response_b_tokens=avg_response_b
    )


def print_cost_summary(cost_data: Dict):
    if 'error' in cost_data:
        print(f"Error: {cost_data['error']}")
        return
    
    print("\n" + "="*60)
    print("COST ESTIMATION SUMMARY")
    print("="*60)
    print(f"Judge Model: {cost_data['judge_model']}")
    if cost_data['available_model'] != cost_data['judge_model']:
        print(f"  → Using: {cost_data['available_model']} (closest available)")
    print(f"Evaluation Mode: {cost_data['mode']}")
    print(f"\nConversations: {cost_data['num_conversations']:,}")
    print(f"Tasks per conversation: {cost_data['tasks_per_conversation']}")
    print(f"Total tasks: {cost_data['total_tasks']:,}")
    print(f"\nTokens per task:")
    print(f"  Input: {cost_data['tokens_per_task']['input']:,}")
    print(f"  Output: {cost_data['tokens_per_task']['output']:,}")
    print(f"\nTotal tokens:")
    print(f"  Input: {cost_data['total_tokens']['input']:,} ({cost_data['total_tokens']['input']/1_000_000:.2f}M)")
    print(f"  Output: {cost_data['total_tokens']['output']:,} ({cost_data['total_tokens']['output']/1_000_000:.2f}M)")
    print(f"  Total: {cost_data['total_tokens']['total']:,} ({cost_data['total_tokens']['total']/1_000_000:.2f}M)")
    print(f"\nPricing (per 1M tokens):")
    print(f"  Input: ${cost_data['pricing_per_1M']['input']:.2f}")
    print(f"  Output: ${cost_data['pricing_per_1M']['output']:.2f}")
    print(f"\nESTIMATED COSTS:")
    print(f"  Input: ${cost_data['costs']['input']:.2f}")
    print(f"  Output: ${cost_data['costs']['output']:.2f}")
    print(f"  {'='*40}")
    print(f"  TOTAL: ${cost_data['costs']['total']:.2f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate API costs for judge evaluations")
    parser.add_argument('--evaluation-set', type=str, help='Path to evaluation set JSON file')
    parser.add_argument('--judge-model', type=str, required=True,
                        help='Judge model name (can be retired model, will map to available)')
    parser.add_argument('--mode', choices=['pairwise', 'absolute', 'both', 'dimension_pairwise', 'all'], 
                        default='pairwise',
                        help='Evaluation mode: pairwise, absolute, both, dimension_pairwise, or all')
    parser.add_argument('--conversations', type=int, help='Number of conversations (if not using --evaluation-set)')
    parser.add_argument('--avg-prompt-tokens', type=int, help='Average prompt tokens (overrides estimation)')
    parser.add_argument('--avg-response-tokens', type=int, help='Average response tokens (overrides estimation)')
    
    args = parser.parse_args()
    
    if args.evaluation_set:
        cost_data = calculate_from_evaluation_set(
            evaluation_set_file=args.evaluation_set,
            judge_model=args.judge_model,
            mode=args.mode
        )
    elif args.conversations:
        cost_data = calculate_evaluation_cost(
            num_conversations=args.conversations,
            mode=args.mode,
            judge_model=args.judge_model,
            avg_prompt_tokens=args.avg_prompt_tokens,
            avg_response_tokens=args.avg_response_tokens
        )
    else:
        parser.error("Either --evaluation-set or --conversations must be provided")
    
    print_cost_summary(cost_data)
    print("\nJSON output:")
    print(json.dumps(cost_data, indent=2))


if __name__ == "__main__":
    main()


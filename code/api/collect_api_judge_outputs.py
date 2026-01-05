#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import uuid
from datetime import datetime
import os

try:
    from dotenv import load_dotenv
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    # Also try parent directory
    parent_env = script_dir.parent / '.env'
    if parent_env.exists():
        load_dotenv(parent_env)
except ImportError:
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)
    parent_env = script_dir.parent / '.env'
    if parent_env.exists():
        with open(parent_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)

sys.path.insert(0, str(Path(__file__).parent.parent / 'cluster'))
from judge_prompts import (
    create_pairwise_prompt,
    create_absolute_scoring_prompt,
    create_dimension_pairwise_prompt,
    create_optimized_pairwise_prompt,
    get_all_dimension_keys,
    EVALUATION_DIMENSIONS
)

try:
    import openai
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Install with: pip install anthropic")
    sys.exit(1)


def get_api_client(judge_model):
    if judge_model.startswith('gpt-') or judge_model.startswith('o1-'):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY environment variable not set for {judge_model}")
        try:
            import httpx
            http_client = httpx.Client(timeout=60.0)
            return 'openai', openai.OpenAI(api_key=api_key, http_client=http_client)
        except Exception:
            return 'openai', openai.OpenAI(api_key=api_key)
    elif judge_model.startswith('claude-'):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(f"ANTHROPIC_API_KEY environment variable not set for {judge_model}")
        return 'anthropic', anthropic.Anthropic(api_key=api_key)
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                import httpx
                http_client = httpx.Client(timeout=60.0)
                return 'openai', openai.OpenAI(api_key=api_key, http_client=http_client)
            except Exception:
                return 'openai', openai.OpenAI(api_key=api_key)
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            return 'anthropic', anthropic.Anthropic(api_key=api_key)
        raise ValueError(f"Could not determine API client for model '{judge_model}'. "
                        f"Model should start with 'gpt-' or 'o1-' for OpenAI, or 'claude-' for Anthropic. "
                        f"Or set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")


class APIJudgeOutputCollector:
    def __init__(self, judge_model, temperature=0.0, top_p=None, max_tokens=1024, mode='pairwise'):
        self.judge_model = judge_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.mode = mode  # 'pairwise', 'absolute', 'both', 'dimension_pairwise', 'all', or 'optimized'
        self.api_type, self.client = get_api_client(judge_model)
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        
    def get_judge_output(self, judge_prompt, retries=3):
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                if self.api_type == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "user", "content": judge_prompt}
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p if self.top_p is not None else None,
                        max_tokens=self.max_tokens
                    )
                    judge_output = response.choices[0].message.content.strip()
                    
                elif self.api_type == 'anthropic':
                    response = self.client.messages.create(
                        model=self.judge_model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p if self.top_p is not None else None,
                        messages=[
                            {"role": "user", "content": judge_prompt}
                        ]
                    )
                    judge_output = response.content[0].text.strip()
                
                return {
                    'judge_output': judge_output,
                    'inference_time_seconds': time.time() - start_time,
                    'attempt': attempt + 1,
                    'status': 'success'
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {
                        'judge_output': None,
                        'inference_time_seconds': None,
                        'attempt': attempt + 1,
                        'status': 'error',
                        'error': str(e)
                    }
                time.sleep(2 ** attempt)
    
    def load_progress_state(self, state_file):
        state_path = Path(state_file)
        if state_path.exists():
            with open(state_path, 'r') as f:
                data = json.load(f)
                if 'completed_task_keys' in data:
                    data['completed_task_keys'] = set(data['completed_task_keys'])
                if 'completed_ids' in data and 'completed_task_keys' not in data:
                    data['completed_task_keys'] = set(data['completed_ids'])
                return data
        return {'completed_evaluations': [], 'completed_task_keys': set(), 'last_saved': None, 'session_info': {}}

    def save_progress_state(self, state, state_file):
        state_to_save = state.copy()
        state_to_save['completed_task_keys'] = list(state['completed_task_keys'])
        state_to_save['last_saved'] = datetime.now().isoformat()
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2)
    
    def get_task_key(self, conversation_id, evaluation_type, dimension=None, model=None, flipped=False):
        flip_suffix = "_flipped" if flipped else ""
        if evaluation_type == 'pairwise':
            return f"{conversation_id}_pairwise{flip_suffix}"
        elif evaluation_type == 'absolute':
            return f"{conversation_id}_absolute_{dimension}_{model}"
        elif evaluation_type == 'dimension_pairwise':
            return f"{conversation_id}_dimension_pairwise_{dimension}{flip_suffix}"
        else:
            raise ValueError(f"Unknown evaluation_type: {evaluation_type}")

    def evaluate_pairwise(self, eval_item, flip_responses=False):
        judge_prompt = create_pairwise_prompt(
            eval_item['prompt'],
            eval_item['response_a'],
            eval_item['response_b'],
            flip_responses=flip_responses
        )
        judge_result = self.get_judge_output(judge_prompt)
        
        return {
            'conversation_id': eval_item['conversation_id'],
            'evaluation_type': 'pairwise',
            'judge_model': self.judge_model,
            'target_models': eval_item.get('target_models', []),
            'model_a': eval_item['model_a'],
            'model_b': eval_item['model_b'],
            'human_winner': eval_item['human_winner'],
            'is_flipped': eval_item.get('is_flipped', False),
            'judge_prompt_flipped': flip_responses,  # New field to track if prompt was flipped
            'judge_output': judge_result['judge_output'],
            'inference_time_seconds': judge_result['inference_time_seconds'],
            'status': judge_result['status'],
            'error': judge_result.get('error'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def evaluate_absolute(self, eval_item, dimension_key, model_key, response):
        model_name = eval_item[model_key]
        judge_prompt = create_absolute_scoring_prompt(
            eval_item['prompt'],
            response,
            dimension_key,
            model_name
        )
        dimension_info = EVALUATION_DIMENSIONS[dimension_key]
        judge_result = self.get_judge_output(judge_prompt)
        
        return {
            'conversation_id': eval_item['conversation_id'],
            'evaluation_type': 'absolute',
            'dimension': dimension_key,
            'dimension_name': dimension_info['name'],
            'dimension_scale': dimension_info['scale'],
            'judge_model': self.judge_model,
            'target_models': eval_item.get('target_models', []),
            'model_a': eval_item['model_a'],
            'model_b': eval_item['model_b'],
            'evaluated_model': model_key,
            'evaluated_model_name': model_name,
            'human_winner': eval_item['human_winner'],
            'is_flipped': eval_item.get('is_flipped', False),
            'judge_output': judge_result['judge_output'],
            'inference_time_seconds': judge_result['inference_time_seconds'],
            'status': judge_result['status'],
            'error': judge_result.get('error'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def evaluate_dimension_pairwise(self, eval_item, dimension_key, flip_responses=False):
        judge_prompt = create_dimension_pairwise_prompt(
            eval_item['prompt'],
            eval_item['response_a'],
            eval_item['response_b'],
            dimension_key,
            flip_responses=flip_responses
        )
        dimension_info = EVALUATION_DIMENSIONS[dimension_key]
        judge_result = self.get_judge_output(judge_prompt)
        
        return {
            'conversation_id': eval_item['conversation_id'],
            'evaluation_type': 'dimension_pairwise',
            'dimension': dimension_key,
            'dimension_name': dimension_info['name'],
            'dimension_scale': dimension_info['scale'],
            'judge_model': self.judge_model,
            'target_models': eval_item.get('target_models', []),
            'model_a': eval_item['model_a'],
            'model_b': eval_item['model_b'],
            'human_winner': eval_item['human_winner'],
            'is_flipped': eval_item.get('is_flipped', False),
            'judge_prompt_flipped': flip_responses,
            'judge_output': judge_result['judge_output'],
            'inference_time_seconds': judge_result['inference_time_seconds'],
            'status': judge_result['status'],
            'error': judge_result.get('error'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def evaluate_optimized_pairwise(self, eval_item, flip_responses=False):
        judge_prompt = create_optimized_pairwise_prompt(
            eval_item['prompt'],
            eval_item['response_a'],
            eval_item['response_b'],
            flip_responses=flip_responses
        )
        judge_result = self.get_judge_output(judge_prompt)
        
        return {
            'conversation_id': eval_item['conversation_id'],
            'evaluation_type': 'optimized_pairwise',
            'judge_model': self.judge_model,
            'target_models': eval_item.get('target_models', []),
            'model_a': eval_item['model_a'],
            'model_b': eval_item['model_b'],
            'human_winner': eval_item['human_winner'],
            'is_flipped': eval_item.get('is_flipped', False),
            'judge_prompt_flipped': flip_responses,
            'judge_output': judge_result['judge_output'],
            'inference_time_seconds': judge_result['inference_time_seconds'],
            'status': judge_result['status'],
            'error': judge_result.get('error'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def evaluate_conversation_complete(self, eval_item, completed_task_keys):
        results = []
        dimension_keys = get_all_dimension_keys()
        
        if self.mode == 'optimized':
            # Original order (A/B)
            task_key = self.get_task_key(eval_item['conversation_id'], 'pairwise', flipped=False)
            if task_key not in completed_task_keys:
                print(f"  Evaluating optimized pairwise (A/B) for {eval_item['conversation_id']}")
                result = self.evaluate_optimized_pairwise(eval_item, flip_responses=False)
                results.append(result)
                completed_task_keys.add(task_key)
            
            # Flipped order (B/A)
            task_key_flipped = self.get_task_key(eval_item['conversation_id'], 'pairwise', flipped=True)
            if task_key_flipped not in completed_task_keys:
                print(f"  Evaluating optimized pairwise (B/A) for {eval_item['conversation_id']}")
                result_flipped = self.evaluate_optimized_pairwise(eval_item, flip_responses=True)
                results.append(result_flipped)
                completed_task_keys.add(task_key_flipped)
            
            return results, completed_task_keys
        
        if self.mode in ['pairwise', 'both', 'all']:
            # Original order
            task_key = self.get_task_key(eval_item['conversation_id'], 'pairwise', flipped=False)
            if task_key not in completed_task_keys:
                print(f"  Evaluating pairwise (original) for {eval_item['conversation_id']}")
                result = self.evaluate_pairwise(eval_item, flip_responses=False)
                results.append(result)
                completed_task_keys.add(task_key)
            
            # Flipped order
            task_key_flipped = self.get_task_key(eval_item['conversation_id'], 'pairwise', flipped=True)
            if task_key_flipped not in completed_task_keys:
                print(f"  Evaluating pairwise (flipped) for {eval_item['conversation_id']}")
                result_flipped = self.evaluate_pairwise(eval_item, flip_responses=True)
                results.append(result_flipped)
                completed_task_keys.add(task_key_flipped)
        
        if self.mode in ['absolute', 'both', 'all']:
            for dimension_key in dimension_keys:
                task_key_a = self.get_task_key(
                    eval_item['conversation_id'], 'absolute', 
                    dimension_key, 'model_a'
                )
                if task_key_a not in completed_task_keys:
                    print(f"  Evaluating {dimension_key} for {eval_item['model_a']} ({eval_item['conversation_id']})")
                    result_a = self.evaluate_absolute(
                        eval_item, dimension_key, 'model_a', eval_item['response_a']
                    )
                    results.append(result_a)
                    completed_task_keys.add(task_key_a)
                
                task_key_b = self.get_task_key(
                    eval_item['conversation_id'], 'absolute',
                    dimension_key, 'model_b'
                )
                if task_key_b not in completed_task_keys:
                    print(f"  Evaluating {dimension_key} for {eval_item['model_b']} ({eval_item['conversation_id']})")
                    result_b = self.evaluate_absolute(
                        eval_item, dimension_key, 'model_b', eval_item['response_b']
                    )
                    results.append(result_b)
                    completed_task_keys.add(task_key_b)
        
        if self.mode in ['dimension_pairwise', 'all']:
            for dimension_key in dimension_keys:
                # Original order
                task_key = self.get_task_key(
                    eval_item['conversation_id'], 'dimension_pairwise', dimension_key, flipped=False
                )
                if task_key not in completed_task_keys:
                    print(f"  Evaluating {dimension_key} pairwise (original) for {eval_item['conversation_id']}")
                    result = self.evaluate_dimension_pairwise(eval_item, dimension_key, flip_responses=False)
                    results.append(result)
                    completed_task_keys.add(task_key)
                
                # Flipped order
                task_key_flipped = self.get_task_key(
                    eval_item['conversation_id'], 'dimension_pairwise', dimension_key, flipped=True
                )
                if task_key_flipped not in completed_task_keys:
                    print(f"  Evaluating {dimension_key} pairwise (flipped) for {eval_item['conversation_id']}")
                    result_flipped = self.evaluate_dimension_pairwise(eval_item, dimension_key, flip_responses=True)
                    results.append(result_flipped)
                    completed_task_keys.add(task_key_flipped)
        
        return results, completed_task_keys

    def collect_outputs(self, evaluation_set_file, output_file, save_interval=10):
        with open(evaluation_set_file, 'r') as f:
            eval_data = json.load(f)
        evaluations = eval_data['evaluations']
        metadata = eval_data.get('metadata', {})
        output_path = Path(output_file)
        state_file = output_path.with_suffix('.state.json')
        state = self.load_progress_state(state_file)
        completed_task_keys = state.get('completed_task_keys', set())
        
        dimension_count = len(get_all_dimension_keys())
        if self.mode == 'pairwise':
            tasks_per_conv = 2  # original + flipped
        elif self.mode == 'optimized':
            tasks_per_conv = 2  # A/B + B/A with optimized prompt
        elif self.mode == 'absolute':
            tasks_per_conv = dimension_count * 2  # dimensions * 2 models (no flipping for absolute)
        elif self.mode == 'dimension_pairwise':
            tasks_per_conv = dimension_count * 2  # 2 per dimension (original + flipped)
        elif self.mode == 'both':
            tasks_per_conv = 2 + dimension_count * 2  # 2 pairwise (original + flipped) + dimensions * 2 models
        else:  # all
            tasks_per_conv = 2 + dimension_count * 2 + dimension_count * 2  # 2 pairwise + dimensions * 2 absolute + dimensions * 2 pairwise
        
        results = state.get('completed_evaluations', []).copy()
        total_tasks = len(evaluations) * tasks_per_conv
        completed_tasks = len(completed_task_keys)
        
        print(f"Mode: {self.mode}")
        print(f"Tasks per conversation: {tasks_per_conv}")
        print(f"Total conversations: {len(evaluations)}")
        print(f"Total tasks: {total_tasks}")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Remaining tasks: {total_tasks - completed_tasks}")
        
        for i, eval_item in enumerate(evaluations):
            conv_id = eval_item['conversation_id']
            
            # Check if all tasks for this conversation are completed
            dimension_keys = get_all_dimension_keys()
            if self.mode == 'pairwise' or self.mode == 'optimized':
                task_key = self.get_task_key(conv_id, 'pairwise', flipped=False)
                task_key_flipped = self.get_task_key(conv_id, 'pairwise', flipped=True)
                if task_key in completed_task_keys and task_key_flipped in completed_task_keys:
                    continue
            elif self.mode == 'absolute':
                all_completed = True
                for dim in dimension_keys:
                    if (self.get_task_key(conv_id, 'absolute', dim, 'model_a') not in completed_task_keys or
                        self.get_task_key(conv_id, 'absolute', dim, 'model_b') not in completed_task_keys):
                        all_completed = False
                        break
                if all_completed:
                    continue
            elif self.mode == 'dimension_pairwise':
                all_completed = True
                for dim in dimension_keys:
                    if (self.get_task_key(conv_id, 'dimension_pairwise', dim, flipped=False) not in completed_task_keys or
                        self.get_task_key(conv_id, 'dimension_pairwise', dim, flipped=True) not in completed_task_keys):
                        all_completed = False
                        break
                if all_completed:
                    continue
            elif self.mode == 'both':
                pairwise_done = (self.get_task_key(conv_id, 'pairwise', flipped=False) in completed_task_keys and
                                self.get_task_key(conv_id, 'pairwise', flipped=True) in completed_task_keys)
                absolute_done = True
                for dim in dimension_keys:
                    if (self.get_task_key(conv_id, 'absolute', dim, 'model_a') not in completed_task_keys or
                        self.get_task_key(conv_id, 'absolute', dim, 'model_b') not in completed_task_keys):
                        absolute_done = False
                        break
                if pairwise_done and absolute_done:
                    continue
            else:  # all
                pairwise_done = (self.get_task_key(conv_id, 'pairwise', flipped=False) in completed_task_keys and
                                self.get_task_key(conv_id, 'pairwise', flipped=True) in completed_task_keys)
                absolute_done = True
                for dim in dimension_keys:
                    if (self.get_task_key(conv_id, 'absolute', dim, 'model_a') not in completed_task_keys or
                        self.get_task_key(conv_id, 'absolute', dim, 'model_b') not in completed_task_keys):
                        absolute_done = False
                        break
                dimension_pairwise_done = True
                for dim in dimension_keys:
                    if (self.get_task_key(conv_id, 'dimension_pairwise', dim, flipped=False) not in completed_task_keys or
                        self.get_task_key(conv_id, 'dimension_pairwise', dim, flipped=True) not in completed_task_keys):
                        dimension_pairwise_done = False
                        break
                if pairwise_done and absolute_done and dimension_pairwise_done:
                    continue
            
            print(f"\nProcessing conversation {i+1}/{len(evaluations)}: {conv_id}")
            conv_results, completed_task_keys = self.evaluate_conversation_complete(
                eval_item, completed_task_keys
            )
            results.extend(conv_results)
            completed_tasks = len(completed_task_keys)
            
            should_save = (
                len(conv_results) > 0 or
                completed_tasks % save_interval == 0
            )
            if should_save:
                print(f"Progress: {completed_tasks}/{total_tasks} tasks completed (saving...)")
                state['completed_evaluations'] = results
                state['completed_task_keys'] = completed_task_keys
                state['session_info'] = {
                    'session_id': self.session_id,
                    'start_time': self.start_time.isoformat(),
                    'judge_model': self.judge_model,
                    'mode': self.mode,
                    'completed_tasks': completed_tasks,
                    'total_tasks': total_tasks,
                    'completed_conversations': len(set(r['conversation_id'] for r in results))
                }
                self.save_progress_state(state, state_file)
                self.save_final_results(results, metadata, output_path)
        
        state['completed_evaluations'] = results
        state['completed_task_keys'] = completed_task_keys
        state['session_info']['completed'] = True
        state['session_info']['end_time'] = datetime.now().isoformat()
        self.save_progress_state(state, state_file)
        self.save_final_results(results, metadata, output_path)
        print(f"\nDone: {completed_tasks} tasks completed, saved to {output_path}")

    def save_final_results(self, results, metadata, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        times = [r['inference_time_seconds'] for r in results if r['inference_time_seconds']]
        avg_time = sum(times) / len(times) if times else None
        
        pairwise_results = [r for r in results if r.get('evaluation_type') == 'pairwise']
        optimized_pairwise_results = [r for r in results if r.get('evaluation_type') == 'optimized_pairwise']
        absolute_results = [r for r in results if r.get('evaluation_type') == 'absolute']
        dimension_pairwise_results = [r for r in results if r.get('evaluation_type') == 'dimension_pairwise']
        
        sampling_config = {'temperature': self.temperature}
        if self.top_p is not None:
            sampling_config['top_p'] = self.top_p
        sampling_config['max_tokens'] = self.max_tokens
        
        final_data = {
            'metadata': {
                **metadata,
                'judge_model': self.judge_model,
                'api_type': self.api_type,
                'evaluation_mode': self.mode,
                'sampling_config': sampling_config,
                'collection_session': self.session_id,
                'collection_start': self.start_time.isoformat(),
                'collection_end': datetime.now().isoformat(),
                'total_evaluations': len(results),
                'pairwise_evaluations': len(pairwise_results),
                'optimized_pairwise_evaluations': len(optimized_pairwise_results),
                'absolute_evaluations': len(absolute_results),
                'dimension_pairwise_evaluations': len(dimension_pairwise_results),
                'successful_evaluations': successful,
                'failed_evaluations': failed,
                'average_inference_time_seconds': avg_time
            },
            'results': results
        }
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect LLM judge outputs using API endpoints")
    parser.add_argument('--judge-model', required=True, 
                        help='Judge model name. OpenAI models (e.g., gpt-3.5-turbo, gpt-4, gpt-3.5-turbo-0125) or Anthropic models (e.g., claude-3-opus, claude-3-sonnet). API client is auto-detected based on model name prefix.')
    parser.add_argument('--evaluation-set', required=True, help='Path to evaluation set JSON file')
    parser.add_argument('--output', required=True, help='Output file for results')
    parser.add_argument('--mode', choices=['pairwise', 'absolute', 'both', 'dimension_pairwise', 'all', 'optimized'], 
                        default='pairwise',
                        help='Evaluation mode: pairwise, absolute, both, dimension_pairwise, all, or optimized (Phase 2 A/B + B/A only with optimized prompt)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save progress every N tasks')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p sampling')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum tokens in response')
    
    args = parser.parse_args()
    
    collector = APIJudgeOutputCollector(
        judge_model=args.judge_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        mode=args.mode
    )
    
    try:
        collector.collect_outputs(
            evaluation_set_file=args.evaluation_set,
            output_file=args.output,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        print("\nInterrupted - progress saved")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



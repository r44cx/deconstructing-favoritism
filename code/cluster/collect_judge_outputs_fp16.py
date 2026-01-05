#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import uuid
from datetime import datetime

# Check for vLLM availability
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Install with: pip install vllm")

from judge_prompts import (
    create_pairwise_prompt,
    create_optimized_pairwise_prompt,
    create_dimension_pairwise_prompt,
    get_all_dimension_keys,
    EVALUATION_DIMENSIONS
)


class FP16JudgeCollector:
    def __init__(self, judge_model, temperature=0.0, max_tokens=1024, 
                 tensor_parallel_size=1, mode='optimized'):
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        
        # Initialize vLLM
        if VLLM_AVAILABLE:
            print(f"Loading {judge_model} with vLLM (FP16)...")
            print(f"Tensor parallel size: {tensor_parallel_size}")
            self.llm = LLM(
                model=judge_model,
                tensor_parallel_size=tensor_parallel_size,
                dtype="float16",  # FP16 precision
                trust_remote_code=True,
                max_model_len=4096,  # Match Llama-2 context
                gpu_memory_utilization=0.95, # Ensure we use available VRAM
            )
            # Create sampling params
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=None, 
            )
            print("Model loaded successfully!")
        else:
            raise RuntimeError("vLLM is required for FP16 inference")
    
    def get_judge_output(self, judge_prompt, retries=3):
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                sys_header = (
                    "<<SYS>>\n"
                    "You are an impartial AI judge. Your task is to evaluate AI assistant responses "
                    "and output your verdict in strict JSON format.\n"
                    "<</SYS>>\n\n"
                )
                
                formatted_prompt = f"<s>[INST] {sys_header}{judge_prompt} [/INST]"
                
                outputs = self.llm.generate([formatted_prompt], self.sampling_params)
                response_text = outputs[0].outputs[0].text.strip()
                
                return {
                    'judge_output': response_text,
                    'inference_time_seconds': time.time() - start_time,
                    'attempt': attempt + 1,
                    'status': 'success',
                    'precision': 'FP16',
                    'backend': 'vLLM'
                }
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    return {
                        'judge_output': None,
                        'inference_time_seconds': None,
                        'attempt': attempt + 1,
                        'status': 'error',
                        'error': str(e),
                        'precision': 'FP16',
                        'backend': 'vLLM'
                    }
                time.sleep(2 ** attempt)
    
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
            'precision': 'FP16',
            'backend': 'vLLM',
            'target_model': eval_item.get('target_model', None),
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
            'precision': 'FP16',
            'backend': 'vLLM',
            'target_model': eval_item.get('target_model', None),
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
            'judge_model': self.judge_model,
            'precision': 'FP16',
            'backend': 'vLLM',
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
    
    def load_progress_state(self, state_file):
        state_path = Path(state_file)
        if state_path.exists():
            with open(state_path, 'r') as f:
                data = json.load(f)
                if 'completed_task_keys' in data:
                    data['completed_task_keys'] = set(data['completed_task_keys'])
                return data
        return {'completed_evaluations': [], 'completed_task_keys': set(), 
                'last_saved': None, 'session_info': {}}
    
    def save_progress_state(self, state, state_file):
        state_to_save = state.copy()
        state_to_save['completed_task_keys'] = list(state['completed_task_keys'])
        state_to_save['last_saved'] = datetime.now().isoformat()
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2)
    
    def get_task_key(self, conversation_id, evaluation_type, dimension=None, flipped=False):
        flip_suffix = "_flipped" if flipped else ""
        if evaluation_type in ['pairwise', 'optimized_pairwise']:
            return f"{conversation_id}_{evaluation_type}{flip_suffix}"
        elif evaluation_type == 'dimension_pairwise':
            return f"{conversation_id}_dimension_pairwise_{dimension}{flip_suffix}"
        else:
            raise ValueError(f"Unknown evaluation_type: {evaluation_type}")
    
    def collect_outputs(self, evaluation_set_file, output_file, save_interval=10):
        with open(evaluation_set_file, 'r') as f:
            eval_data = json.load(f)
        
        evaluations = eval_data['evaluations']
        metadata = eval_data.get('metadata', {})
        
        output_path = Path(output_file)
        state_file = output_path.with_suffix('.state.json')
        state = self.load_progress_state(state_file)
        completed_task_keys = state.get('completed_task_keys', set())
        
        # Calculate tasks per conversation
        if self.mode == 'optimized':
            tasks_per_conv = 2  # A/B + B/A
        elif self.mode == 'pairwise':
            tasks_per_conv = 2  # original + flipped
        elif self.mode == 'dimension_pairwise':
            tasks_per_conv = len(get_all_dimension_keys()) * 2
        elif self.mode == 'all':
            tasks_per_conv = 2 + len(get_all_dimension_keys()) * 2
        else:
            tasks_per_conv = 2
        
        results = state.get('completed_evaluations', []).copy()
        total_tasks = len(evaluations) * tasks_per_conv
        completed_tasks = len(completed_task_keys)
        
        print(f"\n{'='*60}")
        print(f"FP16 Precision Ablation Study")
        print(f"{'='*60}")
        print(f"Model: {self.judge_model}")
        print(f"Precision: FP16")
        print(f"Backend: vLLM")
        print(f"Mode: {self.mode}")
        print(f"Tasks per conversation: {tasks_per_conv}")
        print(f"Total conversations: {len(evaluations)}")
        print(f"Total tasks: {total_tasks}")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Remaining tasks: {total_tasks - completed_tasks}")
        print(f"{'='*60}\n")
        
        for i, eval_item in enumerate(evaluations):
            conv_id = eval_item['conversation_id']
            
            # Check if already complete
            if self.mode == 'optimized':
                task_key = self.get_task_key(conv_id, 'optimized_pairwise', flipped=False)
                task_key_flipped = self.get_task_key(conv_id, 'optimized_pairwise', flipped=True)
                if task_key in completed_task_keys and task_key_flipped in completed_task_keys:
                    continue
            
            print(f"\nProcessing conversation {i+1}/{len(evaluations)}: {conv_id}")
            
            if self.mode == 'optimized':
                # Original order
                task_key = self.get_task_key(conv_id, 'optimized_pairwise', flipped=False)
                if task_key not in completed_task_keys:
                    print(f"  Evaluating optimized pairwise (A/B)...")
                    result = self.evaluate_optimized_pairwise(eval_item, flip_responses=False)
                    results.append(result)
                    completed_task_keys.add(task_key)
                
                # Flipped order
                task_key_flipped = self.get_task_key(conv_id, 'optimized_pairwise', flipped=True)
                if task_key_flipped not in completed_task_keys:
                    print(f"  Evaluating optimized pairwise (B/A)...")
                    result_flipped = self.evaluate_optimized_pairwise(eval_item, flip_responses=True)
                    results.append(result_flipped)
                    completed_task_keys.add(task_key_flipped)
            
            elif self.mode == 'pairwise':
                task_key = self.get_task_key(conv_id, 'pairwise', flipped=False)
                if task_key not in completed_task_keys:
                    print(f"  Evaluating pairwise (A/B)...")
                    result = self.evaluate_pairwise(eval_item, flip_responses=False)
                    results.append(result)
                    completed_task_keys.add(task_key)
                
                task_key_flipped = self.get_task_key(conv_id, 'pairwise', flipped=True)
                if task_key_flipped not in completed_task_keys:
                    print(f"  Evaluating pairwise (B/A)...")
                    result_flipped = self.evaluate_pairwise(eval_item, flip_responses=True)
                    results.append(result_flipped)
                    completed_task_keys.add(task_key_flipped)
            
            elif self.mode == 'dimension_pairwise':
                for dimension_key in get_all_dimension_keys():
                    task_key = self.get_task_key(conv_id, 'dimension_pairwise', dimension_key, flipped=False)
                    if task_key not in completed_task_keys:
                        print(f"  Evaluating {dimension_key} pairwise (A/B)...")
                        result = self.evaluate_dimension_pairwise(eval_item, dimension_key, flip_responses=False)
                        results.append(result)
                        completed_task_keys.add(task_key)
                    
                    task_key_flipped = self.get_task_key(conv_id, 'dimension_pairwise', dimension_key, flipped=True)
                    if task_key_flipped not in completed_task_keys:
                        print(f"  Evaluating {dimension_key} pairwise (B/A)...")
                        result_flipped = self.evaluate_dimension_pairwise(eval_item, dimension_key, flip_responses=True)
                        results.append(result_flipped)
                        completed_task_keys.add(task_key_flipped)
            
            completed_tasks = len(completed_task_keys)
            
            if completed_tasks % save_interval == 0:
                print(f"Progress: {completed_tasks}/{total_tasks} tasks completed")
                state['completed_evaluations'] = results
                state['completed_task_keys'] = completed_task_keys
                state['session_info'] = {
                    'session_id': self.session_id,
                    'start_time': self.start_time.isoformat(),
                    'judge_model': self.judge_model,
                    'precision': 'FP16',
                    'backend': 'vLLM',
                    'mode': self.mode,
                    'completed_tasks': completed_tasks,
                    'total_tasks': total_tasks,
                }
                self.save_progress_state(state, state_file)
                self.save_final_results(results, metadata, output_path)
        
        state['completed_evaluations'] = results
        state['completed_task_keys'] = completed_task_keys
        state['session_info']['completed'] = True
        state['session_info']['end_time'] = datetime.now().isoformat()
        self.save_progress_state(state, state_file)
        self.save_final_results(results, metadata, output_path)
        
        print(f"\n{'='*60}")
        print(f"Collection Complete")
        print(f"{'='*60}")
        print(f"Total tasks: {completed_tasks}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
    
    def save_final_results(self, results, metadata, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        times = [r['inference_time_seconds'] for r in results if r['inference_time_seconds']]
        avg_time = sum(times) / len(times) if times else None
        
        final_data = {
            'metadata': {
                **metadata,
                'judge_model': self.judge_model,
                'precision': 'FP16',
                'backend': 'vLLM',
                'evaluation_mode': self.mode,
                'collection_session': self.session_id,
                'collection_start': self.start_time.isoformat(),
                'collection_end': datetime.now().isoformat(),
                'total_evaluations': len(results),
                'successful_evaluations': successful,
                'failed_evaluations': failed,
                'average_inference_time_seconds': avg_time
            },
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="FP16 Precision Judge Collection")
    parser.add_argument('--judge-model', required=True, help='HuggingFace model ID')
    parser.add_argument('--evaluation-set', required=True, help='Path to evaluation set JSON file')
    parser.add_argument('--output', required=True, help='Output file for results')
    parser.add_argument('--mode', choices=['pairwise', 'optimized', 'dimension_pairwise', 'all'],
                        default='optimized', help='Evaluation mode')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum tokens to generate')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--save-interval', type=int, default=10, help='Save progress every N tasks')
    
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("Error: vLLM is required for FP16 inference.")
        print("Install with: pip install vllm")
        sys.exit(1)
    
    collector = FP16JudgeCollector(
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
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
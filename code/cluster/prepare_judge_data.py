#!/usr/bin/env python3
from pathlib import Path
import json
import argparse
import pandas as pd
import random

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

OLLAMA_JUDGES = [
    'llama-2-70b-chat',     # Meta
    'llama-2-13b-chat',     # Meta
    'mistral-7b-instruct',  # Mistral
    'mixtral-8x7b-instruct-v0.1',  # Mistral
]

API_JUDGES = [
    'gpt-4-0613',           # OpenAI
    'gpt-3.5-turbo-0613',   # OpenAI
]


ALL_TARGET_MODELS = OLLAMA_JUDGES + API_JUDGES


def normalize_model_name(model_name):
    return model_name.replace('/', '_').replace(':', '_').replace(' ', '_').replace('-', '_')


def extract_conversation_data(conv_a, conv_b):
    if hasattr(conv_a, 'tolist'):
        conv_a = conv_a.tolist()
    if hasattr(conv_b, 'tolist'):
        conv_b = conv_b.tolist()
    
    prompt = response_a = response_b = None
    
    for turn in conv_a:
        if turn.get('role') == 'user' and not prompt:
            prompt = turn.get('content', '')
        elif turn.get('role') == 'assistant' and not response_a:
            response_a = turn.get('content', '')
            
    for turn in conv_b:
        if turn.get('role') == 'assistant' and not response_b:
            response_b = turn.get('content', '')
    
    if not prompt or not response_a or not response_b:
        return None
    return {'prompt': prompt, 'response_a': response_a, 'response_b': response_b}


def parse_csv_winner(row):
    try:
        winner_a = int(row.get('winner_model_a', 0)) if row.get('winner_model_a') else 0
        winner_b = int(row.get('winner_model_b', 0)) if row.get('winner_model_b') else 0
        winner_tie = int(row.get('winner_tie', 0)) if row.get('winner_tie') else 0
    except (ValueError, TypeError):
        return None
    
    if winner_a == 1: return 'model_a'
    if winner_b == 1: return 'model_b'
    if winner_tie == 1: return 'tie'
    return None


def split_dataset_by_model_parquet(dataset, target_model, dataset_name="33k", 
                                   randomize_positions=False, random_seed=None, 
                                   flip_positions=False):
    model_conversations = []
    if randomize_positions and random_seed is not None:
        random.seed(random_seed)
    
    for idx, item in enumerate(dataset):
        model_a = item.get('model_a', '')
        model_b = item.get('model_b', '')
        if target_model not in [model_a, model_b]:
            continue
        if 'winner' not in item:
            continue
        if 'conversation_a' not in item or 'conversation_b' not in item:
            continue
        if not item['conversation_a'] or not item['conversation_b']:
            continue
        conv_data = extract_conversation_data(item['conversation_a'], item['conversation_b'])
        if not conv_data:
            continue
        
        is_flipped = (target_model == model_b)
        
        should_swap = False
        if randomize_positions:
            should_swap = random.choice([True, False])
        
        if flip_positions:
            should_swap = not should_swap
        
        if should_swap:
            is_flipped = not is_flipped
        
        if is_flipped:
            evaluation = {
                'conversation_id': f"{dataset_name}_{idx}",
                'prompt': conv_data['prompt'],
                'model_a': model_b,
                'model_b': model_a,
                'response_a': conv_data['response_b'],
                'response_b': conv_data['response_a'],
                'human_winner': 'model_b' if item['winner'] == 'model_a' else ('model_a' if item['winner'] == 'model_b' else item['winner']),
                'target_model': target_model,
                'is_flipped': True
            }
        else:
            evaluation = {
                'conversation_id': f"{dataset_name}_{idx}",
                'prompt': conv_data['prompt'],
                'model_a': model_a,
                'model_b': model_b,
                'response_a': conv_data['response_a'],
                'response_b': conv_data['response_b'],
                'human_winner': item['winner'],
                'target_model': target_model,
                'is_flipped': False
            }
        model_conversations.append(evaluation)
    return model_conversations


def split_dataset_by_model_csv(df, target_model, dataset_name="55k",
                               randomize_positions=False, random_seed=None,
                               flip_positions=False):
    model_conversations = []
    if randomize_positions and random_seed is not None:
        random.seed(random_seed)
    
    for idx, row in df.iterrows():
        model_a = row.get('model_a', '')
        model_b = row.get('model_b', '')
        if target_model not in [model_a, model_b]:
            continue
        winner = parse_csv_winner(row)
        if not winner:
            continue
        if pd.isna(row.get('prompt')) or pd.isna(row.get('response_a')) or pd.isna(row.get('response_b')):
            continue
        prompt = str(row['prompt'])
        response_a = str(row['response_a'])
        response_b = str(row['response_b'])
        if not prompt or not response_a or not response_b:
            continue
        
        is_flipped = (target_model == model_b)
        
        should_swap = False
        if randomize_positions:
            should_swap = random.choice([True, False])
        
        if flip_positions:
            should_swap = not should_swap
        
        if should_swap:
            is_flipped = not is_flipped
        
        if is_flipped:
            evaluation = {
                'conversation_id': f"{dataset_name}_{row.get('id', idx)}",
                'prompt': prompt,
                'model_a': model_b,
                'model_b': model_a,
                'response_a': response_b,
                'response_b': response_a,
                'human_winner': 'model_b' if winner == 'model_a' else ('model_a' if winner == 'model_b' else winner),
                'target_model': target_model,
                'is_flipped': True
            }
        else:
            evaluation = {
                'conversation_id': f"{dataset_name}_{row.get('id', idx)}",
                'prompt': prompt,
                'model_a': model_a,
                'model_b': model_b,
                'response_a': response_a,
                'response_b': response_b,
                'human_winner': winner,
                'target_model': target_model,
                'is_flipped': False
            }
        model_conversations.append(evaluation)
    return model_conversations


def split_dataset_by_models_csv(df, target_models, dataset_name="55k",
                                 randomize_positions=False, random_seed=None,
                                 flip_positions=False):
    model_conversations = []
    seen_pairs = set()
    target_models_set = set(target_models)
    
    if randomize_positions and random_seed is not None:
        random.seed(random_seed)
    
    for idx, row in df.iterrows():
        model_a = row.get('model_a', '')
        model_b = row.get('model_b', '')
        
        if model_a not in target_models_set and model_b not in target_models_set:
            continue
        
        pair_key = tuple(sorted([model_a, model_b]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        winner = parse_csv_winner(row)
        if not winner:
            continue
        if pd.isna(row.get('prompt')) or pd.isna(row.get('response_a')) or pd.isna(row.get('response_b')):
            continue
        prompt = str(row['prompt'])
        response_a = str(row['response_a'])
        response_b = str(row['response_b'])
        if not prompt or not response_a or not response_b:
            continue
        
        should_swap = False
        if randomize_positions:
            should_swap = random.choice([True, False])
        
        if flip_positions:
            should_swap = not should_swap
        
        if should_swap:
            evaluation = {
                'conversation_id': f"{dataset_name}_{row.get('id', idx)}",
                'prompt': prompt,
                'model_a': model_b,
                'model_b': model_a,
                'response_a': response_b,
                'response_b': response_a,
                'human_winner': 'model_b' if winner == 'model_a' else ('model_a' if winner == 'model_b' else winner),
                'target_models': target_models,
                'is_flipped': True
            }
        else:
            evaluation = {
                'conversation_id': f"{dataset_name}_{row.get('id', idx)}",
                'prompt': prompt,
                'model_a': model_a,
                'model_b': model_b,
                'response_a': response_a,
                'response_b': response_b,
                'human_winner': winner,
                'target_models': target_models,
                'is_flipped': False
            }
        model_conversations.append(evaluation)
    return model_conversations


def split_dataset_by_models_parquet(dataset, target_models, dataset_name="33k",
                                     randomize_positions=False, random_seed=None,
                                     flip_positions=False):
    model_conversations = []
    seen_pairs = set()
    target_models_set = set(target_models)
    
    if randomize_positions and random_seed is not None:
        random.seed(random_seed)
    
    for idx, item in enumerate(dataset):
        model_a = item.get('model_a', '')
        model_b = item.get('model_b', '')
        
        if model_a not in target_models_set and model_b not in target_models_set:
            continue
        
        pair_key = tuple(sorted([model_a, model_b]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        if 'winner' not in item:
            continue
        if 'conversation_a' not in item or 'conversation_b' not in item:
            continue
        if not item['conversation_a'] or not item['conversation_b']:
            continue
        conv_data = extract_conversation_data(item['conversation_a'], item['conversation_b'])
        if not conv_data:
            continue
        
        should_swap = False
        if randomize_positions:
            should_swap = random.choice([True, False])
        
        if flip_positions:
            should_swap = not should_swap
        
        if should_swap:
            evaluation = {
                'conversation_id': f"{dataset_name}_{idx}",
                'prompt': conv_data['prompt'],
                'model_a': model_b,
                'model_b': model_a,
                'response_a': conv_data['response_b'],
                'response_b': conv_data['response_a'],
                'human_winner': 'model_b' if item['winner'] == 'model_a' else ('model_a' if item['winner'] == 'model_b' else item['winner']),
                'target_models': target_models,
                'is_flipped': True
            }
        else:
            evaluation = {
                'conversation_id': f"{dataset_name}_{idx}",
                'prompt': conv_data['prompt'],
                'model_a': model_a,
                'model_b': model_b,
                'response_a': conv_data['response_a'],
                'response_b': conv_data['response_b'],
                'human_winner': item['winner'],
                'target_models': target_models,
                'is_flipped': False
            }
        model_conversations.append(evaluation)
    return model_conversations


def save_evaluation_set(evaluations, output_path, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {'metadata': metadata, 'evaluations': evaluations, 'count': len(evaluations)}
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    return len(evaluations)


def main():
    parser = argparse.ArgumentParser(description="Prepare judge evaluation datasets")
    parser.add_argument('--models', type=str, help='Comma-separated list of target models')
    parser.add_argument('--dataset', type=str, default='33k', help='Dataset: 33k or 55k')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory (default: root-level data/judge_eval)')
    parser.add_argument('--randomize-positions', action='store_true',
                       help='Randomly assign models to positions A/B')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for position randomization (default: 42)')
    parser.add_argument('--flip-positions', action='store_true',
                       help='Swap model_a and model_b positions')
    parser.add_argument('--pair-mode', action='store_true',
                       help='Enable model-pair mode (instead of model-vs-all)')
    parser.add_argument('--model-pairs', type=str,
                       help='Comma-separated pairs like "modelA:modelB,modelC:modelD"')
    parser.add_argument('--models-vs-all', action='store_true',
                       help='Enable models-vs-all mode: create single dataset with all specified models (avoids duplicate pairs)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        base_dir = Path(__file__).parent.parent.parent  # Go to repository root
        output_dir = base_dir / 'data' / 'judge_eval'
    else:
        output_dir = Path(args.output_dir)
    
    if args.pair_mode and not args.model_pairs:
        print("Error: --model-pairs required when using --pair-mode")
        return
    
    if args.randomize_positions:
        position_strategy = 'randomized'
    elif args.flip_positions:
        position_strategy = 'flipped'
    else:
        position_strategy = 'fixed'
    
    base_dir = Path(__file__).parent.parent.parent
    dataset = None
    df = None
    is_csv = False
    
    if args.dataset == '33k':
        if not HAS_DATASETS:
            print("Error: 'datasets' package required for 33k dataset. Install with: pip install datasets")
            return
        data_file = base_dir / "data/lmsys_chatbot_arena_conversations_data_train-00000-of-00001-cced8514c7ed782a.parquet"
        if not data_file.exists():
            print(f"Error: {data_file} not found")
            return
        dataset = load_dataset("parquet", data_files=str(data_file), split="train")
        is_csv = False
    elif args.dataset == '55k':
        filtered_file = base_dir / "data/lmarena-ai_55k_filtered_subset.csv"
        csv_file = base_dir / "data/marena-ai_arena-human-preference-55k_train.csv"
        
        if filtered_file.exists():
            print(f"Loading filtered subset CSV: {filtered_file}")
            df = pd.read_csv(filtered_file)
        elif csv_file.exists():
            print(f"Loading full local CSV: {csv_file}")
            df = pd.read_csv(csv_file)
        else:
            if not HAS_DATASETS:
                print(f"Error: 'datasets' package required to download 55k dataset. Install with: pip install datasets")
                return
            print(f"Local file {csv_file} not found. Attempting download from Hugging Face (lmarena-ai/arena-human-preference-55k)...")
            try:
                ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
                df = ds.to_pandas()
                print("Successfully loaded dataset from Hugging Face")
            except Exception as e:
                print(f"Error loading from Hugging Face: {e}")
                return
        is_csv = True
    else:
        print(f"Error: Unknown dataset '{args.dataset}'")
        return
    
    model_counts = {}
    if is_csv:
        for _, row in df.iterrows():
            for col in ['model_a', 'model_b']:
                m = row.get(col, '')
                if m:
                    model_counts[m] = model_counts.get(m, 0) + 1
    else:
        for item in dataset:
            for key in ['model_a', 'model_b']:
                m = item.get(key, '')
                if m:
                    model_counts[m] = model_counts.get(m, 0) + 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    total_created = 0
    total_evaluations = 0
    
    if args.pair_mode:
        pairs = []
        for pair_str in args.model_pairs.split(','):
            pair_str = pair_str.strip()
            if ':' not in pair_str:
                print(f"Warning: Invalid pair format '{pair_str}', skipping")
                continue
            model_a, model_b = pair_str.split(':', 1)
            pairs.append((model_a.strip(), model_b.strip()))
        
        if not pairs:
            print("Error: No valid model pairs provided")
            return
        
        for model_a, model_b in pairs:
            if model_a not in model_counts or model_b not in model_counts:
                print(f"Skipped pair {model_a}:{model_b}: one or both models not in dataset")
                continue
            
            pair_evaluations = []
            if is_csv:
                for idx, row in df.iterrows():
                    row_model_a = row.get('model_a', '')
                    row_model_b = row.get('model_b', '')
                    if not ((row_model_a == model_a and row_model_b == model_b) or
                           (row_model_a == model_b and row_model_b == model_a)):
                        continue
                    winner = parse_csv_winner(row)
                    if not winner:
                        continue
                    if pd.isna(row.get('prompt')) or pd.isna(row.get('response_a')) or pd.isna(row.get('response_b')):
                        continue
                    prompt = str(row['prompt'])
                    response_a = str(row['response_a'])
                    response_b = str(row['response_b'])
                    if not prompt or not response_a or not response_b:
                        continue
                    
                    needs_swap = (row_model_a == model_b)
                    
                    should_swap = False
                    if args.randomize_positions:
                        random.seed(args.random_seed + idx)  # Use idx to vary seed per conversation
                        should_swap = random.choice([True, False])
                    if args.flip_positions:
                        should_swap = not should_swap
                    if needs_swap:
                        should_swap = not should_swap
                    
                    if should_swap:
                        evaluation = {
                            'conversation_id': f"{args.dataset}_{row.get('id', idx)}",
                            'prompt': prompt,
                            'model_a': model_b,
                            'model_b': model_a,
                            'response_a': response_b,
                            'response_b': response_a,
                            'human_winner': 'model_b' if winner == 'model_a' else ('model_a' if winner == 'model_b' else winner),
                            'target_model': model_a,
                            'is_flipped': True
                        }
                    else:
                        evaluation = {
                            'conversation_id': f"{args.dataset}_{row.get('id', idx)}",
                            'prompt': prompt,
                            'model_a': model_a,
                            'model_b': model_b,
                            'response_a': response_a,
                            'response_b': response_b,
                            'human_winner': winner,
                            'target_model': model_a,
                            'is_flipped': False
                        }
                    pair_evaluations.append(evaluation)
            else:
                for idx, item in enumerate(dataset):
                    item_model_a = item.get('model_a', '')
                    item_model_b = item.get('model_b', '')
                    if not ((item_model_a == model_a and item_model_b == model_b) or
                           (item_model_a == model_b and item_model_b == model_a)):
                        continue
                    if 'winner' not in item:
                        continue
                    if 'conversation_a' not in item or 'conversation_b' not in item:
                        continue
                    if not item['conversation_a'] or not item['conversation_b']:
                        continue
                    conv_data = extract_conversation_data(item['conversation_a'], item['conversation_b'])
                    if not conv_data:
                        continue
                    
                    needs_swap = (item_model_a == model_b)
                    
                    should_swap = False
                    if args.randomize_positions:
                        random.seed(args.random_seed + idx)
                        should_swap = random.choice([True, False])
                    if args.flip_positions:
                        should_swap = not should_swap
                    if needs_swap:
                        should_swap = not should_swap
                    
                    if should_swap:
                        evaluation = {
                            'conversation_id': f"{args.dataset}_{idx}",
                            'prompt': conv_data['prompt'],
                            'model_a': model_b,
                            'model_b': model_a,
                            'response_a': conv_data['response_b'],
                            'response_b': conv_data['response_a'],
                            'human_winner': 'model_b' if item['winner'] == 'model_a' else ('model_a' if item['winner'] == 'model_b' else item['winner']),
                            'target_model': model_a,
                            'is_flipped': True
                        }
                    else:
                        evaluation = {
                            'conversation_id': f"{args.dataset}_{idx}",
                            'prompt': conv_data['prompt'],
                            'model_a': model_a,
                            'model_b': model_b,
                            'response_a': conv_data['response_a'],
                            'response_b': conv_data['response_b'],
                            'human_winner': item['winner'],
                            'target_model': model_a,
                            'is_flipped': False
                        }
                    pair_evaluations.append(evaluation)
            
            if pair_evaluations:
                output_file = output_dir / f"{args.dataset}_{normalize_model_name(model_a)}_{normalize_model_name(model_b)}.json"
                metadata = {
                    'type': 'model_pair',
                    'model_a': model_a,
                    'model_b': model_b,
                    'dataset': args.dataset,
                    'dataset_format': 'csv' if is_csv else 'parquet',
                    'position_strategy': position_strategy,
                    'total_conversations': len(pair_evaluations),
                    'available_judges': len(ALL_TARGET_MODELS),
                    'ollama_judges': len(OLLAMA_JUDGES),
                    'api_judges': len(API_JUDGES)
                }
                if args.randomize_positions:
                    metadata['random_seed'] = args.random_seed
                if args.pair_mode:
                    metadata['model_pairs'] = args.model_pairs
                
                count = save_evaluation_set(pair_evaluations, output_file, metadata)
                print(f"Created {count} evaluations for pair {model_a}:{model_b}")
                total_created += 1
                total_evaluations += count
    
    elif args.models_vs_all:
        if not args.models:
            print("Error: --models required when using --models-vs-all")
            return
        
        target_models = [m.strip() for m in args.models.split(',')]
        missing_models = [m for m in target_models if m not in model_counts]
        if missing_models:
            print(f"Warning: Some models not in dataset: {missing_models}")
            target_models = [m for m in target_models if m in model_counts]
        
        if not target_models:
            print("Error: No valid models found")
            return
        
        if is_csv:
            evaluations = split_dataset_by_models_csv(
                df, target_models, args.dataset,
                randomize_positions=args.randomize_positions,
                random_seed=args.random_seed,
                flip_positions=args.flip_positions
            )
        else:
            evaluations = split_dataset_by_models_parquet(
                dataset, target_models, args.dataset,
                randomize_positions=args.randomize_positions,
                random_seed=args.random_seed,
                flip_positions=args.flip_positions
            )
        
        if evaluations:
            model_names_str = '_'.join([normalize_model_name(m) for m in target_models])
            output_file = output_dir / f"{args.dataset}_{model_names_str}_vs_all.json"
            metadata = {
                'type': 'models_vs_all',
                'target_models': target_models,
                'dataset': args.dataset,
                'dataset_format': 'csv' if is_csv else 'parquet',
                'position_strategy': position_strategy,
                'total_conversations': len(evaluations),
                'available_judges': len(ALL_TARGET_MODELS),
                'ollama_judges': len(OLLAMA_JUDGES),
                'api_judges': len(API_JUDGES)
            }
            if args.randomize_positions:
                metadata['random_seed'] = args.random_seed
            
            count = save_evaluation_set(evaluations, output_file, metadata)
            print(f"Created {count} evaluations for models: {', '.join(target_models)}")
            total_created += 1
            total_evaluations += count
    
    else:
        target_models = [m.strip() for m in args.models.split(',')] if args.models else ALL_TARGET_MODELS
        
        for target_model in target_models:
            if target_model not in model_counts:
                print(f"Skipped {target_model}: not in dataset")
                continue
            
            if is_csv:
                evaluations = split_dataset_by_model_csv(
                    df, target_model, args.dataset,
                    randomize_positions=args.randomize_positions,
                    random_seed=args.random_seed,
                    flip_positions=args.flip_positions
                )
            else:
                evaluations = split_dataset_by_model_parquet(
                    dataset, target_model, args.dataset,
                    randomize_positions=args.randomize_positions,
                    random_seed=args.random_seed,
                    flip_positions=args.flip_positions
                )
            
            if evaluations:
                output_file = output_dir / f"{args.dataset}_{normalize_model_name(target_model)}_vs_all.json"
                metadata = {
                    'type': 'model_vs_all',
                    'target_model': target_model,
                    'dataset': args.dataset,
                    'dataset_format': 'csv' if is_csv else 'parquet',
                    'position_strategy': position_strategy,
                    'total_conversations_with_model': model_counts[target_model],
                    'available_judges': len(ALL_TARGET_MODELS),
                    'ollama_judges': len(OLLAMA_JUDGES),
                    'api_judges': len(API_JUDGES)
                }
                if args.randomize_positions:
                    metadata['random_seed'] = args.random_seed
                
                count = save_evaluation_set(evaluations, output_file, metadata)
                print(f"Created {count} evaluations for {target_model}")
                total_created += 1
                total_evaluations += count
    
    print(f"Done: {total_created} sets, {total_evaluations} evaluations")


if __name__ == "__main__":
    main()
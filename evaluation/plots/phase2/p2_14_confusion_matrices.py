#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from core.data_loader import load_phase2_data, extract_pairwise, PairwiseResults
from core.plot_style import setup_style, save_plot, save_csv
from core.statistical_framework import compute_confusion_matrix_3x3, compute_faviscore
from core.config import FAMILY_MAP, JUDGE_ORDER, MODEL_ORDER, FAMILY_COLORS


def compute_confusion_matrix(judge_ratings, human_ratings):
    matrix = np.zeros((3, 3), dtype=int)
    
    for h, j in zip(human_ratings, judge_ratings):
        h_idx = 1 - h
        j_idx = 1 - j
        matrix[h_idx, j_idx] += 1
    
    return matrix


def create_judge_plot(judge, data, pairs_data):
    setup_style()
    
    n_pairs = len(pairs_data)
    if n_pairs == 0:
        return None
    
    pairs_sorted = sorted(pairs_data.items(), key=lambda x: x[1]['n'], reverse=True)
    n_pairs = len(pairs_sorted)
    
    if n_pairs == 0:
        return None
    
    n_left = (n_pairs + 1) // 2
    n_right = n_pairs - n_left
    n_rows = max(n_left, n_right)
    
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 2.2 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    labels = ['A', 'Tie', 'B']
    
    def plot_pair(row, col_offset, pair, pdata, is_first_in_group, is_last_in_group):
        model_a, model_b = pair
        
        ab_cm = compute_confusion_matrix(pdata['ab'], pdata['human'])
        ba_cm = compute_confusion_matrix(pdata['ba'], pdata['human'])
        agg_cm = compute_confusion_matrix(pdata['agg'], pdata['human'])
        
        matrices = [ab_cm, ba_cm, agg_cm]
        titles = ['A/B Order', 'B/A Order', 'Aggregated']
        
        for col, (cm, title) in enumerate(zip(matrices, titles)):
            ax = axes[row, col_offset + col]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels if col == 0 else False,
                       cbar=False, ax=ax, vmin=0, vmax=cm.max(),
                       annot_kws={'fontsize': 9, 'fontweight': 'bold'})
            
            if is_first_in_group:
                ax.set_title(title, fontsize=10, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{model_a[:12]}\nvs\n{model_b[:12]}', 
                            fontsize=8, fontweight='bold')
            
            if is_last_in_group:
                ax.set_xlabel('Judge', fontsize=9)
            
            favi = compute_faviscore(cm)
            ax.text(0.5, -0.22, f'Φ={favi:.2f}', transform=ax.transAxes,
                   ha='center', fontsize=8, fontweight='bold',
                   color='#e74c3c' if abs(favi) > 0.2 else 'black')
    
    for row, (pair, pdata) in enumerate(pairs_sorted[:n_left]):
        is_first = (row == 0)
        is_last = (row == n_left - 1)
        plot_pair(row, 0, pair, pdata, is_first, is_last)
    
    for row, (pair, pdata) in enumerate(pairs_sorted[n_left:]):
        is_first = (row == 0)
        is_last = (row == n_right - 1)
        plot_pair(row, 3, pair, pdata, is_first, is_last)
    
    for row in range(n_right, n_rows):
        for col in range(3, 6):
            axes[row, col].axis('off')
    
    fig.text(0.02, 0.5, 'Human\n↓', va='center', ha='center', 
            fontsize=11, fontweight='bold', rotation=0)
    fig.text(0.52, 0.5, 'Human\n↓', va='center', ha='center', 
            fontsize=11, fontweight='bold', rotation=0)
    
    judge_family = FAMILY_MAP.get(judge, 'Unknown')
    fig.suptitle(f'Confusion Matrices - {judge} ({judge_family})\n'
                f'Rows: Human verdict | Cols: Judge verdict | Φ = FaviScore',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.06, right=0.98, bottom=0.06, wspace=0.35, hspace=0.45)
    
    return fig


def compute_metrics():
    data = load_phase2_data(quiet=True)
    
    if not data:
        return {}
    
    all_data = {}
    
    for judge in data.keys():
        jdata = data[judge]
        pair_data = extract_pairwise(jdata, None)
        pw = PairwiseResults(pair_data)
        
        pairs_data = defaultdict(lambda: {'ab': [], 'ba': [], 'agg': [], 'human': [], 'n': 0})
        
        for i, pair in enumerate(pw.pairs):
            pairs_data[pair]['ab'].append(pw.ab_ratings[i])
            pairs_data[pair]['ba'].append(pw.ba_ratings[i])
            pairs_data[pair]['agg'].append(pw.agg_ratings[i])
            pairs_data[pair]['human'].append(pw.human_ratings[i])
            pairs_data[pair]['n'] += 1
        
        for pair in pairs_data:
            for key in ['ab', 'ba', 'agg', 'human']:
                pairs_data[pair][key] = np.array(pairs_data[pair][key])
        
        all_data[judge] = dict(pairs_data)
    
    return all_data


def generate():
    all_data = compute_metrics()
    
    if not all_data:
        return None
    
    results = {}
    
    for judge in JUDGE_ORDER:
        if judge not in all_data:
            continue
        
        pairs_data = all_data[judge]
        fig = create_judge_plot(judge, all_data, pairs_data)
        
        if fig:
            save_plot(fig, f"P2_14_confusion_{judge.replace('-', '_')}")
            plt.close(fig)
            results[judge] = True
    
    return results


if __name__ == "__main__":
    generate()

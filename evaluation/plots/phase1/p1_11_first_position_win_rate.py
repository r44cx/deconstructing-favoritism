#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loader import load_phase1_data, extract_pairwise, PairwiseResults
from core.plot_style import (
    setup_style, save_plot, save_csv,
    get_family_color, add_baseline_line, add_sample_size_annotation
)
from core.statistical_framework import bootstrap_ci
from core.config import FAMILY_MAP, JUDGE_ORDER


def compute_metrics():
    data = load_phase1_data(quiet=True)
    
    if not data:
        return pd.DataFrame()
    
    results = []
    
    for judge in data.keys():
        jdata = data[judge]
        
        pairwise_results = [r for r in jdata['results'] if r['evaluation_type'] == 'pairwise']
        
        if not pairwise_results:
            continue
        
        from core.data_loader import extract_winner
        
        ab_runs = [r for r in pairwise_results if not r.get('judge_prompt_flipped', False)]
        ba_runs = [r for r in pairwise_results if r.get('judge_prompt_flipped', False)]
        
        ab_picks_a = sum(1 for r in ab_runs if extract_winner(r.get('judge_output', '')) == 'A')
        ab_total = len([r for r in ab_runs if extract_winner(r.get('judge_output', '')) in ['A', 'B', 'tie']])
        
        ba_picks_a = sum(1 for r in ba_runs if extract_winner(r.get('judge_output', '')) == 'A')
        ba_total = len([r for r in ba_runs if extract_winner(r.get('judge_output', '')) in ['A', 'B', 'tie']])
        
        if ab_total == 0 or ba_total == 0:
            continue
        
        ab_first_pct = (ab_picks_a / ab_total) * 100
        ba_first_pct = (ba_picks_a / ba_total) * 100
        avg_first_pos = (ab_first_pct + ba_first_pct) / 2
        
        ab_first_binary = np.array([1 if extract_winner(r.get('judge_output', '')) == 'A' else 0 
                                     for r in ab_runs if extract_winner(r.get('judge_output', '')) in ['A', 'B', 'tie']])
        ba_first_binary = np.array([1 if extract_winner(r.get('judge_output', '')) == 'A' else 0 
                                     for r in ba_runs if extract_winner(r.get('judge_output', '')) in ['A', 'B', 'tie']])
        combined_first = np.concatenate([ab_first_binary, ba_first_binary])
        
        mean_first, low_first, high_first = bootstrap_ci(combined_first)
        
        results.append({
            'judge': judge,
            'family': FAMILY_MAP.get(judge, 'Unknown'),
            'first_pos_rate': mean_first * 100,
            'first_pos_ci_lower': low_first * 100,
            'first_pos_ci_upper': high_first * 100,
            'ab_first_pct': ab_first_pct,
            'ba_first_pct': ba_first_pct,
            'n': ab_total
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df = df.sort_values('first_pos_rate', ascending=False)
    
    x = np.arange(len(df))
    colors = [get_family_color(f) for f in df['family']]
    
    bars = ax.bar(x, df['first_pos_rate'], color=colors, edgecolor='black', linewidth=1.5)
    
    yerr_lower = df['first_pos_rate'] - df['first_pos_ci_lower']
    yerr_upper = df['first_pos_ci_upper'] - df['first_pos_rate']
    ax.errorbar(x, df['first_pos_rate'], yerr=[yerr_lower, yerr_upper],
               fmt='none', color='black', capsize=5, linewidth=2)
    
    add_baseline_line(ax, 33.3, label='Random baseline (33.3%)', color='gray', linestyle='--')
    
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        y = row.first_pos_ci_upper
        ax.text(bar.get_x() + bar.get_width()/2, y + 2,
               f'{row.first_pos_rate:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['judge'], rotation=0)
    ax.set_xlabel('Judge Model', fontsize=12)
    ax.set_ylabel('First Position Win Rate (%)', fontsize=12)
    ax.set_title('Naive Positional Bias: First-Position Win Rate\n(Random baseline = 33.3% for ternary task)',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    
    from matplotlib.patches import Patch
    from core.plot_style import create_family_legend
    create_family_legend(ax, loc='upper right')
    
    add_sample_size_annotation(ax, 150, location='bottom_left', approximate=True)
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P1_11_first_position_win_rate")
    save_csv(df, "P1_11_first_position_win_rate")
    
    for _, row in df.iterrows():
        print(f"    {row['judge']}: {row['first_pos_rate']:.1f}% [{row['first_pos_ci_lower']:.1f}, {row['first_pos_ci_upper']:.1f}]")
    
    return df


if __name__ == "__main__":
    generate()

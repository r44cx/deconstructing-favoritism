#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loader import load_phase1_data, extract_pairwise, PairwiseResults
from core.plot_style import setup_style, save_plot, save_csv, FAMILY_COLORS
from core.config import FAMILY_MAP


def compute_metrics():
    data = load_phase1_data(quiet=True)
    
    if not data:
        return pd.DataFrame()
    
    results = []
    
    for judge in data.keys():
        jdata = data[judge]
        pair_data = extract_pairwise(jdata, 'pairwise')
        pw = PairwiseResults(pair_data)
        
        if len(pw.ab_ratings) == 0:
            continue
        
        n = len(pw.ab_ratings)
        
        ab_a = np.sum(pw.ab_ratings == 1) / n * 100
        ab_b = np.sum(pw.ab_ratings == -1) / n * 100
        ab_tie = np.sum(pw.ab_ratings == 0) / n * 100
        
        agg_a = np.sum(pw.agg_ratings == 1) / n * 100
        agg_b = np.sum(pw.agg_ratings == -1) / n * 100
        agg_tie = np.sum(pw.agg_ratings == 0) / n * 100
        
        consistent = np.sum(pw.ab_ratings == pw.ba_ratings) / n * 100
        
        results.append({
            'judge': judge,
            'family': FAMILY_MAP.get(judge, 'Unknown'),
            'ab_a': ab_a,
            'ab_b': ab_b,
            'ab_tie': ab_tie,
            'agg_a': agg_a,
            'agg_b': agg_b,
            'agg_tie': agg_tie,
            'consistency': consistent,
            'ties_generated': agg_tie - ab_tie,
            'n': n
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    ax1 = axes[0]
    bars1_a = ax1.bar(x - width, df['ab_a'], width, label='A Wins', color='#3498db', edgecolor='black')
    bars1_b = ax1.bar(x, df['ab_b'], width, label='B Wins', color='#e74c3c', edgecolor='black')
    bars1_tie = ax1.bar(x + width, df['ab_tie'], width, label='Tie', color='#95a5a6', edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['judge'], fontsize=10)
    ax1.set_xlabel('Judge Model', fontsize=11)
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_title('Single-Direction (A/B) Verdicts', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 75)
    ax1.legend(loc='upper right')
    
    for bars in [bars1_a, bars1_b, bars1_tie]:
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.0f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    bars2_a = ax2.bar(x - width, df['agg_a'], width, label='A Wins', color='#3498db', edgecolor='black')
    bars2_b = ax2.bar(x, df['agg_b'], width, label='B Wins', color='#e74c3c', edgecolor='black')
    bars2_tie = ax2.bar(x + width, df['agg_tie'], width, label='Tie (Aggregated)', color='#f39c12', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['judge'], fontsize=10)
    ax2.set_xlabel('Judge Model', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title('After Bidirectional Aggregation', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 75)
    ax2.legend(loc='upper right')
    
    for bars in [bars2_a, bars2_b, bars2_tie]:
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.0f}%', ha='center', fontsize=9, fontweight='bold')
    
    fig.suptitle('Effect of Bidirectional Aggregation on Verdict Distribution',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P1_XX_aggregation_ties")
    save_csv(df, "P1_XX_aggregation_ties")
    
    for _, row in df.iterrows():
        print(f"    {row['judge']}: AB ties={row['ab_tie']:.1f}% → Agg ties={row['agg_tie']:.1f}% (+{row['ties_generated']:.1f}%)")
    
    return df


if __name__ == "__main__":
    generate()

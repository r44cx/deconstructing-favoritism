#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loader import load_phase1_data, extract_pairwise, PairwiseResults
from core.plot_style import (
    setup_style, save_plot, save_csv, add_baseline_line,
    STRATEGY_COLORS
)
from core.statistical_framework import bootstrap_ci, format_ci
from core.config import RANDOM_BASELINE, FAMILY_MAP


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
        
        ab_agree = (pw.ab_ratings == pw.human_ratings).astype(int)
        ab_val, ab_low, ab_high = bootstrap_ci(ab_agree)
        
        ba_agree = (pw.ba_ratings == pw.human_ratings).astype(int)
        ba_val, ba_low, ba_high = bootstrap_ci(ba_agree)
        
        agg_agree = (pw.agg_ratings == pw.human_ratings).astype(int)
        agg_val, agg_low, agg_high = bootstrap_ci(agg_agree)
        
        results.append({
            'judge': judge,
            'family': FAMILY_MAP.get(judge, 'Unknown'),
            'ab_agreement': ab_val * 100,
            'ab_ci_lower': ab_low * 100,
            'ab_ci_upper': ab_high * 100,
            'ba_agreement': ba_val * 100,
            'ba_ci_lower': ba_low * 100,
            'ba_ci_upper': ba_high * 100,
            'agg_agreement': agg_val * 100,
            'agg_ci_lower': agg_low * 100,
            'agg_ci_upper': agg_high * 100,
            'improvement': (agg_val - max(ab_val, ba_val)) * 100,
            'n': len(pw.ab_ratings)
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars_ab = ax.bar(x - width, df['ab_agreement'], width, 
                     label='A/B Only', color=STRATEGY_COLORS['A/B'], 
                     edgecolor='black', linewidth=1)
    bars_ba = ax.bar(x, df['ba_agreement'], width,
                     label='B/A Only', color=STRATEGY_COLORS['B/A'],
                     edgecolor='black', linewidth=1)
    bars_agg = ax.bar(x + width, df['agg_agreement'], width,
                      label='Aggregated', color=STRATEGY_COLORS['Aggregated'],
                      edgecolor='black', linewidth=1)
    
    ax.errorbar(x - width, df['ab_agreement'],
               yerr=[df['ab_agreement'] - df['ab_ci_lower'], 
                     df['ab_ci_upper'] - df['ab_agreement']],
               fmt='none', color='black', capsize=3)
    ax.errorbar(x, df['ba_agreement'],
               yerr=[df['ba_agreement'] - df['ba_ci_lower'],
                     df['ba_ci_upper'] - df['ba_agreement']],
               fmt='none', color='black', capsize=3)
    ax.errorbar(x + width, df['agg_agreement'],
               yerr=[df['agg_agreement'] - df['agg_ci_lower'],
                     df['agg_ci_upper'] - df['agg_agreement']],
               fmt='none', color='black', capsize=3)
    
    add_baseline_line(ax, RANDOM_BASELINE, label=f'Random ({RANDOM_BASELINE:.1f}%)')
    
    for bars in [bars_ab, bars_ba, bars_agg]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['judge'])
    ax.set_xlabel('Judge Model', fontsize=12)
    ax.set_ylabel('Human Agreement (%)', fontsize=12)
    ax.set_title('Human Agreement by Evaluation Strategy',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P1_02_human_agreement")
    save_csv(df, "P1_02_human_agreement")
    
    for _, row in df.iterrows():
        print(f"    {row['judge']}: AB={row['ab_agreement']:.1f}%, BA={row['ba_agreement']:.1f}%, Agg={row['agg_agreement']:.1f}% (Δ={row['improvement']:+.1f}%)")
    
    return df


if __name__ == "__main__":
    generate()

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
    get_family_color, add_value_labels, add_baseline_line, create_family_legend,
    add_sample_size_annotation
)
from core.statistical_framework import bootstrap_ci, significance_stars, format_ci
from core.config import RANDOM_BASELINE, FAMILY_MAP, JUDGE_ORDER


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
        
        consistent = (pw.ab_ratings == pw.ba_ratings).astype(int)
        cons, cons_low, cons_high = bootstrap_ci(consistent)
        
        inconsistent_mask = pw.ab_ratings != pw.ba_ratings
        if np.any(inconsistent_mask):
            ab_inconsistent = pw.ab_ratings[inconsistent_mask]
            first_pos_pref = np.mean(ab_inconsistent == 1) * 100
        else:
            first_pos_pref = 50.0
        
        results.append({
            'judge': judge,
            'family': FAMILY_MAP.get(judge, 'Unknown'),
            'consistency': cons * 100,
            'consistency_ci_lower': cons_low * 100,
            'consistency_ci_upper': cons_high * 100,
            'first_pos_pref': first_pos_pref,
            'n': len(pw.ab_ratings)
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df = df.sort_values('consistency', ascending=False)
    
    x = np.arange(len(df))
    colors = [get_family_color(f) for f in df['family']]
    
    bars = ax.bar(x, df['consistency'], color=colors, edgecolor='black', linewidth=1.5)
    
    yerr_lower = df['consistency'] - df['consistency_ci_lower']
    yerr_upper = df['consistency_ci_upper'] - df['consistency']
    ax.errorbar(x, df['consistency'], yerr=[yerr_lower, yerr_upper],
               fmt='none', color='black', capsize=5, linewidth=2)
    
    add_baseline_line(ax, RANDOM_BASELINE, label='Random baseline (33.3%)')
    
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        y = row.consistency_ci_upper
        ax.text(bar.get_x() + bar.get_width()/2, y + 2,
               f'{row.consistency:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['judge'], rotation=0)
    ax.set_xlabel('Judge Model', fontsize=12)
    ax.set_ylabel('Flip Consistency (%)', fontsize=12)
    ax.set_title('Flip Consistency Analysis\n(Position Bias Detection)',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    
    create_family_legend(ax, loc='upper right')
    
    add_sample_size_annotation(ax, 150, location='bottom_left', approximate=True)
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P1_01_flip_consistency")
    save_csv(df, "P1_01_flip_consistency")
    
    for _, row in df.iterrows():
        print(f"    {row['judge']}: {row['consistency']:.1f}% [{row['consistency_ci_lower']:.1f}, {row['consistency_ci_upper']:.1f}]")
    
    return df


if __name__ == "__main__":
    generate()

#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from core.data_loader import load_phase1_data, extract_dimension_pairwise, PairwiseResults
from core.plot_style import (
    setup_style, save_plot, save_csv, get_consistency_color,
    add_baseline_line, add_significance_stars
)
from core.statistical_framework import (
    bootstrap_ci, chi_square_test,
    significance_stars, faviscore_with_ci
)
from core.config import (
    RANDOM_BASELINE, DIMENSION_ORDER, DIMENSIONS_INTERNAL,
    RETAINED_DIMENSIONS, EXCLUDED_DIMENSIONS
)


def compute_metrics():
    data = load_phase1_data(quiet=True)
    
    if not data:
        return pd.DataFrame()
    
    results = []
    
    dim_name_map = {
        'logical_robustness': 'Logical Robustness',
        'completeness': 'Completeness',
        'logical_correctness': 'Logical Correctness',
        'helpfulness': 'Helpfulness',
        'faithfulness': 'Faithfulness',
        'conciseness': 'Conciseness',
    }
    
    for judge in data.keys():
        jdata = data[judge]
        
        for dim_internal in DIMENSIONS_INTERNAL:
            dim_display = dim_name_map.get(dim_internal, dim_internal)
            
            try:
                pair_data = extract_dimension_pairwise(jdata, dim_internal)
                pw = PairwiseResults(pair_data)
                
                if len(pw.ab_ratings) < 10:
                    continue
                
                agreement_mask = (pw.human_ratings == pw.agg_ratings).astype(int)
                agreement, agree_low, agree_high = bootstrap_ci(agreement_mask)
                
                favi, favi_low, favi_high, _ = faviscore_with_ci(pw.human_ratings, pw.agg_ratings)
                
                results.append({
                    'dimension': dim_display,
                    'judge': judge,
                    'agreement': agreement * 100,
                    'agreement_ci_lower': agree_low * 100,
                    'agreement_ci_upper': agree_high * 100,
                    'faviscore': favi,
                    'faviscore_ci_lower': favi_low,
                    'faviscore_ci_upper': favi_high,
                    'retained': dim_display in RETAINED_DIMENSIONS,
                    'n': len(pw.ab_ratings)
                })
                
            except Exception as e:
                continue
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    df_agg = df.groupby('dimension').agg({
        'agreement': 'mean',
        'agreement_ci_lower': 'mean',
        'agreement_ci_upper': 'mean',
        'faviscore': 'mean',
        'faviscore_ci_lower': 'mean',
        'faviscore_ci_upper': 'mean',
        'retained': 'first',
        'n': 'sum'
    }).reset_index()
    
    return df_agg


def create_plot(df):
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = df.sort_values('agreement', ascending=True)
    
    y = np.arange(len(df))
    colors = ['#2ecc71' if r else '#e74c3c' for r in df['retained']]
    
    ax1 = axes[0]
    bars = ax1.barh(y, df['agreement'], color=colors, edgecolor='black', height=0.7)
    
    ax1.errorbar(df['agreement'], y,
                xerr=[df['agreement'] - df['agreement_ci_lower'],
                      df['agreement_ci_upper'] - df['agreement']],
                fmt='none', color='black', capsize=3)
    
    ax1.axvline(33.3, color='gray', linestyle='--', linewidth=2, label='Random (33%)')
    
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        ax1.text(row.agreement + 1, bar.get_y() + bar.get_height()/2 + 0.15,
                f'{row.agreement:.1f}%', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(df['dimension'])
    ax1.set_xlabel('Human Agreement (%)', fontsize=11)
    ax1.set_title('Validity\n(Human Agreement)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.legend(loc='lower right')
    
    ax2 = axes[1]
    
    bars2 = ax2.barh(y, df['faviscore'], color=colors, edgecolor='black', height=0.7)
    
    ax2.errorbar(df['faviscore'], y,
                xerr=[df['faviscore'] - df['faviscore_ci_lower'],
                      df['faviscore_ci_upper'] - df['faviscore']],
                fmt='none', color='black', capsize=3)
    
    ax2.axvline(0, color='gray', linestyle='-', linewidth=2)
    
    for i, (bar, row) in enumerate(zip(bars2, df.itertuples())):
        offset = 0.02 if row.faviscore >= 0 else -0.02
        ha = 'left' if row.faviscore >= 0 else 'right'
        ax2.text(row.faviscore + offset, bar.get_y() + bar.get_height()/2 + 0.15,
                f'{row.faviscore:.3f}', va='bottom', ha=ha, fontsize=9, fontweight='bold')
    
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('FaviScore (Φ)', fontsize=11)
    ax2.set_title('Fairness/Bias\n(FaviScore)', fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Retained'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Excluded'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    fig.suptitle('Dimension Analysis: Validity vs. Fairness',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P1_04_dimension_analysis")
    save_csv(df, "P1_04_dimension_analysis")
    
    for _, row in df.iterrows():
        status = "RETAINED" if row['retained'] else "EXCLUDED"
        print(f"    {row['dimension']}: Agree={row['agreement']:.1f}%, Φ={row['faviscore']:.3f} [{status}]")
    
    return df


if __name__ == "__main__":
    generate()

#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loader import load_phase2_data, extract_pairwise, PairwiseResults, compute_faviscore_simple
from core.plot_style import (
    setup_style, save_plot, save_csv, get_family_color
)
from core.statistical_framework import (
    bootstrap_ci_delta, permutation_test,
    significance_stars, format_p_value
)
from core.config import FAMILY_MAP, JUDGE_ORDER


def compute_metrics():
    data = load_phase2_data(quiet=True)
    
    if not data:
        return pd.DataFrame()
    
    results = []
    
    for judge in data.keys():
        jdata = data[judge]
        pair_data = extract_pairwise(jdata, None)
        pw = PairwiseResults(pair_data)
        
        if len(pw.agg_ratings) == 0:
            continue
        
        judge_family = FAMILY_MAP.get(judge, 'Unknown')
        
        intra_family_faviscores = []
        cross_family_faviscores = []
        
        faviscore_by_pair = pw.agg_faviscore_by_pair()
        
        for pair_tuple, favi in faviscore_by_pair.items():
            if favi is None:
                continue
                
            model_a, model_b = pair_tuple
            model_a_family = FAMILY_MAP.get(model_a, 'Unknown')
            model_b_family = FAMILY_MAP.get(model_b, 'Unknown')
            
            if model_a_family == judge_family and model_b_family != judge_family:
                intra_family_faviscores.append(favi)
                
            elif model_b_family == judge_family and model_a_family != judge_family:
                intra_family_faviscores.append(-favi)
                
            elif model_a_family != judge_family and model_b_family != judge_family:
                cross_family_faviscores.append(favi)
            
        if len(intra_family_faviscores) < 3 or len(cross_family_faviscores) < 3:
            continue
        
        intra = np.array(intra_family_faviscores)
        cross = np.array(cross_family_faviscores)
        
        delta, delta_low, delta_high = bootstrap_ci_delta(intra, cross)
        
        _, p_value = permutation_test(intra, cross)
        
        if delta > 0:
            classification = 'Narcissism'
        else:
            classification = 'Rigidity'
        
        results.append({
            'judge': judge,
            'family': judge_family,
            'intra_faviscore_mean': np.mean(intra),
            'cross_faviscore_mean': np.mean(cross),
            'delta': delta,
            'delta_ci_lower': delta_low,
            'delta_ci_upper': delta_high,
            'p_value': p_value,
            'classification': classification,
            'significant': p_value < 0.05,
            'n_intra': len(intra),
            'n_cross': len(cross)
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df = df.sort_values('delta', ascending=False)
    
    x = np.arange(len(df))
    colors = [get_family_color(f) for f in df['family']]
    
    bars = ax.bar(x, df['delta'], color=colors, edgecolor='black', linewidth=1.5)
    
    yerr_lower = df['delta'] - df['delta_ci_lower']
    yerr_upper = df['delta_ci_upper'] - df['delta']
    ax.errorbar(x, df['delta'], yerr=[yerr_lower, yerr_upper],
               fmt='none', color='black', capsize=5, linewidth=2)
    
    ax.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.5)
    
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        y = bar.get_height()
        stars = significance_stars(row.p_value)
        
        ci_extent = row.delta_ci_upper if y >= 0 else row.delta_ci_lower
        va = 'bottom' if y >= 0 else 'top'
        y_pos = ci_extent + 0.05 if y >= 0 else ci_extent - 0.05
        
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'{y:.3f}{stars}',
               ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['judge'], rotation=45, ha='right')
    ax.set_xlabel('Judge Model', fontsize=12)
    ax.set_ylabel('Family Bias Δ (FaviScore units)', fontsize=12)
    ax.set_title('Family Bias Analysis\n(Δ = FaviScore_same_family - FaviScore_cross_family)\nAggregated Ratings',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P2_06_family_bias")
    save_csv(df, "P2_06_family_bias")
    
    for _, row in df.iterrows():
        sig = "*" if row['significant'] else ""
        print(f"    {row['judge']} ({row['family']}): Δ={row['delta']:+.3f} [{row['classification']}] p={row['p_value']:.3f}{sig}")
    
    for _, row in df[df['significant']].iterrows():
        print(f"      {row['judge']}: Δ={row['delta']:+.3f}")
    
    return df


if __name__ == "__main__":
    generate()

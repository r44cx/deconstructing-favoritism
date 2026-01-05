#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.plot_style import setup_style, save_plot, save_csv, FAMILY_COLORS
from core.config import FAMILY_MAP, RANDOM_BASELINE


def compute_metrics():
    p1_path = Path(__file__).parent.parent.parent / "output" / "data" / "P1_01_flip_consistency.csv"
    p2_path = Path(__file__).parent.parent.parent / "output" / "data" / "P2_01_judge_hierarchy.csv"
    
    if not p1_path.exists() or not p2_path.exists():
        return pd.DataFrame()
    
    p1 = pd.read_csv(p1_path)
    p2 = pd.read_csv(p2_path)
    
    common_judges = ['gpt-4o-mini', 'mixtral-8x7b', 'llama2-70b']
    
    results = []
    for judge in common_judges:
        p1_row = p1[p1['judge'] == judge]
        p2_row = p2[p2['judge'] == judge]
        
        if len(p1_row) == 0 or len(p2_row) == 0:
            continue
        
        p1_cons = p1_row['consistency'].values[0]
        p2_cons = p2_row['consistency'].values[0]
        delta = p2_cons - p1_cons
        
        results.append({
            'judge': judge,
            'family': FAMILY_MAP.get(judge, 'Unknown'),
            'phase1_consistency': p1_cons,
            'phase2_consistency': p2_cons,
            'delta': delta,
            'verdict': 'Improved' if delta > 0 else ('Stable' if delta > -5 else 'Collapsed')
        })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['phase1_consistency'], width,
                   label='Phase 1 (Baseline)', color='#3498db', edgecolor='black', linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, df['phase2_consistency'], width,
                   label='Phase 2 (Optimized)', color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax.axhline(RANDOM_BASELINE, color='gray', linestyle='--', linewidth=2,
               label=f'Random Baseline ({RANDOM_BASELINE:.1f}%)')
    
    for bar, row in zip(bars1, df.itertuples()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, row in zip(bars2, df.itertuples()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for i, row in enumerate(df.itertuples()):
        delta = row.delta
        color = '#27ae60' if delta > 0 else '#c0392b'
        y_pos = max(row.phase1_consistency, row.phase2_consistency) + 10
        ax.text(i, y_pos, f'Δ = {delta:+.1f}%', ha='center', fontsize=11, 
               fontweight='bold', color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['judge'], fontsize=11)
    ax.set_xlabel('Judge Model', fontsize=12)
    ax.set_ylabel('Flip Consistency (%)', fontsize=12)
    ax.set_title('Effect of Optimization Bundle on Judge Reliability\n(Phase 1 → Phase 2)',
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_metrics()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P2_XX_optimization_comparison")
    save_csv(df, "P2_XX_optimization_comparison")
    
    for _, row in df.iterrows():
        print(f"    {row['judge']}: P1={row['phase1_consistency']:.1f}% → P2={row['phase2_consistency']:.1f}% ({row['delta']:+.1f}%)")
    
    return df


if __name__ == "__main__":
    generate()

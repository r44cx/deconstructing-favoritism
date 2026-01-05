#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from core.data_loader import (
    load_phase2_data, extract_pairwise, PairwiseResults
)
from core.plot_style import setup_style, save_plot, save_csv, add_sample_size_annotation
from core.statistical_framework import faviscore_with_ci
from core.config import (
    FAMILY_MAP, JUDGE_ORDER, MODEL_ORDER, MODEL_SHORT,
    FAMILY_COLORS
)


def compute_model_favoritism():
    data = load_phase2_data(quiet=True)
    
    if not data:
        return pd.DataFrame()
    
    judge_model_data = defaultdict(lambda: defaultdict(list))
    
    for judge in data.keys():
        jdata = data[judge]
        pair_data = extract_pairwise(jdata, None)
        pw = PairwiseResults(pair_data)
        
        for pair_tuple, pdata in pw.per_pair.items():
            model_a, model_b = pair_tuple
            human = np.array(pdata['human'])
            agg = np.array(pdata['agg'])
            
            for i, (h, j) in enumerate(zip(human, agg)):
                if j == 1:
                    judge_model_data[judge][model_a].append(1)
                    judge_model_data[judge][model_b].append(-1)
                elif j == -1:
                    judge_model_data[judge][model_a].append(-1)
                    judge_model_data[judge][model_b].append(1)
    
    results = []
    for judge in JUDGE_ORDER:
        if judge not in judge_model_data:
            continue
        for model in MODEL_ORDER:
            if model not in judge_model_data[judge]:
                continue
            scores = judge_model_data[judge][model]
            if len(scores) > 10:
                favi = np.mean(scores)
                results.append({
                    'judge': judge,
                    'model': model,
                    'model_short': MODEL_SHORT.get(model, model[:6]),
                    'faviscore': favi,
                    'n': len(scores)
                })
    
    return pd.DataFrame(results)


def create_plot(df):
    setup_style()
    
    pivot = df.pivot(index='judge', columns='model_short', values='faviscore')
    
    judge_order = [j for j in JUDGE_ORDER if j in pivot.index]
    model_order = [MODEL_SHORT.get(m, m[:6]) for m in MODEL_ORDER if MODEL_SHORT.get(m, m[:6]) in pivot.columns]
    
    pivot = pivot.reindex(index=judge_order, columns=model_order)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=True,
        fmt='.2f',
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'FaviScore'},
        ax=ax,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    n_judges = len(judge_order)
    n_models = len(model_order)
    
    block_positions = [
        (0, 0, 2, 2),
        (2, 2, 2, 2),
        (4, 4, 2, 2),
    ]
    
    for y, x, h, w in block_positions:
        if y + h <= n_judges and x + w <= n_models:
            rect = plt.Rectangle((x, y), w, h, fill=False, 
                                 edgecolor='black', linewidth=3, linestyle='--')
            ax.add_patch(rect)
    
    for i, label in enumerate(ax.get_yticklabels()):
        judge = judge_order[i]
        family = FAMILY_MAP.get(judge, 'Unknown')
        label.set_color(FAMILY_COLORS.get(family, 'black'))
        label.set_fontweight('bold')
    
    for i, label in enumerate(ax.get_xticklabels()):
        model_short = model_order[i]
        for full, short in MODEL_SHORT.items():
            if short == model_short:
                family = FAMILY_MAP.get(full, 'Unknown')
                label.set_color(FAMILY_COLORS.get(family, 'black'))
                label.set_fontweight('bold')
                break
    
    ax.set_title('Judge-Model Favoritism Heatmap',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Being Evaluated', fontsize=12)
    ax.set_ylabel('Judge', fontsize=12)
    
    plt.tight_layout()
    return fig


def generate():
    df = compute_model_favoritism()
    
    if df.empty:
        return None
    
    fig = create_plot(df)
    save_plot(fig, "P2_05_faviscore_heatmap")
    save_csv(df, "P2_05_faviscore_heatmap")
    
    extreme_pos = df.nlargest(3, 'faviscore')
    extreme_neg = df.nsmallest(3, 'faviscore')
    
    for _, row in extreme_pos.iterrows():
        print(f"      {row['judge']} → {row['model_short']}: Φ={row['faviscore']:.2f}")
    
    for _, row in extreme_neg.iterrows():
        print(f"      {row['judge']} → {row['model_short']}: Φ={row['faviscore']:.2f}")
    
    return df


if __name__ == "__main__":
    generate()

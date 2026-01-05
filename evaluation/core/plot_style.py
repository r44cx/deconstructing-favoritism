import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union

from .config import (
    PLOTS_DIR, TABLES_DIR, DATA_DIR,
    FAMILY_COLORS, JUDGE_COLORS, STRATEGY_COLORS, GRADE_COLORS,
    RANDOM_BASELINE
)

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Georgia'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'figure.figsize': (10, 6),
    })


def save_plot(fig: plt.Figure, name: str, formats: List[str] = ['pdf', 'png']) -> List[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for fmt in formats:
        path = PLOTS_DIR / f'{name}.{fmt}'
        fig.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
        paths.append(path)
    
    plt.close(fig)
    return paths


def save_csv(df, name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f'{name}.csv'
    df.to_csv(path, index=False)
    return path


def save_latex_table(content: str, name: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f'{name}.tex'
    with open(path, 'w') as f:
        f.write(content)
    return path


def get_family_color(name: str) -> str:
    if name in FAMILY_COLORS:
        return FAMILY_COLORS[name]
    if name in JUDGE_COLORS:
        return JUDGE_COLORS[name]
    return '#7f8c8d'


def get_grade_color(grade: str) -> str:
    return GRADE_COLORS.get(grade, '#7f8c8d')


def get_consistency_color(value: float) -> str:
    if value >= 60:
        return '#2ecc71'
    elif value >= 40:
        return '#f39c12'
    elif value >= RANDOM_BASELINE:
        return '#e74c3c'
    else:
        return '#8e44ad'


def add_significance_stars(
    ax: plt.Axes,
    x: float,
    y: float,
    stars: str,
    fontsize: int = 12,
    color: str = 'black',
    offset: float = 0.02
) -> None:
    if not stars:
        return
    
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = y_range * offset
    
    ax.text(x, y + y_offset, stars, ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold', color=color)


def add_ci_errorbar(
    ax: plt.Axes,
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    ci_lower: Union[float, np.ndarray],
    ci_upper: Union[float, np.ndarray],
    color: str = 'black',
    capsize: int = 4,
    linewidth: float = 1.5,
    horizontal: bool = False
) -> None:
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    ci_lower, ci_upper = np.atleast_1d(ci_lower), np.atleast_1d(ci_upper)
    
    yerr_lower = y - ci_lower
    yerr_upper = ci_upper - y
    
    if horizontal:
        ax.errorbar(y, x, xerr=[yerr_lower, yerr_upper], fmt='none',
                   color=color, capsize=capsize, linewidth=linewidth)
    else:
        ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt='none',
                   color=color, capsize=capsize, linewidth=linewidth)


def add_baseline_line(
    ax: plt.Axes,
    value: float = RANDOM_BASELINE,
    orientation: str = 'horizontal',
    label: str = 'Random Baseline',
    color: str = 'gray',
    linestyle: str = '--',
    linewidth: float = 2,
    alpha: float = 0.7
) -> None:
    if orientation == 'horizontal':
        ax.axhline(value, ls=linestyle, c=color, lw=linewidth, 
                  alpha=alpha, label=label)
    else:
        ax.axvline(value, ls=linestyle, c=color, lw=linewidth,
                  alpha=alpha, label=label)


def add_quadrant_labels(
    ax: plt.Axes,
    x_mid: float,
    y_mid: float,
    labels: dict = None
) -> None:
    if labels is None:
        labels = {
            'top_right': 'RELIABLE',
            'top_left': 'CONSISTENT\nBUT WRONG',
            'bottom_right': 'UNSTABLE',
            'bottom_left': 'UNUSABLE'
        }
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    positions = {
        'top_right': ((x_mid + xlim[1]) / 2, (y_mid + ylim[1]) / 2),
        'top_left': ((xlim[0] + x_mid) / 2, (y_mid + ylim[1]) / 2),
        'bottom_right': ((x_mid + xlim[1]) / 2, (ylim[0] + y_mid) / 2),
        'bottom_left': ((xlim[0] + x_mid) / 2, (ylim[0] + y_mid) / 2),
    }
    
    colors = {
        'top_right': '#27ae60',
        'top_left': '#f39c12',
        'bottom_right': '#f39c12',
        'bottom_left': '#c0392b',
    }
    
    for key, (x, y) in positions.items():
        if key in labels:
            ax.text(x, y, labels[key], ha='center', va='center',
                   fontsize=10, fontweight='bold', color=colors[key],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_family_legend(ax: plt.Axes, loc: str = 'upper left') -> None:
    legend_elements = [
        mpatches.Patch(facecolor=c, edgecolor='black', label=f)
        for f, c in FAMILY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, title='Family', loc=loc)


def create_grade_legend(ax: plt.Axes, loc: str = 'upper right') -> None:
    legend_elements = [
        mpatches.Patch(facecolor=c, edgecolor='black', label=g.capitalize())
        for g, c in GRADE_COLORS.items()
    ]
    ax.legend(handles=legend_elements, title='Grade', loc=loc)


def create_strategy_legend(ax: plt.Axes, loc: str = 'upper right') -> None:
    legend_elements = [
        mpatches.Patch(facecolor=c, edgecolor='black', label=s)
        for s, c in STRATEGY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, title='Strategy', loc=loc)


def format_judge_labels(judges: List[str], short: bool = False) -> List[str]:
    if short:
        return [j.replace('-', '\n') for j in judges]
    return judges


def add_value_labels(
    ax: plt.Axes,
    bars,
    fmt: str = '{:.1f}%',
    offset: float = 1,
    fontsize: int = 9,
    rotation: int = 0
) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight='bold',
            rotation=rotation
        )


def create_title_with_subtitle(
    ax: plt.Axes,
    title: str,
    subtitle: str,
    title_fontsize: int = 13,
    subtitle_fontsize: int = 10
) -> None:
    ax.set_title(f'{title}\n{subtitle}', fontsize=title_fontsize, fontweight='bold')


def add_sample_size_annotation(
    ax: plt.Axes,
    n: int,
    location: str = 'bottom_right',
    fontsize: int = 10,
    approximate: bool = False
) -> None:
    symbol = '≈' if approximate else '='
    text = f'n {symbol} {n:,}'
    
    positions = {
        'bottom_right': (0.95, 0.05, 'right', 'bottom'),
        'top_right': (0.95, 0.95, 'right', 'top'),
        'bottom_left': (0.05, 0.05, 'left', 'bottom'),
        'top_left': (0.05, 0.95, 'left', 'top'),
    }
    
    if location in positions:
        x, y, ha, va = positions[location]
        ax.text(x, y, text, transform=ax.transAxes,
               ha=ha, va=va, fontsize=fontsize,
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='gray', alpha=0.9, pad=0.5))


def generate_latex_table(
    df,
    caption: str,
    label: str,
    bold_columns: List[str] = None,
    highlight_max: List[str] = None,
    highlight_min: List[str] = None
) -> str:
    n_cols = len(df.columns)
    col_spec = 'l' + 'c' * (n_cols - 1)
    
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
    ]
    
    header = ' & '.join([f'\\textbf{{{c}}}' for c in df.columns]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            cell = str(val)
            
            if col == df.columns[0]:
                cell = f'\\textbf{{{cell}}}'
            
            cells.append(cell)
        
        lines.append(' & '.join(cells) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\end{table}',
    ])
    
    return '\n'.join(lines)


def create_bar_with_ci_plot(
    categories: List[str],
    values: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    colors: List[str] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    figsize: Tuple[int, int] = (10, 6),
    show_baseline: bool = True,
    baseline_value: float = RANDOM_BASELINE
) -> Tuple[plt.Figure, plt.Axes]:
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(categories))
    
    if colors is None:
        colors = [get_consistency_color(v) for v in values]
    
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.5)
    
    yerr_lower = [v - l for v, l in zip(values, ci_lowers)]
    yerr_upper = [u - v for v, u in zip(values, ci_uppers)]
    ax.errorbar(x, values, yerr=[yerr_lower, yerr_upper], fmt='none',
               color='black', capsize=5, linewidth=2)
    
    if show_baseline:
        add_baseline_line(ax, baseline_value)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    return fig, ax


def create_grouped_bar_plot(
    categories: List[str],
    groups: dict,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    figsize: Tuple[int, int] = (12, 6),
    show_baseline: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    n_categories = len(categories)
    n_groups = len(groups)
    width = 0.8 / n_groups
    
    x = np.arange(n_categories)
    
    for i, (group_name, group_data) in enumerate(groups.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, group_data['values'], width,
                     label=group_name, color=group_data.get('color', f'C{i}'),
                     edgecolor='black', linewidth=1)
    
    if show_baseline:
        add_baseline_line(ax)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    
    return fig, ax

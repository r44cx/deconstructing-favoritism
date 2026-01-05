from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PKG_DIR.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

OUTPUT_DIR = PKG_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"

SOURCE_DATA_DIR = PROJECT_ROOT / "data" / "judge_outputs"
PLOT_DATA_DIR = ANALYSIS_DIR / "plot_data"
DISAGREEMENT_DIR = ANALYSIS_DIR / "plots" / "disagreement_against_humans"

for d in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
ALPHA = 0.05
CONFIDENCE_LEVEL = 95
RANDOM_SEED = 42

SIG_LEVELS = {
    0.001: "***",
    0.01: "**",
    0.05: "*",
    1.0: ""
}

RANDOM_BASELINE = 33.3
CONSISTENCY_ACCEPTABLE = 60.0
CONSISTENCY_MARGINAL = 40.0
FAVISCORE_EXTREME_THRESHOLD = 0.5
MIN_N_FOR_ANALYSIS = 30
TEMPLATE_SIMILARITY_THRESHOLD = 0.95

JUDGE_ORDER = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "mixtral-8x7b",
    "mistral-7b",
    "llama2-13b",
    "llama2-70b",
]

MODEL_ORDER = [
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "llama-2-70b-chat",
    "llama-2-13b-chat",
    "mixtral-8x7b-instruct-v0.1",
    "mistral-7b-instruct",
]

FAMILY_ORDER = ["OpenAI", "Mistral", "Meta"]

FAMILY_MAP = {
    "gpt-4o-mini": "OpenAI",
    "gpt-3.5-turbo": "OpenAI",
    "mixtral-8x7b": "Mistral",
    "mistral-7b": "Mistral",
    "llama2-13b": "Meta",
    "llama2-70b": "Meta",
    "gpt-4-0613": "OpenAI",
    "gpt-3.5-turbo-0613": "OpenAI",
    "llama-2-70b-chat": "Meta",
    "llama-2-13b-chat": "Meta",
    "mixtral-8x7b-instruct-v0.1": "Mistral",
    "mistral-7b-instruct": "Mistral",
}

JUDGE_TO_MODEL = {
    'gpt-4o-mini': 'gpt-4-0613',
    'gpt-3.5-turbo': 'gpt-3.5-turbo-0613',
    'llama2-70b': 'llama-2-70b-chat',
    'llama2-13b': 'llama-2-13b-chat',
    'mixtral-8x7b': 'mixtral-8x7b-instruct-v0.1',
    'mistral-7b': 'mistral-7b-instruct',
}

MODEL_SHORT = {
    'gpt-4-0613': 'GPT-4',
    'gpt-3.5-turbo-0613': 'GPT-3.5',
    'llama-2-70b-chat': 'L2-70b',
    'llama-2-13b-chat': 'L2-13b',
    'mixtral-8x7b-instruct-v0.1': 'Mix-8x7b',
    'mistral-7b-instruct': 'Mis-7b',
}

DIMENSION_ORDER = [
    "Logical Robustness",
    "Completeness",
    "Logical Correctness",
    "Helpfulness",
    "Faithfulness",
    "Conciseness",
]

DIMENSIONS_INTERNAL = [
    'helpfulness', 'completeness', 'conciseness',
    'logical_correctness', 'logical_robustness', 'faithfulness'
]

DIMENSIONS = [
    'Completeness', 'Conciseness', 'Faithfulness',
    'Helpfulness', 'Logical Correctness', 'Logical Robustness'
]

DIMENSION_MAP = {
    'Completeness': 'completeness',
    'Conciseness': 'conciseness', 
    'Faithfulness': 'faithfulness',
    'Helpfulness': 'helpfulness',
    'Logical Correctness': 'logical_correctness',
    'Logical Robustness': 'logical_robustness'
}

RETAINED_DIMENSIONS = ["Logical Robustness", "Completeness", "Logical Correctness"]
EXCLUDED_DIMENSIONS = ["Helpfulness", "Faithfulness", "Conciseness"]

FAMILY_COLORS = {
    "OpenAI": "#2ca02c",
    "Mistral": "#ff7f0e",
    "Meta": "#1f77b4",
}

JUDGE_COLORS = {
    'gpt-4o-mini': '#2ca02c',
    'gpt-3.5-turbo': '#98df8a',
    'llama2-70b': '#1f77b4',
    'llama2-13b': '#aec7e8',
    'mixtral-8x7b': '#ff7f0e',
    'mistral-7b': '#ffbb78',
}

STRATEGY_COLORS = {
    'A/B': '#4285f4',
    'B/A': '#ea4335',
    'Aggregated': '#9c27b0',
}

GRADE_COLORS = {
    "acceptable": "#27ae60",
    "marginal": "#f39c12",
    "poor": "#e74c3c",
    "unusable": "#8e44ad",
}

FAVISCORE_CMAP = "RdYlGn"
FAVISCORE_VMIN = -2.0
FAVISCORE_VMAX = 2.0

PHASE1_FILES = {
    'gpt-4o-mini': 'high_density_sample_150_gpt4o_mini_outputs.json',
    'llama2-70b': 'high_density_sample_150_llama2_70b_chat_outputs.json',
    'mixtral-8x7b': 'high_density_sample_150_mixtral_8x7b_instruct_v0.1_q4_0_outputs.json',
}

PHASE2_FILES = {
    'gpt-4o-mini': 'high_density_full_2198_gpt4o_mini_optimized_outputs.json',
    'gpt-3.5-turbo': 'high_density_full_2198_gpt35_turbo_optimized_outputs.json',
    'llama2-70b': 'high_density_full_2198_llama2_70b_chat_outputs.json',
    'llama2-13b': 'high_density_full_2198_llama2_13b_chat_outputs.json',
    'mixtral-8x7b': 'high_density_full_2198_mixtral_8x7b_instruct_v0.1_q4_0_outputs.json',
    'mistral-7b': 'high_density_full_2198_mistral_7b_instruct_outputs.json',
}

CRITERION_KEYWORDS = {
    'robustness': ['robust', 'coherent', 'contradiction', 'consistent', 'logical flow'],
    'completeness': ['complete', 'thorough', 'comprehensive', 'addresses all', 'covers', 'detailed'],
    'correctness': ['correct', 'accurate', 'factual', 'right', 'true', 'valid', 'precise']
}

RUBRIC_PHRASES = [
    'more complete', 'more thorough', 'more detailed', 'more accurate',
    'logically robust', 'logically correct', 'addresses the question',
    'better overall', 'superior', 'preferred', 'more comprehensive'
]

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent.parent / 'prompts'

EVALUATION_DIMENSIONS = {
    'completeness': {
        'name': 'Completeness',
        'description': 'Whether the output includes all needed information and details.',
        'scale': 5,
        'scale_description': '5-point Likert scale (1-5)',
        'template_file': 'completeness.txt',
        'pairwise_template_file': 'pairwise_completeness.txt'
    },
    'conciseness': {
        'name': 'Conciseness',
        'description': 'Whether the output is focused on the input without irrelevant content.',
        'scale': 5,
        'scale_description': '5-point Likert scale (1-5)',
        'template_file': 'conciseness.txt',
        'pairwise_template_file': 'pairwise_conciseness.txt'
    },
    'logical_robustness': {
        'name': 'Logical Robustness',
        'description': 'Whether the reasoning in the output follows a clear flow.',
        'scale': 5,
        'scale_description': '5-point Likert scale (1-5)',
        'template_file': 'logical_robustness.txt',
        'pairwise_template_file': 'pairwise_logical_robustness.txt'
    },
    'logical_correctness': {
        'name': 'Logical Correctness',
        'description': 'Whether the output is factually accurate and addresses the input.',
        'scale': 3,
        'scale_description': '3-point scale (1-3)',
        'template_file': 'logical_correctness.txt',
        'pairwise_template_file': 'pairwise_logical_correctness.txt'
    },
    'helpfulness': {
        'name': 'Helpfulness',
        'description': 'How useful and supportive the output is for most users.',
        'scale': 7,
        'scale_description': '7-point Likert scale (1-7)',
        'template_file': 'helpfulness.txt',
        'pairwise_template_file': 'pairwise_helpfulness.txt'
    },
    'faithfulness': {
        'name': 'Faithfulness',
        'description': 'Whether the output reflects input without adding unrelated information.',
        'scale': 5,
        'scale_description': '5-point Likert scale (1-5)',
        'template_file': 'faithfulness.txt',
        'pairwise_template_file': 'pairwise_faithfulness.txt'
    }
}


def _load_template(template_name):
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding='utf-8')


def create_pairwise_prompt(prompt, response_a, response_b, flip_responses=False):
    template = _load_template('pairwise.txt')
    if flip_responses:
        return template.format(
            prompt=prompt,
            response_a=response_b,
            response_b=response_a
        )
    else:
        return template.format(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b
        )


def create_absolute_scoring_prompt(prompt, response, dimension_key, model_name):
    dimension = EVALUATION_DIMENSIONS[dimension_key]
    template = _load_template(dimension['template_file'])
    return template.format(
        prompt=prompt,
        response=response,
        model_name=model_name
    )


def get_all_dimension_keys():
    return list(EVALUATION_DIMENSIONS.keys())


def get_dimension_info(dimension_key):
    return EVALUATION_DIMENSIONS.get(dimension_key)


def create_dimension_pairwise_prompt(prompt, response_a, response_b, dimension_key, flip_responses=False):
    dimension = EVALUATION_DIMENSIONS[dimension_key]
    template = _load_template(dimension['pairwise_template_file'])
    if flip_responses:
        return template.format(
            prompt=prompt,
            response_a=response_b,
            response_b=response_a
        )
    else:
        return template.format(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b
        )


def create_optimized_pairwise_prompt(prompt, response_a, response_b, flip_responses=False):
    template = _load_template('pairwise_optimized.txt')
    if flip_responses:
        return template.format(
            prompt=prompt,
            response_a=response_b,
            response_b=response_a
        )
    else:
        return template.format(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b
        )


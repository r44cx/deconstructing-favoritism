import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from .config import (
    PROJECT_ROOT, SOURCE_DATA_DIR,
    PHASE1_FILES, PHASE2_FILES,
    JUDGE_TO_MODEL, FAMILY_MAP, MODEL_SHORT,
    DIMENSIONS_INTERNAL as DIMENSIONS
)

import sys
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'translations'))
try:
    from faviscore import FaviScore, RATING_INDEX_MAP
    FAVISCORE_CALC = FaviScore()
except ImportError:
    FAVISCORE_CALC = None
    RATING_INDEX_MAP = {1: 0, 0: 1, -1: 2}


def normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([a, b]))


def pair_label(pair_tuple: Tuple[str, str]) -> str:
    return f"{MODEL_SHORT.get(pair_tuple[0], pair_tuple[0][:6])} vs {MODEL_SHORT.get(pair_tuple[1], pair_tuple[1][:6])}"


def extract_winner(output: str) -> Optional[str]:
    if not output:
        return None
    
    if '```json' in output:
        m = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                w = data.get('winner')
                if w:
                    return w.upper() if w.lower() in ['a', 'b'] else w.lower()
            except:
                pass
    
    try:
        data = json.loads(output)
        w = data.get('winner')
        if w:
            return w.upper() if w.lower() in ['a', 'b'] else w.lower()
    except:
        pass
    
    m = re.search(r'["\']winner["\']\s*:\s*["\']([ABab])["\']', output)
    if m:
        return m.group(1).upper()
    
    m = re.search(r'\bwinner["\']\s*:\s*["\']([ABab])["\']', output)
    if m:
        return m.group(1).upper()
    
    nl_patterns = [
        r'(?:the\s+)?winner\s+is\s+Assistant\s+([AB])\b',
        r'(?:I\s+)?declare\s+(?:that\s+)?(?:Assistant\s+)?([AB])\s+(?:as\s+)?(?:the\s+)?winner',
        r'Assistant\s+([AB])\s+(?:is\s+)?(?:the\s+)?winner',
        r'(?:my\s+)?(?:final\s+)?verdict\s*(?:is)?:?\s*(?:Assistant\s+)?([AB])\b',
        r'\b([AB])\s+(?:is\s+)?(?:the\s+)?better\s+(?:response|assistant)',
        r'(?:choose|pick|select)\s+(?:Assistant\s+)?([AB])\b',
    ]
    for pattern in nl_patterns:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    
    tie_patterns = [
        r'["\']winner["\']\s*:\s*["\']tie["\']',
        r'\bwinner["\']\s*:\s*["\']tie["\']',
        r'(?:the\s+)?winner\s+is\s+(?:a\s+)?tie\b',
        r'\b(?:it\'?s\s+a\s+)?tie\b',
        r'\bboth\s+(?:assistants?\s+)?(?:are\s+)?equal',
        r'\bneither\s+(?:assistant\s+)?(?:is\s+)?better',
    ]
    for pattern in tie_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return 'tie'
    
    return None


def extract_reasoning(output: str) -> Optional[str]:
    if not output:
        return None
    
    try:
        if '```json' in output:
            m = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
                reasoning = data.get('reasoning') or data.get('analysis') or data.get('explanation')
                if reasoning:
                    return reasoning
        
        data = json.loads(output)
        reasoning = data.get('reasoning') or data.get('analysis') or data.get('explanation')
        if reasoning:
            return reasoning
    except:
        pass
    
    text = re.sub(r'```json.*?```', '', output, flags=re.DOTALL)
    text = re.sub(r'\{[^}]*"winner"[^}]*\}', '', text)
    return text.strip() if text.strip() else output


def extract_score(output: str) -> Optional[int]:
    if not output:
        return None
    try:
        if '```json' in output:
            m = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
            if m:
                output = m.group(1)
        data = json.loads(output)
        score = data.get('score')
        if score is not None:
            return int(score)
    except:
        pass
    return None


def compute_faviscore_simple(human_ratings: List[int], judge_ratings: List[int]) -> Optional[float]:
    if FAVISCORE_CALC is not None:
        valid = [(h, j) for h, j in zip(human_ratings, judge_ratings) 
                 if h in RATING_INDEX_MAP and j in RATING_INDEX_MAP]
        if len(valid) < 2:
            return None
        h_arr = np.array([p[0] for p in valid])
        j_arr = np.array([p[1] for p in valid])
        C = FAVISCORE_CALC.compute_confusion_matrix(h_arr, j_arr)
        return FAVISCORE_CALC.compute_faviscore(C)
    
    from .statistical_framework import compute_confusion_matrix_3x3, compute_faviscore
    valid = [(h, j) for h, j in zip(human_ratings, judge_ratings) 
             if h in [1, 0, -1] and j in [1, 0, -1]]
    if len(valid) < 2:
        return None
    h_arr = np.array([p[0] for p in valid])
    j_arr = np.array([p[1] for p in valid])
    cm = compute_confusion_matrix_3x3(h_arr, j_arr)
    return compute_faviscore(cm)


def get_judge_family(judge_name: str) -> str:
    return FAMILY_MAP.get(judge_name, 'Unknown')


def get_model_family(model_name: str) -> str:
    return FAMILY_MAP.get(model_name, 'Unknown')


def load_phase1_data(quiet: bool = False) -> Dict[str, Any]:
    data = {}
    for judge, fname in PHASE1_FILES.items():
        fpath = SOURCE_DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data[judge] = json.load(f)
        else:
            pass
    return data


def load_phase2_data(quiet: bool = False) -> Dict[str, Any]:
    data = {}
    for judge, fname in PHASE2_FILES.items():
        fpath = SOURCE_DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data[judge] = json.load(f)
        else:
            pass
    return data


def extract_pairwise(data: Dict, eval_type: str = 'pairwise') -> Dict:
    if eval_type is None:
        types = set(r.get('evaluation_type') for r in data['results'])
        if 'optimized_pairwise' in types:
            eval_type = 'optimized_pairwise'
        elif 'pairwise' in types:
            eval_type = 'pairwise'
    
    results = [r for r in data['results'] if r['evaluation_type'] == eval_type]
    pair_data = defaultdict(lambda: {'ab': {}, 'ba': {}})
    
    for r in results:
        cid = r['conversation_id']
        ma, mb = r.get('model_a'), r.get('model_b')
        pk = normalize_pair(ma, mb)
        
        flipped = r.get('judge_prompt_flipped', False)
        
        judge_output = r.get('judge_output', '')
        pos_winner = extract_winner(judge_output)
        reasoning = extract_reasoning(judge_output)
        
        if pos_winner in ['A', 'a']:
            rating = 1 if not flipped else -1
        elif pos_winner in ['B', 'b']:
            rating = -1 if not flipped else 1
        elif pos_winner in ['tie', 'Tie', 'TIE']:
            rating = 0
        else:
            rating = None
        
        hw = r.get('human_winner')
        if hw == 'model_a':
            h_rating = 1
        elif hw == 'model_b':
            h_rating = -1
        elif hw == 'tie':
            h_rating = 0
        else:
            h_rating = None
        
        if pk[0] != ma:
            if rating is not None:
                rating = -rating
            if h_rating is not None:
                h_rating = -h_rating
        
        entry = {
            'rating': rating,
            'human': h_rating,
            'model_a': ma,
            'model_b': mb,
            'pos_winner': pos_winner,
            'reasoning': reasoning,
        }
        
        if flipped:
            pair_data[pk]['ba'][cid] = entry
        else:
            pair_data[pk]['ab'][cid] = entry
    
    return dict(pair_data)


def extract_dimension_pairwise(data: Dict, dimension: str) -> Dict:
    results = [r for r in data['results'] 
               if r['evaluation_type'] == 'dimension_pairwise' and r.get('dimension') == dimension]
    
    pair_data = defaultdict(lambda: {'ab': {}, 'ba': {}})
    
    for r in results:
        cid = r['conversation_id']
        ma, mb = r.get('model_a'), r.get('model_b')
        pk = normalize_pair(ma, mb)
        flipped = r.get('judge_prompt_flipped', False)
        
        pos_winner = extract_winner(r.get('judge_output', ''))
        
        if pos_winner in ['A', 'a']:
            rating = 1 if not flipped else -1
        elif pos_winner in ['B', 'b']:
            rating = -1 if not flipped else 1
        elif pos_winner in ['tie', 'Tie', 'TIE']:
            rating = 0
        else:
            rating = None
        
        hw = r.get('human_winner')
        if hw == 'model_a':
            h_rating = 1
        elif hw == 'model_b':
            h_rating = -1
        elif hw == 'tie':
            h_rating = 0
        else:
            h_rating = None
        
        if pk[0] != ma:
            if rating is not None:
                rating = -rating
            if h_rating is not None:
                h_rating = -h_rating
        
        entry = {'rating': rating, 'human': h_rating}
        
        if flipped:
            pair_data[pk]['ba'][cid] = entry
        else:
            pair_data[pk]['ab'][cid] = entry
    
    return dict(pair_data)


def extract_absolute_scores(data: Dict) -> Dict:
    results = [r for r in data['results'] if r['evaluation_type'] == 'absolute']
    
    scores = defaultdict(lambda: defaultdict(dict))
    
    for r in results:
        cid = r['conversation_id']
        dim = r.get('dimension')
        model = r.get('evaluated_model_name')
        score = extract_score(r.get('judge_output', ''))
        
        if score is not None and dim and model:
            scores[cid][model][dim] = score
    
    return dict(scores)


def compute_aggregated_rating(ab_rating: int, ba_rating: int) -> Optional[int]:
    if ab_rating is None or ba_rating is None:
        return None
    
    if ab_rating == ba_rating:
        return ab_rating
    
    if (ab_rating == 1 and ba_rating == -1) or (ab_rating == -1 and ba_rating == 1):
        return 0
    
    if ab_rating == 0:
        return ba_rating
    if ba_rating == 0:
        return ab_rating
    
    return 0


class PairwiseResults:
    
    def __init__(self, pair_data: Dict):
        self.pair_data = pair_data
        self._process()
    
    def _process(self):
        self.per_pair = {}
        
        self.ab_ratings = []
        self.ba_ratings = []
        self.agg_ratings = []
        self.human_ratings = []
        self.conversation_ids = []
        self.pairs = []
        self.ab_reasoning = []
        self.ba_reasoning = []
        
        for pk, runs in self.pair_data.items():
            ab, ba = runs['ab'], runs['ba']
            common = set(ab.keys()) & set(ba.keys())
            
            pair_ab = []
            pair_ba = []
            pair_agg = []
            pair_human = []
            pair_ab_reasoning = []
            pair_ba_reasoning = []
            
            for cid in common:
                ab_r = ab[cid]['rating']
                ba_r = ba[cid]['rating']
                h = ab[cid]['human']
                
                if ab_r is None or ba_r is None or h is None:
                    continue
                
                pair_ab.append(ab_r)
                pair_ba.append(ba_r)
                pair_agg.append(compute_aggregated_rating(ab_r, ba_r))
                pair_human.append(h)
                pair_ab_reasoning.append(ab[cid].get('reasoning', ''))
                pair_ba_reasoning.append(ba[cid].get('reasoning', ''))
                
                self.ab_ratings.append(ab_r)
                self.ba_ratings.append(ba_r)
                self.agg_ratings.append(compute_aggregated_rating(ab_r, ba_r))
                self.human_ratings.append(h)
                self.conversation_ids.append(cid)
                self.pairs.append(pk)
                self.ab_reasoning.append(ab[cid].get('reasoning', ''))
                self.ba_reasoning.append(ba[cid].get('reasoning', ''))
            
            if pair_human:
                self.per_pair[pk] = {
                    'ab': pair_ab,
                    'ba': pair_ba,
                    'agg': pair_agg,
                    'human': pair_human,
                    'ab_reasoning': pair_ab_reasoning,
                    'ba_reasoning': pair_ba_reasoning,
                    'n': len(pair_human),
                }
        
        self.ab_ratings = np.array(self.ab_ratings)
        self.ba_ratings = np.array(self.ba_ratings)
        self.agg_ratings = np.array(self.agg_ratings)
        self.human_ratings = np.array(self.human_ratings)
    
    def get_pairs(self) -> List[Tuple[str, str]]:
        return list(self.per_pair.keys())
    
    def flip_consistency(self) -> float:
        if len(self.ab_ratings) == 0:
            return 0
        return float(np.mean(self.ab_ratings == self.ba_ratings) * 100)
    
    def flip_consistency_by_pair(self) -> Dict[Tuple[str, str], float]:
        result = {}
        for pk, data in self.per_pair.items():
            ab = np.array(data['ab'])
            ba = np.array(data['ba'])
            result[pk] = float(np.mean(ab == ba) * 100) if len(ab) > 0 else 0
        return result
    
    def ab_agreement(self) -> float:
        if len(self.ab_ratings) == 0:
            return 0
        return float(np.mean(self.ab_ratings == self.human_ratings) * 100)
    
    def ba_agreement(self) -> float:
        if len(self.ba_ratings) == 0:
            return 0
        return float(np.mean(self.ba_ratings == self.human_ratings) * 100)
    
    def agg_agreement(self) -> float:
        if len(self.agg_ratings) == 0:
            return 0
        return float(np.mean(self.agg_ratings == self.human_ratings) * 100)
    
    def ab_agreement_by_pair(self) -> Dict[Tuple[str, str], float]:
        result = {}
        for pk, data in self.per_pair.items():
            ab = np.array(data['ab'])
            h = np.array(data['human'])
            result[pk] = float(np.mean(ab == h) * 100) if len(ab) > 0 else 0
        return result
    
    def ba_agreement_by_pair(self) -> Dict[Tuple[str, str], float]:
        result = {}
        for pk, data in self.per_pair.items():
            ba = np.array(data['ba'])
            h = np.array(data['human'])
            result[pk] = float(np.mean(ba == h) * 100) if len(ba) > 0 else 0
        return result
    
    def agg_agreement_by_pair(self) -> Dict[Tuple[str, str], float]:
        result = {}
        for pk, data in self.per_pair.items():
            agg = np.array(data['agg'])
            h = np.array(data['human'])
            result[pk] = float(np.mean(agg == h) * 100) if len(agg) > 0 else 0
        return result
    
    def ab_faviscore_by_pair(self) -> Dict[Tuple[str, str], Optional[float]]:
        result = {}
        for pk, data in self.per_pair.items():
            favi = compute_faviscore_simple(data['human'], data['ab'])
            result[pk] = favi
        return result
    
    def ba_faviscore_by_pair(self) -> Dict[Tuple[str, str], Optional[float]]:
        result = {}
        for pk, data in self.per_pair.items():
            favi = compute_faviscore_simple(data['human'], data['ba'])
            result[pk] = favi
        return result
    
    def agg_faviscore_by_pair(self) -> Dict[Tuple[str, str], Optional[float]]:
        result = {}
        for pk, data in self.per_pair.items():
            favi = compute_faviscore_simple(data['human'], data['agg'])
            result[pk] = favi
        return result
    
    def first_position_preference(self) -> float:
        inconsistent_mask = self.ab_ratings != self.ba_ratings
        if not np.any(inconsistent_mask):
            return 50.0
        
        ab_inconsistent = self.ab_ratings[inconsistent_mask]
        ba_inconsistent = self.ba_ratings[inconsistent_mask]
        
        first_pos_count = np.sum((ab_inconsistent == 1) | (ab_inconsistent == 0))
        
        return float(first_pos_count / len(ab_inconsistent) * 100)


if __name__ == '__main__':
    p1 = load_phase1_data()
    p2 = load_phase2_data()
    
    if p1:
        judge = list(p1.keys())[0]
        pd = extract_pairwise(p1[judge], 'pairwise')
        
        results = PairwiseResults(pd)

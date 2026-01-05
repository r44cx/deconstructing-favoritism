import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "projects" / "minos-data-main"))
sys.path.append(str(Path(__file__).parent.parent / "projects" / "minos-core-main"))

from minos_data.wmt_metrics import WMTMetricsLoader
from minos_core.dataset import RatingType


def prepare_wmt_data():
    year = "wmt24"
    lang_pair = "en-de" 
    ref = "refA"
    data_dir = Path("data")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    loader = WMTMetricsLoader(base_dir=data_dir)
    
    print(f"Downloading WMT data for: {year} {lang_pair} {ref}")
    loader.download()
    
    domain = (year, lang_pair, ref)
    print(f"Loading preference data for: {domain}")
    
    preference_dataset = loader.load(
        domain=domain,
        human_rating_type=RatingType.PREFERENCE,
        metric_rating_type=RatingType.SCALAR
    )
    
    from minos_core.dataset.ratings.preference_conversions import NaivePairedMetricToPreferenceConversion
    
    preference_dataset = preference_dataset.metric_to_preference(
        NaivePairedMetricToPreferenceConversion()
    )
    
    print(f"  Systems: {list(preference_dataset.systems())}")
    print(f"  Metrics: {list(preference_dataset.metrics())}")
    print(f"  Comparisons: {len(preference_dataset.raw_ratings)}")
    
    return preference_dataset


def main():
    dataset = prepare_wmt_data()
    print(f"\nData preparation complete!")

if __name__ == "__main__":
    main()

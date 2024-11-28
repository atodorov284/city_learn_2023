import pandas as pd

def format_evaluation(evaluation_data: dict) -> pd.DataFrame:
    kpis = pd.DataFrame.from_dict(
        evaluation_data, orient="index", columns=["value", "display_name", "weight"]
    )
    kpis_reset = kpis.reset_index().rename(columns={"index": "metric"})
    return kpis_reset
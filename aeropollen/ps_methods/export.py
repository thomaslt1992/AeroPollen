import json
from pathlib import Path

import pandas as pd

from .clinical import (
    _canonical_pollen_type,
    _resolve_clinical_params,
    load_clinical_cfg,
)


def export_results(
    results: pd.DataFrame,
    source_name: str | Path,
    arguments: dict,
    pollen_columns: list,
    method: str,
) -> Path:
    """Write results followed by JSON-valued, comment-prefixed metadata rows."""
    source = Path(source_name)
    output = source.with_name(f"{source.stem}_ps.csv")
    metadata = {
        "source_name": str(source),
        "pollen_columns": list(pollen_columns),
        "arguments": arguments,
        "method": method,
    }
    if method == "clinical":
        config = load_clinical_cfg()
        effective = {}
        for column in pollen_columns:
            selected = arguments["clinical_pollen_type"]
            if selected is None and _canonical_pollen_type(str(column)) in config:
                selected = str(column)
            params = _resolve_clinical_params(
                pollen_params=config,
                clinical_pollen_type=selected,
                n_clinical=arguments["n_clinical"],
                window_clinical=arguments["window_clinical"],
                th_pollen=arguments["th_pollen"],
                th_sum=arguments["th_sum"],
                day_threshold=arguments["th_day"],
            )
            effective[str(column)] = {
                **params,
                "ppp_min_consecutive_days": 3,
                "ppp_threshold": params["day_threshold"],
            }
        metadata["effective_clinical_parameters"] = effective
    if method == "moving":
        man = arguments["man"]
        metadata["effective_man"] = man if man % 2 else man + 1
    metadata["unused_compatibility_arguments"] = [
        "reduction",
        "red_level",
        "derivative",
        "season_type",
        "plot",
        "export_plot",
    ]
    footer = [
        f"# {key}={json.dumps(value, default=str, ensure_ascii=False)}\n"
        for key, value in metadata.items()
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        results.to_csv(stream, index=False)
        stream.write("\n# AeroPollen parameters (JSON values)\n")
        stream.writelines(footer)
    return output

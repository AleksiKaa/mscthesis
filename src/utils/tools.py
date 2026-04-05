import json
import re
from pathlib import Path

import pandas as pd

def get_config(path):
    config = None
    with open(path, "r", encoding="utf-8") as config_json:
        config = json.loads("\n".join(config_json.readlines()))
    return config


def collect_jobs(base_dir, model_families=None):
    if model_families is None:
        model_families = ["Qwen", "meta-llama", "mistralai"]

    jobs = {}

    base_path = Path(base_dir)

    for model_family in base_path.iterdir():
        if not model_family.is_dir():
            continue

        if model_family.stem not in model_families:
            continue

        for model in model_family.iterdir():
            if not model.is_dir():
                continue

            for job in model.iterdir():
                if not job.is_dir():
                    continue

                results_file = job / "result.csv"
                config_file = job / "config.json"

                if results_file.exists() and config_file.exists():
                    model_name = model.name

                    if model_name not in jobs.keys():
                        jobs[model_name] = []

                    job_id = job.name

                    jobs[model_name].append(
                        {
                            "job_id": job_id,
                            "model_family": model_family.name,
                            "model": model.name,
                            "path": str(job),
                            "result_csv": str(results_file),
                            "config_json": str(config_file),
                        }
                    )

    return jobs


def prettify_table(df):
    # Sort values
    df = df.sort_values(
        by=["number_of_demonstrations", "type_of_demonstrations", "use_instructions"],
        axis=0,
    )
    # Map column to yes/no
    df["use_instructions"] = df["use_instructions"].apply(
        lambda x: "yes" if bool(x) else "no"
    )
    #  Map column to negative/mixed/positive
    df["type_of_demonstrations"] = df["type_of_demonstrations"].apply(
        lambda x: "positive" if x > 0 else "mixed" if x == 0 else "negative"
    )
    # When num is 0, remove value of type
    df.loc[df["number_of_demonstrations"] == 0, "type_of_demonstrations"] = "none"

    return df


def bold_extreme_values(s, by_model=True):
    # Bold max for mean

    if 'mean' not in s.name and 'std' not in s.name:
        return ['' for v in s]


    if not by_model:
        if "mean" in s.name:
            is_max = s == s.max()
            return ['font-weight: bold' if v else '' for v in is_max]
        if "std" in s.name:
            is_min = s == s.min()
            return ['font-weight: bold' if v else '' for v in is_min]
    
    font_array = []

    model_level = s.index.names.index('model')
    models = s.index.get_level_values(model_level)
    models = pd.Series(list(models)).unique()

    idx = pd.IndexSlice
    
    for model in models:   
        values_by_model = s.loc[idx[model]]
        
        if "mean" in s.name:
            is_max = values_by_model == values_by_model.max()
            font_array += ['font-weight: bold' if v else '' for v in is_max]
        if "std" in s.name:
            is_min = values_by_model == values_by_model.min()
            font_array += ['font-weight: bold' if v else '' for v in is_min]

    return font_array


def aggregate_results(df, by_cols, cols, funs=None):
    grouped = df.groupby(
        by=by_cols,
        as_index=True,
    )
    
    if funs is None:
        funs = [
        "mean",
        "std",
        "count"
    ]

    agg = grouped.agg({col: funs for col in cols})

    return agg


def format_table(df):
    """
    Replace '_' with ' ' in all levels of both columns and index.
    Take only model family name and param count for models.
    """
    df = df.copy()

    def _format_model_name(name):
        parts = name.split("-")
        match = re.search(r"\d{1,2}B", name)
        return f"{parts[0]}-{match.group(0)}" if match else name

    def _replace_in_index(idx):
        if isinstance(idx, pd.MultiIndex):
            return pd.MultiIndex.from_tuples(
                [
                    tuple(
                        str(level).replace("_", " ") if level is not None else level
                        for level in tup
                    )
                    for tup in idx
                ],
                names=[name.replace('_', ' ') if name is not None else None for name in idx.names]
            )
        else:
            return pd.Index(
                [
                    str(val).replace("_", " ") if val is not None else val
                    for val in idx
                ],
                name=idx.name
            )

    df.columns = _replace_in_index(df.columns)
    df.index = _replace_in_index(df.index)

    model_level = df.index.names.index('model')
    models = df.index.get_level_values(model_level).unique()
    df.index = df.index.set_levels(levels=[_format_model_name(model) for model in models], level=model_level)

    return df
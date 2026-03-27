import json
from pathlib import Path


def get_config(path):
    config = None
    with open(path, "r", encoding="utf-8") as config_json:
        config = json.loads("\n".join(config_json.readlines()))
    return config


def collect_jobs(base_dir, model_families=None):
    if model_families is None:
        model_families = ["Qwen", "meta-llama"]
    
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
    df = df.sort_values(by=["number_of_demonstrations", "type_of_demonstrations", "use_instructions"], axis=0)
    # Map column to yes/no
    df["use_instructions"] = df["use_instructions"].apply(lambda x: "yes" if bool(x) else "no")
    #  Map column to negative/mixed/positive
    df["type_of_demonstrations"] = df["type_of_demonstrations"].apply(lambda x: "positive" if x > 0 else "mixed" if x == 0 else "negative")
    # When num is 0, remove value of type
    df.loc[df["number_of_demonstrations"] == 0, "type_of_demonstrations"] = "none"
    
    return df

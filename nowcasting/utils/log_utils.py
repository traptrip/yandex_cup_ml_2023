from pathlib import Path


def get_exp_name(log_dir: Path):
    new_exp_name = log_dir.name
    prev_exps = [exp.name for exp in log_dir.parent.iterdir()]
    prev_exps = [exp for exp in prev_exps if "_".join(exp.split("_")[:-1]) == new_exp_name]
    prev_exps_nums = [
        int(exp.split("_")[-1])
        for exp in prev_exps
        if "_".join(exp.split("_")[:-1]) == new_exp_name
    ]
    if prev_exps:
        last_exp_num = max(prev_exps_nums)
        log_dir = log_dir.parent / f"{new_exp_name}_{last_exp_num + 1}"
    else:
        log_dir = log_dir.parent / f"{new_exp_name}_1"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

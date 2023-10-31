from pathlib import Path


def get_exp_name(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    new_exp_name = log_dir.name
    prev_exps = [exp.name for exp in log_dir.parent.iterdir()]
    last_exp_num = ""
    for exp in prev_exps:
        if new_exp_name in exp:
            tmp = str(exp).split("_")
            if len(tmp) > 1:
                last_exp_num = int(tmp[-1]) + 1
            else:
                last_exp_num = 1
            last_exp_num = f"_{last_exp_num}"
    return log_dir.parent / f"{new_exp_name}{last_exp_num}"

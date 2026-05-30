import scripts._bootstrap  # noqa: F401

from scripts._experiment_runner import run_experiment

if __name__ == "__main__":
    run_experiment("config_rrtmil", "rrtmil")

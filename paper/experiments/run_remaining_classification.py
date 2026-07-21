"""Run the three classification datasets still missing full results
(Breast Cancer Wisconsin, Wine, Titanic) using the already-incremental
run_benchmark.run_classification_dataset, which saves after each dataset."""
import warnings

warnings.filterwarnings("ignore")

from datasets import load_breast_cancer, load_titanic, load_wine
from run_benchmark import run_classification_dataset

if __name__ == "__main__":
    for loader in (load_breast_cancer, load_wine, load_titanic):
        run_classification_dataset(loader)
    print("\nDone with remaining classification datasets.")

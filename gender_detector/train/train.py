import utils
import argparse

parser = argparse.ArgumentParser(description="Train SVM on UTKFace")
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to load (stratified)')
parser.add_argument('--n_runs', type=int, default=1, help='Number of times to run nested cross-validation')
args = parser.parse_args()

DATASET_DIR = "train/UTKFace"
n_samples = args.n_samples
n_runs = args.n_runs

X, y = utils.load_dataset(DATASET_DIR, n_samples)

results, y_true, y_pred, y_score = utils.run_multiple_nested_cv(X, y, n_runs=n_runs, n_splits=5)

print()
print(results)

utils.evaluate_model(results, y_true, y_pred, y_score)

final_model, scaler = utils.retrain_final_model(X, y, results)
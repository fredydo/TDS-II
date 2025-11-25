import os
import utils
import argparse

parser = argparse.ArgumentParser(description="Train SVM on UTKFace")
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to load (stratified)')
parser.add_argument('--n_runs', type=int, default=1, help='Number of times to run nested cross-validation')
parser.add_argument('--model_choice', type=str, default="nb", choices=['logreg', 'nb', 'qda'], help='Model to use: logreg | nb | qda')
args = parser.parse_args()

BASE_DIR = os.path.dirname(__file__)  # folder where train.py is
DATASET_DIR = os.path.join(BASE_DIR, "UTKFace")  # assuming UTKFace is inside the train folder

n_samples = args.n_samples
n_runs = args.n_runs

X, y = utils.load_dataset(DATASET_DIR, n_samples)

results, y_true, y_pred = utils.run_multiple_nested_cv(X, y, n_runs=n_runs, n_splits=5, model_choice=args.model_choice)

print()
print(results)

utils.evaluate_model(results, y_true, y_pred)

final_model, scaler = utils.retrain_final_model(X, y, results, model_choice=args.model_choice)
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--model', type=str, default="model/svm_gender_model.joblib")
parser.add_argument('--scaler', type=str, default="model/scaler.joblib")
args = parser.parse_args()

model, scaler = utils.load_model(args.model, args.scaler)
utils.predict(model, scaler, args.image)
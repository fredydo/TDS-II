import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.svm import SVC
from collections import Counter
import matplotlib.pyplot as plt
from skimage.feature import hog
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, recall_score, precision_score

IMG_SIZE = 64
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def extract_gender(filename):
    """Filename format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg"""
    try:
        return int(filename.split('_')[1])  # 0: male, 1: female
    except:
        return None

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=("true"))
    cf_matrix_v = confusion_matrix(y_true, y_pred)

    cmap = sns.color_palette("Blues", as_cmap=True)

    plt.figure(figsize = (6,6))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap=cmap, fmt='d', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'], square=True, cbar=True)

    for i in range(len(cf_matrix)):
        for j in range(len(cf_matrix[i])):
            text = f"{cf_matrix[i][j]*100: 0.2f}%\n({cf_matrix_v[i][j]})"

            cell_bg_color = np.mean(cmap(cf_matrix[i][j]))
            text_color = 'black' if cell_bg_color > 0.7 else 'white'

            heatmap.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color)

    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(filename, dpi=300)

    return cf_matrix

def plot_ROC_curve(all_y_true, all_y_score, filename='roc_curve.png'):
    """Generate and save ROC curve"""

    fpr, tpr, _ = roc_curve(all_y_true, all_y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_probability_density(y_true, y_proba, filename='probability_density.png'):
    """Plot density distribution of predicted probabilities for each class"""

    df = pd.DataFrame({
        'Probability': y_proba,
        'Group': ['Male' if label == 0 else 'Female' for label in y_true]
    })

    blues = sns.color_palette("Blues", n_colors=6)
    colors = [blues[1], blues[4]]

    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 4]}, figsize=(6, 6), sharex=True)

    sns.boxplot(x="Probability", y="Group", hue="Group", data=df, dodge=False, palette=colors, ax=axes[0], legend=False)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_yticks([])
    axes[0].set_title("Score Distribution")

    sns.histplot(data=df, x="Probability", hue="Group", kde=True, palette=colors, stat="density", bins=10, ax=axes[1], legend=True)

    axes[1].set_xlabel("Score")
    legend_handles = [
        Patch(color=colors[0], label='Male'),
        Patch(color=colors[1], label='Female')
    ]
    axes[1].legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_samples(dataset_dir, num_rows=5, num_cols=6):

    gender_map = {0: 'Male', 1: 'Female'}
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

    sampled_files = random.sample(image_files, num_rows * num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle("Random UTKFace Samples (Age - Gender)", fontsize=16)

    for ax, file in zip(axes.flatten(), sampled_files):
        parts = file.split('_')

        try:
            age = int(parts[0])
            gender = gender_map.get(int(parts[1]), 'Unknown')
            label = f"{age} - {gender}"
        except:
            label = "Unknown"

        img_path = os.path.join(dataset_dir, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_gender_distribution(image_files):

    genders = []
    for f in image_files:
        try:
            gender = int(f.split('_')[1])
            genders.append(gender)
        except:
            continue

    # Count genders
    gender_counts = Counter(genders)
    labels = ['Male', 'Female']
    counts = [gender_counts.get(0, 0), gender_counts.get(1, 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=['skyblue', 'lightcoral'])
    plt.title('Gender Distribution in UTKFace')
    plt.xlabel('Gender')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def extract_hog_features(image_path, img_size=IMG_SIZE):
    """Read image, convert to grayscale, resize and extract HOG features."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    features = hog(resized,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   feature_vector=True)
    return features

def load_dataset(dataset_dir, n_samples=None):
    """Load dataset; if n_samples is set, returns a stratified sample."""
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

    labeled_files = [(f, extract_gender(f)) for f in image_files]
    
    files = [file[0] for file in labeled_files]
    labels = [labels[1] for labels in labeled_files]

    if n_samples is not None and n_samples < len(files):
        sampled_files, _, sampled_labels, _ = train_test_split(
            files, labels, train_size=n_samples, stratify=labels, random_state=42
        )
    else:
        sampled_files, sampled_labels = files, labels

    X, y = [], []
    for file, label in zip(sampled_files, sampled_labels):
        path = os.path.join(dataset_dir, file)
        features = extract_hog_features(path)
        if features is not None:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def retrain_final_model(X_full, y_full, results, model_path="svm_gender_model.joblib", scaler_path="scaler.joblib", random_state=42):
    """Retrain the final SVM model on all data using the best hyperparameters."""

    results['C'] = results['best_params'].apply(lambda x: x['C'])
    results['kernel'] = results['best_params'].apply(lambda x: x['kernel'])

    # Get mode (most common) C and kernel
    best_C = Counter(results['C']).most_common(1)[0][0]
    best_kernel = Counter(results['kernel']).most_common(1)[0][0]
    best_params = {'C': best_C, 'kernel': best_kernel}

    print("Retraining final model on all data with best params:", best_params)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    final_model = SVC(C=best_C, kernel=best_kernel, probability=True, random_state=random_state)
    final_model.fit(X_scaled, y_full)

    dump(final_model, model_path)
    print(f"Final model saved to {model_path}")

    return final_model, scaler

def nested_cv(X, y, n_splits=5, random_state=42):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_true, all_y_pred, all_y_score = [], [], []

    param_grid = {
        'C': [0.01, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }

    results = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nOuter Fold {i + 1}:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(random_state=random_state)
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)
        
        best_params = grid_search.best_params_
        best_model = SVC(C=best_params['C'], kernel=best_params['kernel'], random_state=random_state)
        best_model.fit(X_train_scaled, y_train)

        y_pred = best_model.predict(X_test_scaled)
        y_score = best_model.decision_function(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_score.extend(y_score)

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Inner CV best score: {grid_search.best_score_:.4f}")
        print(f"  Outer fold accuracy: {acc:.4f}")

        results.append({
            'fold': i + 1,
            'best_params': grid_search.best_params_,
            'accuracy': acc,
            'precision': prec,
            'sensitivity': recall,
            'specificity': specificity,
            'f1_score': f1,
            'conf_matrix': cm.tolist(),
        })

    return pd.DataFrame(results), all_y_true, all_y_pred, all_y_score

def evaluate_model(results, y_true, y_pred, y_score,
                            cm_file='confusion_matrix.png',
                            roc_file='roc_curve.png',
                            density_file='probability_density.png'):
    """Evaluate a binary classifier and show/save confusion matrix, ROC curve, and predicted probability density."""

    def show(name, values):
        print(f"{name:<12}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    print()    
    print("=== Evaluation Results ===")

    show("Accuracy", results["accuracy"])
    show("Sensitivity", results["sensitivity"])
    show("Specificity", results["specificity"])
    show("Precision", results["precision"])
    show("F1 Score", results["f1_score"])
    print()

    plot_confusion_matrix(y_true, y_pred, filename=cm_file)
    plot_ROC_curve(y_true, y_score, filename=roc_file)
    plot_probability_density(y_true, y_score, filename=density_file)

def run_multiple_nested_cv(X, y, n_runs=5, n_splits=5):
    all_results = []
    all_y_true_all_runs = []
    all_y_pred_all_runs = []
    all_y_score_all_runs = []

    for run in range(n_runs):
        print(f"\n====== Run {run + 1} ======")
        results_df, y_true, y_pred, y_score = nested_cv(X, y, n_splits=n_splits, random_state=run)

        results_df['run'] = run + 1
        all_results.append(results_df)

        all_y_true_all_runs.extend(y_true)
        all_y_pred_all_runs.extend(y_pred)
        all_y_score_all_runs.extend(y_score)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df, all_y_true_all_runs, all_y_pred_all_runs, all_y_score_all_runs
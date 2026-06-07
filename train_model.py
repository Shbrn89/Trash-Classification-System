import os
import json
import random
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from feature_extraction import extract_features_from_image, IMAGE_SIZE


try:
    cv2.setNumThreads(1)
except Exception:
    pass


DATASET_DIR = "trash_project/dataset/raw/garbage_classification"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trash_xgboost_model.pkl")
FEATURE_CACHE_PATH = os.path.join(MODEL_DIR, "feature_cache_12class_xgb_v1.npz")

CLASSES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CPU_COUNT = os.cpu_count() or 4
N_JOBS_FEATURE = max(1, CPU_COUNT - 1)

RUN_COMPARISON = False


def collect_dataset():
    image_paths = []
    labels = []

    print("\nReading dataset...")

    for label_id, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(
                f"Folder tidak ditemukan: {class_dir}\n"
                "Pastikan nama folder dataset sama persis dengan CLASSES."
            )

        files = [
            file for file in os.listdir(class_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        files = sorted(files)

        if len(files) == 0:
            raise ValueError(f"Tidak ada gambar di folder: {class_dir}")

        print(f"{class_name}: {len(files)} images")

        for file in files:
            image_paths.append(os.path.join(class_dir, file))
            labels.append(label_id)

    if len(image_paths) == 0:
        raise ValueError("Dataset kosong. Cek ulang folder dataset kamu.")

    return np.array(image_paths), np.array(labels)


def process_single_image(path, label):
    image = cv2.imread(path)

    if image is None:
        return None

    features = extract_features_from_image(image)

    return features, int(label)


def build_feature_matrix_parallel(image_paths, labels):
    print(f"\nUsing {N_JOBS_FEATURE} CPU workers for feature extraction...")

    results = joblib.Parallel(
        n_jobs=N_JOBS_FEATURE,
        backend="loky",
        verbose=10
    )(
        joblib.delayed(process_single_image)(path, label)
        for path, label in zip(image_paths, labels)
    )

    feature_list = []
    label_list = []

    for item in results:
        if item is None:
            continue

        features, label = item

        feature_list.append(features)
        label_list.append(label)

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)

    return X, y


def create_xgboost_model():
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.6,
        reg_alpha=0.08,
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED
    )

    return model


def create_comparison_models():
    models = {
        "XGBoost": create_xgboost_model(),

        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(
                C=0.35,
                class_weight="balanced",
                max_iter=12000,
                tol=1e-3,
                random_state=SEED
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
                random_state=SEED
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            ))
        ])
    }

    return models


def save_cache(X_train, y_train, X_test, y_test):
    np.savez_compressed(
        FEATURE_CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("\nFeature cache saved to:")
    print(FEATURE_CACHE_PATH)


def load_cache():
    if not os.path.exists(FEATURE_CACHE_PATH):
        return None

    print("\nLoading feature cache...")

    data = np.load(FEATURE_CACHE_PATH)

    return (
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"]
    )


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cmap="Blues"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - 12 Class XGBoost Trash Classification")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    output_path = os.path.join(MODEL_DIR, "confusion_matrix.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nConfusion matrix saved to:")
    print(output_path)


def save_comparison_chart(results):
    model_names = list(results.keys())
    accuracies = [results[name] * 100 for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies)

    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Traditional Machine Learning Model Comparison")
    plt.ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            fontsize=10
        )

    output_path = os.path.join(MODEL_DIR, "model_comparison.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nModel comparison chart saved to:")
    print(output_path)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    cached_data = load_cache()

    if cached_data is not None:
        X_train, y_train, X_test, y_test = cached_data
    else:
        image_paths, labels = collect_dataset()

        print("\nTotal images:", len(image_paths))
        print("Total classes:", len(CLASSES))

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths,
            labels,
            test_size=0.2,
            random_state=SEED,
            stratify=labels
        )

        print("Train images:", len(train_paths))
        print("Test images:", len(test_paths))

        print("\nExtracting train features...")
        X_train, y_train = build_feature_matrix_parallel(
            train_paths,
            train_labels
        )

        print("\nExtracting test features...")
        X_test, y_test = build_feature_matrix_parallel(
            test_paths,
            test_labels
        )

        save_cache(X_train, y_train, X_test, y_test)

    print("\nX_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nTraining primary model: XGBoost")
    primary_model = create_xgboost_model()
    primary_model.fit(X_train, y_train)

    print("Training finished.")

    print("\nEvaluating model...")
    y_pred = primary_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("Accuracy (%):", accuracy * 100)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        zero_division=0
    ))

    comparison_results = {
        "XGBoost": float(accuracy)
    }

    if RUN_COMPARISON:
        models = create_comparison_models()
        comparison_results = {}

        print("\nTraining and comparing models...")

        for model_name, model in models.items():
            print(f"\nTraining: {model_name}")

            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            comparison_results[model_name] = float(acc)

            print(f"{model_name} Accuracy: {acc:.4f}")
            print(f"{model_name} Accuracy (%): {acc * 100:.2f}%")

        save_comparison_chart(comparison_results)

        results_path = os.path.join(MODEL_DIR, "comparison_results.json")

        with open(results_path, "w") as file:
            json.dump(comparison_results, file, indent=4)

        print("\nComparison results saved to:")
        print(results_path)

    model_package = {
        "model": primary_model,
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "feature_method": "HOG + Color Histogram + LBP + Shape Features",
        "classifier": "XGBoost",
        "accuracy": float(accuracy),
        "comparison_results": comparison_results
    }

    joblib.dump(model_package, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    save_confusion_matrix(y_test, y_pred)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42
OUT = Path("outputs")
FIG = Path("figures")

def generate_demo_data(n_samples=6000, n_spans=19, random_state=RANDOM_STATE):
    """Create synthetic raw measurements and labels for a reproducible classroom demo."""
    rng = np.random.default_rng(random_state)
    y = rng.binomial(1, 0.35, size=n_samples)  # 0=healthy, 1=potentially failing
    # Per-span deviations. Failing examples have a stronger average gain deviation.
    actual_gain = 20 + rng.normal(0, 0.45, (n_samples, n_spans)) - y[:, None] * 0.22
    target_gain = np.full((n_samples, n_spans), 20.0)
    actual_loss = 5 + rng.normal(0, 0.30, (n_samples, n_spans)) + y[:, None] * 0.18
    target_loss = np.full((n_samples, n_spans), 5.0)

    # Raw-to-feature calculations follow the equations described in the reference paper.
    x1_span = actual_gain - target_gain
    x2_span = target_loss - actual_loss
    X1 = x1_span.sum(axis=1)
    X2 = x2_span.sum(axis=1)

    # Add a small amount of label overlap/noise to make classification realistic.
    X1 += rng.normal(0, 0.65, n_samples)
    X2 += rng.normal(0, 0.45, n_samples)

    df = pd.DataFrame({
        "sample_id": np.arange(1, n_samples + 1),
        "aggregated_gain_difference_X1": X1,
        "aggregated_loss_difference_X2": X2,
        "label": y
    })
    df["label_name"] = df["label"].map({0: "healthy", 1: "potentially_failing"})
    return df

def train_evaluate(df):
    X = df[["aggregated_gain_difference_X1", "aggregated_loss_difference_X2"]]
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "GDA (Linear Discriminant Analysis)": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=RANDOM_STATE))
    }
    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append({"model": name, "accuracy": acc, "accuracy_percent": round(acc*100, 2)})
        predictions[name] = (model, pred)
    return X_train, X_test, y_train, y_test, models, results, predictions

def make_figures(df, y_test, predictions, X_test, results):
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Feature scatter plot
    plt.figure(figsize=(7, 5))
    for label, name in [(0, "Healthy"), (1, "Potentially failing")]:
        part = df[df.label == label]
        plt.scatter(part["aggregated_gain_difference_X1"], part["aggregated_loss_difference_X2"],
                    s=10, alpha=0.45, label=name)
    plt.xlabel("Aggregated gain difference (X1)")
    plt.ylabel("Aggregated loss difference (X2)")
    plt.title("Synthetic demo data: engineered features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"feature_scatter.png", dpi=160)
    plt.close()

    # Accuracy chart
    names = [r["model"] for r in results]
    vals = [r["accuracy_percent"] for r in results]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(names, vals)
    plt.ylabel("Test accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Model comparison on synthetic demo data")
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+1, f"{v:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(FIG/"model_accuracy.png", dpi=160)
    plt.close()

    # Confusion matrix for each model
    for name, (model, pred) in predictions.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Failing"])
        disp.plot(values_format="d")
        plt.title(f"Confusion matrix: {name}")
        plt.tight_layout()
        plt.savefig(FIG/f"confusion_{safe}.png", dpi=160)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train link-failure classifiers on synthetic demo data.")
    parser.add_argument("--samples", type=int, default=6000, help="Number of synthetic samples (default: 6000)")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = generate_demo_data(n_samples=args.samples)
    df.to_csv(OUT/"synthetic_demo_dataset.csv", index=False)
    X_train, X_test, y_train, y_test, models, results, predictions = train_evaluate(df)
    pd.DataFrame(results).to_csv(OUT/"model_results.csv", index=False)
    make_figures(df, y_test, predictions, X_test, results)

    print("=" * 68)
    print("LINK FAILURE PREDICTION — SYNTHETIC DEMONSTRATION")
    print("The scores below are demo results, not a reproduction of the paper.")
    print(f"Samples: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("=" * 68)
    print(pd.DataFrame(results).to_string(index=False))
    for name, (model, pred) in predictions.items():
        print(f"\n{name} — classification report")
        print(classification_report(y_test, pred, target_names=["healthy", "potentially_failing"], zero_division=0))
    print("\nSaved dataset, results, and plots in outputs/ and figures/.")

if __name__ == "__main__":
    main()

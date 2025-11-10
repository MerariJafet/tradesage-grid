import sys
from pathlib import Path

from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.train_real import df_features

OUTPUT_PATH = Path("walk_forward.txt")


def main() -> None:
    if df_features.empty:
        raise ValueError("Feature dataframe is empty; generate features before running walk-forward.")

    midpoint = len(df_features) // 2
    train = df_features.iloc[:midpoint]
    test = df_features.iloc[midpoint:]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = float((preds == y_test).mean())
    line = f"Walk-Forward Accuracy: {accuracy:.2%}"
    print(line)
    OUTPUT_PATH.write_text(line + "\n", encoding="ascii")


if __name__ == "__main__":
    main()

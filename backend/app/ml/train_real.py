import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.feature_engineering import engineer_features
from backend.app.ml.xgboost_model import train_xgboost

DATA_PATH = Path("data/btc_1m_12months.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "xgboost_edge.model"
DEFAULT_IMBALANCE = 0.1
DEFAULT_FUNDING_RATE = 0.0001


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_features(
    imbalance: float = DEFAULT_IMBALANCE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
    dataset_path: Path = DATA_PATH,
) -> pd.DataFrame:
    df = load_dataset(dataset_path)
    imbalance_series = pd.Series(imbalance, index=df.index)
    features = engineer_features(df, imbalance_series, funding_rate)
    non_numeric = features.select_dtypes(exclude=["number"]).columns
    if not non_numeric.empty:
        features = features.drop(columns=list(non_numeric))
    return features


def train_and_save(features: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = train_xgboost(features)
    model.save_model(MODEL_PATH)
    print(f"Modelo entrenado: {MODEL_PATH}")


df_features = build_features()


def main() -> None:
    train_and_save(df_features)


if __name__ == "__main__":
    main()

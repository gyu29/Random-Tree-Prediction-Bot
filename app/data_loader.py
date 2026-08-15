import glob
import os

import numpy as np
import pandas as pd

from app.config import FACTOR_CATEGORIES, TEST_ROOT, TRAIN_ROOT, VALIDATION_ROOT


class DataProcessor:
    @staticmethod
    def infer_symbol_from_path(file_path):
        filename = os.path.basename(file_path)
        for suffix in ("_historical.csv", "_historical.parquet", "_historical.xlsx", ".csv", ".parquet", ".xlsx"):
            if filename.endswith(suffix):
                return filename[: -len(suffix)]
        return os.path.splitext(filename)[0]

    @staticmethod
    def load_and_validate_data(file_path):
        try:
            df = pd.read_csv(file_path)
            if "Date" in df.columns:
                # Normalize mixed timezone strings into one UTC-aware index.
                df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
                df = df.dropna(subset=["Date"])
                df.set_index("Date", inplace=True)
            elif df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
                pass
            else:
                try:
                    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                    df = df[~df.index.isna()]
                except (ValueError, TypeError) as error:
                    print(f"Warning: could not convert index of {file_path} to datetime: {error}")

            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            if "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"].copy()

            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=required_columns)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df["symbol"] = DataProcessor.infer_symbol_from_path(file_path)
            return df
        except Exception as error:
            print(f"Error loading data from {file_path}: {error}")
            raise


def list_categories():
    return sorted(FACTOR_CATEGORIES.keys())


def category_train_dir(category):
    return os.path.join(TRAIN_ROOT, category)


def category_validation_dir(category):
    return os.path.join(VALIDATION_ROOT, category)


def category_test_dir(category):
    return os.path.join(TEST_ROOT, category)


_SPLIT_DIRS = {"train": category_train_dir, "validation": category_validation_dir, "test": category_test_dir}


def category_symbols_on_disk(category, split="train"):
    """Symbols that actually have a CSV on disk for this category and split."""
    directory = _SPLIT_DIRS[split](category)
    if not os.path.isdir(directory):
        return []
    return sorted(DataProcessor.infer_symbol_from_path(p) for p in glob.glob(os.path.join(directory, "*.csv")))


def category_for_symbol(symbol, default="market_beta"):
    """Best-effort category lookup for a raw ticker (case-insensitive, caret-tolerant)."""
    from app.config import SYMBOL_TO_CATEGORY

    return SYMBOL_TO_CATEGORY.get(symbol.upper().lstrip("^"), default)

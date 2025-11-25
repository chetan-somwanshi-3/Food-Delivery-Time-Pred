"""
dataset.py

Robust data ingestion and preprocessing pipeline for Food Delivery dataset.
- Written as a scikit-learn style transformer with fit/transform to avoid data leakage.
- Handles parsing, cleaning, range-checking, and imputation (numerical & categorical).
- Saves fitted artefacts so transform() on new (test) data won't leak training info.
- Avoids deprecated APIs and noisy warnings; uses standard sklearn/pandas/numpy.

Usage:
    proc = DatasetProcessor(seed=42)
    df_train_clean = proc.fit_transform(df_train)
    df_test_clean  = proc.transform(df_test)
    proc.save_artifacts('artifacts/')

"""

from __future__ import annotations
import os
import re
import logging
import warnings
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
import joblib

# Suppress only very noisy warnings but keep important ones
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetProcessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline that can be fit on training data and applied to test data.

    Key features:
    - Robust parsing and cleaning of strings/dates/times
    - Range checking for coordinates and ages
    - Iterative imputation for numeric features (fitted on training only)
    - Categorical imputation using training-distribution sampling (no leakage)
    - Helpful metadata saved for transform-time (value distributions etc.)
    """

    def __init__(
        self,
        seed: int = 42,
        coord_bounds: Optional[Dict[str, tuple]] = None,
        age_bounds: tuple = (18, 70),
        iterative_imputer_estimator: Optional[Any] = None,
    ):
        self.seed = seed
        self.random_state = check_random_state(seed)
        # sensible default bounds for India coordinates
        self.coord_bounds = coord_bounds or {
            "Restaurant_latitude": (6.0, 37.0),
            "Restaurant_longitude": (68.0, 98.0),
            "Delivery_location_latitude": (6.0, 37.0),
            "Delivery_location_longitude": (68.0, 98.0),
        }
        self.age_bounds = age_bounds
        self.iterative_imputer_estimator = (
            iterative_imputer_estimator or BayesianRidge()
        )

        # fitted artefacts
        self.num_imputer_: Optional[IterativeImputer] = None
        self.coord_imputer_: Optional[IterativeImputer] = None
        self.multiple_deliveries_imputer_: Optional[SimpleImputer] = None
        self.cat_value_probs_: Dict[str, np.ndarray] = {}
        self.cat_unique_values_: Dict[str, np.ndarray] = {}
        self.fitted_ = False
        
        # Artifacts for Time Category imputation (fitted during fit())
        self.time_category_mode_: Optional[str] = None
        self.time_category_labels_: List[str] = ['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night']


    # ------------------------- helpers -------------------------
    @staticmethod
    def _replace_various_na(x: pd.Series) -> pd.Series:
        # unify lots of textual NaN variants
        na_vals = set(["NaN", "NaN ", "NaN  ", "nan", "conditions NaN", "None", "none", "null", "Null"])
        return x.replace(list(na_vals), np.nan)

    @staticmethod
    def _safe_str_strip(x: pd.Series) -> pd.Series:
        return x.astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan})

    # ------------------------- core cleaning -------------------------
    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Do generic cleaning that does not require fitted artefacts.
        - Normalise whitespace
        - Replace common textual NaN markers
        - Safe casting where needed
        """
        df = df.copy()

        # Replace textual nans across all object columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace(['nan', 'NaN', 'NaN ', 'NaN  ', 'conditions NaN', 'None', 'none', 'null', 'Null'], np.nan)

        return df

    def parse_dates_times(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Order_Date: accept day-first, be permissive
        if 'Order_Date' in df.columns:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce', dayfirst=True)

        # Time_Orderd: Convert to string representation of time for later combination
        if 'Time_Orderd' in df.columns:
            # First, clean textual NaNs (done in basic_clean, but ensure safe handling)
            temp_time = df['Time_Orderd'].astype(str).str.strip()
            
            # Custom parsing function to handle HH:MM:SS or HH:MM formats
            def _to_time_str(x):
                if pd.isna(x) or str(x).lower() in ['nan', 'none']:
                    return np.nan
                try:
                    parts = str(x).split(':')
                    if len(parts) == 2:
                        # HH:MM format
                        hh, mm = int(parts[0]), int(parts[1])
                        return f"{hh:02d}:{mm:02d}:00"
                    if len(parts) >= 3:
                        # HH:MM:SS format
                        hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
                        return f"{hh:02d}:{mm:02d}:{ss:02d}"
                except Exception:
                    return np.nan
                return np.nan

            df['Time_Orderd'] = temp_time.apply(_to_time_str)

        return df

    def clean_strings_and_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Weatherconditions: remove stray word 'conditions'
        if 'Weatherconditions' in df.columns:
            df['Weatherconditions'] = (
                df['Weatherconditions']
                .astype(str)
                .str.replace('conditions', '', case=False, regex=False)
                .str.strip()
                .replace({'nan': np.nan, '': np.nan})
            )

        # Road_traffic_density and similar: title case and coerce empties to NaN
        for col in ['Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                )
                if df[col].dtype == object:
                    # Title case where meaningful
                    df[col] = df[col].where(df[col].isna(), df[col].str.title())

        # Delivery_person_Age: coerce to numeric and clamp unrealistic values to NaN
        if 'Delivery_person_Age' in df.columns:
            df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
            lb, ub = self.age_bounds
            df.loc[(df['Delivery_person_Age'] < lb) | (df['Delivery_person_Age'] > ub), 'Delivery_person_Age'] = np.nan

        # Delivery_person_Ratings: coerce and clamp to [1,5]
        if 'Delivery_person_Ratings' in df.columns:
            df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
            df.loc[(df['Delivery_person_Ratings'] < 1) | (df['Delivery_person_Ratings'] > 5), 'Delivery_person_Ratings'] = np.nan

        # multiple_deliveries to numeric
        if 'multiple_deliveries' in df.columns:
            df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

        # Time_taken(min) remove label text and coerce
        if 'Time_taken(min)' in df.columns:
            df['Time_taken(min)'] = df['Time_taken(min)'].astype(str).str.replace(r"\D+", "", regex=True)
            df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce')

        return df

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        coord_cols = [c for c in self.coord_bounds.keys() if c in df.columns]
        # to numeric
        for c in coord_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            lb, ub = self.coord_bounds[c]
            df.loc[(df[c] < lb) | (df[c] > ub), c] = np.nan
        return df

    # ------------------------- imputation (fit-time) -------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DatasetProcessor":
        """Fit imputers and categorical distributions on training data only.
        After fit, use transform on train/test.
        """
        logger.info("Fitting DatasetProcessor on training data...")
        df = X.copy()

        # run basic cleaning and parsing
        df = self.basic_clean(df)
        df = self.parse_dates_times(df)
        df = self.clean_strings_and_categories(df)
        df = self.clean_coordinates(df)

        # numeric imputer for selected numeric cols (keep simple set)
        num_cols = [c for c in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries'] if c in df.columns]
        if num_cols:
            # iterative imputer for correlated numeric features
            self.num_imputer_ = IterativeImputer(estimator=self.iterative_imputer_estimator, random_state=self.seed)
            # fit on the training set numeric columns
            self.num_imputer_.fit(df[num_cols])
            logger.info(f"Fitted iterative imputer on numeric columns: {num_cols}")

        # coordinates imputer (separate) - useful to treat coordinates jointly
        coord_cols = [c for c in self.coord_bounds.keys() if c in df.columns]
        if coord_cols:
            self.coord_imputer_ = IterativeImputer(random_state=self.seed)
            self.coord_imputer_.fit(df[coord_cols])
            logger.info(f"Fitted iterative imputer on coordinate columns: {coord_cols}")

        # fallback simple imputer for 'multiple_deliveries' if not present in iterative
        if 'multiple_deliveries' in df.columns and self.num_imputer_ is None:
            self.multiple_deliveries_imputer_ = SimpleImputer(strategy='median')
            self.multiple_deliveries_imputer_.fit(df[['multiple_deliveries']])

        # For categorical columns, learn value distributions for sampling without leakage.
        cat_cols = [c for c in ['City', 'Weatherconditions', 'Road_traffic_density', 'Festival', 'Type_of_order', 'Type_of_vehicle'] if c in df.columns]
        for c in cat_cols:
            vals = df[c].dropna().astype(str)
            if vals.empty:
                self.cat_unique_values_[c] = np.array([])
                self.cat_value_probs_[c] = np.array([])
            else:
                counts = vals.value_counts()
                uniques = counts.index.to_numpy()
                probs = (counts / counts.sum()).to_numpy()
                self.cat_unique_values_[c] = uniques
                self.cat_value_probs_[c] = probs
                logger.info(f"Learned categorical distribution for {c}: {len(uniques)} values")

        # FIT: Time Category Mode (needed for imputation in transform)
        if 'Order_Date' in df.columns and 'Time_Orderd' in df.columns:
            temp_df = self._derive_time_features(df.copy())
            if 'Order_Time_Category' in temp_df.columns:
                try:
                    self.time_category_mode_ = temp_df['Order_Time_Category'].mode(dropna=True)[0]
                except Exception:
                    self.time_category_mode_ = self.time_category_labels_[1] # Morning as safe fallback

        self.fitted_ = True
        return self
    
    def _derive_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to create time features (used in both fit and transform)"""
        if 'Order_Date' in df.columns and 'Time_Orderd' in df.columns:
            # Combine Date (datetime64[ns]) and Time (string H:M:S) to create Order_Placed
            df['Order_Placed'] = pd.to_datetime(
                df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time_Orderd'], errors='coerce'
            )
            
            # Extract hour
            df['Order_Hour'] = df['Order_Placed'].dt.hour

            # Categorize into bins
            bins = [0, 6, 12, 17, 21, 24]
            labels = self.time_category_labels_
            df['Order_Time_Category'] = pd.cut(df['Order_Hour'], bins=bins, labels=labels, right=False, ordered=False)

        return df


    def _sample_from_training_dist(self, col: str, n: int) -> np.ndarray:
        """Sample values for a categorical column according to training distribution.
        This avoids leaking global test info while providing realistic imputation.
        """
        uniques = self.cat_unique_values_.get(col, np.array([]))
        probs = self.cat_value_probs_.get(col, np.array([]))
        if uniques.size == 0:
            return np.array([np.nan] * n)
        return self.random_state.choice(uniques, size=n, p=probs, replace=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning + imputation learned in fit.
        Can be used on train (after fit) or test data.
        """
        if not self.fitted_:
            raise RuntimeError("DatasetProcessor must be fitted before calling transform(). Call fit() or fit_transform() first.")

        df = X.copy()
        df = self.basic_clean(df)
        df = self.parse_dates_times(df)
        df = self.clean_strings_and_categories(df)
        df = self.clean_coordinates(df)

        # Impute numeric columns
        num_cols = [c for c in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries'] if c in df.columns]
        if self.num_imputer_ is not None and len(num_cols) > 0:
            # ensure same column order
            df[num_cols] = self.num_imputer_.transform(df[num_cols])

        # Impute coordinates
        coord_cols = [c for c in self.coord_bounds.keys() if c in df.columns]
        if self.coord_imputer_ is not None and len(coord_cols) > 0:
            df[coord_cols] = self.coord_imputer_.transform(df[coord_cols])

        # Categorical sampling imputation (respect training distribution)
        for c in ['City', 'Weatherconditions', 'Road_traffic_density', 'Festival', 'Type_of_order', 'Type_of_vehicle']:
            if c in df.columns:
                missing_mask = df[c].isna()
                nmiss = missing_mask.sum()
                if nmiss > 0:
                    samples = self._sample_from_training_dist(c, nmiss)
                    df.loc[missing_mask, c] = samples

        # If multiple_deliveries still NaN, fill median learned from training via simple imputer
        if 'multiple_deliveries' in df.columns and df['multiple_deliveries'].isna().any():
            if self.multiple_deliveries_imputer_ is not None:
                df['multiple_deliveries'] = self.multiple_deliveries_imputer_.transform(df[['multiple_deliveries']])
            else:
                # fallback to median from non-missing values
                med = df['multiple_deliveries'].median()
                df['multiple_deliveries'] = df['multiple_deliveries'].fillna(med)

        # --- Feature Extraction and Imputation for Time ---
        if 'Order_Date' in df.columns and 'Time_Orderd' in df.columns:
            
            # 1. Derive features: Order_Placed, Order_Hour, Order_Time_Category
            df = self._derive_time_features(df)
            
            # 2. Impute Order_Time_Category using the mode learned during fit()
            if 'Order_Time_Category' in df.columns and df['Order_Time_Category'].isna().any():
                mode_label = self.time_category_mode_
                if mode_label:
                    df['Order_Time_Category'] = df['Order_Time_Category'].fillna(mode_label)
                
            # 3. Drop irrelevant temporary and source features
            df.drop(columns=['Order_Placed', 'Order_Hour', 'Time_Orderd'], inplace=True, errors='ignore')
            
            # Optional: Impute Order_Date if needed (using mode/ffill, but usually not ideal for time series data)
            # Keeping Order_Date as is for now, as it's typically used for splitting/time series analysis.

        # Final tidy: ensure types
        # Convert Time_taken to integer if present
        if 'Time_taken(min)' in df.columns:
            df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce').round().astype('Int64')

        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------- persistence -------------------------
    def save_artifacts(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # save all fitted objects
        if self.num_imputer_ is not None:
            joblib.dump(self.num_imputer_, os.path.join(path, 'num_imputer.joblib'))
        if self.coord_imputer_ is not None:
            joblib.dump(self.coord_imputer_, os.path.join(path, 'coord_imputer.joblib'))
        if self.multiple_deliveries_imputer_ is not None:
            joblib.dump(self.multiple_deliveries_imputer_, os.path.join(path, 'mult_deliv_imputer.joblib'))

        joblib.dump({
            'cat_unique_values': self.cat_unique_values_,
            'cat_value_probs': self.cat_value_probs_,
            'seed': self.seed,
            'time_category_mode': self.time_category_mode_,
        }, os.path.join(path, 'artifacts.joblib'))
        logger.info(f"Saved artifacts to {path}")

    def load_artifacts(self, path: str) -> None:
        # load objects if available
        num_path = os.path.join(path, 'num_imputer.joblib')
        coord_path = os.path.join(path, 'coord_imputer.joblib')
        mult_path = os.path.join(path, 'mult_deliv_imputer.joblib')
        art_path = os.path.join(path, 'artifacts.joblib')

        if os.path.exists(num_path):
            self.num_imputer_ = joblib.load(num_path)
        if os.path.exists(coord_path):
            self.coord_imputer_ = joblib.load(coord_path)
        if os.path.exists(mult_path):
            self.multiple_deliveries_imputer_ = joblib.load(mult_path)
        if os.path.exists(art_path):
            data = joblib.load(art_path)
            self.cat_unique_values_ = data.get('cat_unique_values', {})
            self.cat_value_probs_ = data.get('cat_value_probs', {})
            self.seed = data.get('seed', self.seed)
            self.time_category_mode_ = data.get('time_category_mode', self.time_category_labels_[1])
            self.random_state = check_random_state(self.seed)

        self.fitted_ = True
        logger.info(f"Loaded artifacts from {path}")


# ------------------------- small CLI helper (optional) -------------------------
# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='Fit and save preprocessing artifacts for Food Delivery dataset')
#     parser.add_argument('--input', type=str, required=True, help='Path to raw CSV file')
#     parser.add_argument('--outdir', type=str, required=True, help='Directory to save artifacts and a cleaned sample CSV')
#     args = parser.parse_args()

#     # Create dummy data for testing if a real file isn't available
#     if not os.path.exists(args.input):
#         logger.warning(f"Input file not found: {args.input}. Creating dummy data.")
#         data = {
#             'Order_Date': ['15-03-2022', '16-03-2022', '17-03-2022', '18-03-2022', '19-03-2022'],
#             'Time_Orderd': ['12:00:00', '19:30', '1:00:0', '15:15:00', 'NaN'],
#             'Delivery_person_Age': [25, np.nan, 30, 45, 22],
#             'multiple_deliveries': [1, 2, np.nan, 1, 0],
#             'City': ['Metropolitian', 'Urban', 'Metropolitian', 'Urban', 'Rural'],
#             'Weatherconditions': ['Sunny', 'Storm', 'Sunny', 'Fog', 'NaN'],
#             'Road_traffic_density': ['High', 'Medium', 'Low', 'High', 'NaN'],
#             'Time_taken(min)': ['30 min', '25 min', '20 min', '35 min', '15 min'],
#             'Restaurant_latitude': [20.0, 30.0, 10.0, 25.0, 15.0],
#             'Restaurant_longitude': [80.0, 90.0, 70.0, 75.0, 85.0],
#             'Delivery_location_latitude': [20.1, 30.1, 10.1, 25.1, 15.1],
#             'Delivery_location_longitude': [80.1, 90.1, 70.1, 75.1, 85.1],
#             'Delivery_person_Ratings': [4.5, 4.0, 5.0, 3.5, 4.2]
#         }
#         raw = pd.DataFrame(data)
#     else:
#         raw = pd.read_csv(args.input)
        
#     proc = DatasetProcessor(seed=42)
#     cleaned = proc.fit_transform(raw)
    
#     os.makedirs(args.outdir, exist_ok=True)
#     cleaned.to_csv(os.path.join(args.outdir, 'cleaned_sample.csv'), index=False)
#     proc.save_artifacts(args.outdir)
#     logger.info('Done. Cleaned sample and artifacts saved.')
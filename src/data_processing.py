import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load raw transaction data"""
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df


def create_time_features(df):
    """Extract time-based features from TransactionStartTime"""
    df = df.copy()
    df['Hour']      = df['TransactionStartTime'].dt.hour
    df['Day']       = df['TransactionStartTime'].dt.day
    df['Month']     = df['TransactionStartTime'].dt.month
    df['Year']      = df['TransactionStartTime'].dt.year
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    return df


def encode_categoricals(df):
    """Label encode categorical columns"""
    df = df.copy()
    cat_cols = ['ProviderId', 'ProductId', 'ProductCategory',
                'ChannelId', 'CurrencyCode']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def create_aggregate_features(df):
    """Aggregate transaction data to customer level with RFM features"""
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    agg = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime',
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum'),
        Avg_Amount=('Amount', 'mean'),
        Std_Amount=('Amount', 'std'),
        Max_Amount=('Amount', 'max'),
        Min_Amount=('Amount', 'min'),
        Total_Value=('Value', 'sum'),
        Avg_Value=('Value', 'mean'),
        Total_Fraud=('FraudResult', 'sum'),
        Fraud_Rate=('FraudResult', 'mean'),
        Unique_Products=('ProductId', 'nunique'),
        Unique_Channels=('ChannelId', 'nunique'),
        Unique_Categories=('ProductCategory', 'nunique'),
        Avg_Hour=('Hour', 'mean'),
        Avg_DayOfWeek=('DayOfWeek', 'mean'),
        Weekend_Ratio=('IsWeekend', 'mean'),
    ).reset_index()

    # Fill std NaN for customers with 1 transaction
    agg['Std_Amount'] = agg['Std_Amount'].fillna(0)
    return agg


def compute_woe_iv(df, feature, target, bins=10):
    """
    Compute Weight of Evidence (WoE) and Information Value (IV)
    for a numerical feature against a binary target.

    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = Sum((Dist Events - Dist Non-Events) * WoE)

    Basel II context: WoE transformation makes logistic regression
    coefficients directly interpretable as risk weights.
    """
    df = df.copy()

    # Bin the feature
    try:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    except Exception:
        df['bin'] = pd.cut(df[feature], bins=bins)

    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events     = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    grouped['dist_events']     = grouped['events'] / (total_events + 1e-10)
    grouped['dist_non_events'] = grouped['non_events'] / (total_non_events + 1e-10)

    # Avoid log(0)
    grouped['dist_events']     = grouped['dist_events'].replace(0, 1e-10)
    grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 1e-10)

    grouped['WoE'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['IV']  = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['WoE']

    iv = grouped['IV'].sum()
    return grouped[['WoE', 'IV']], iv


def compute_all_iv(df, features, target):
    """Compute IV for all features and rank by predictive power"""
    iv_results = {}
    for feature in features:
        try:
            _, iv = compute_woe_iv(df, feature, target)
            iv_results[feature] = round(iv, 4)
        except Exception:
            iv_results[feature] = 0.0

    iv_df = pd.DataFrame.from_dict(
        iv_results, orient='index', columns=['IV']
    ).sort_values('IV', ascending=False)

    iv_df['Predictive_Power'] = iv_df['IV'].apply(
        lambda x: 'Strong' if x > 0.3
        else 'Medium' if x > 0.1
        else 'Weak' if x > 0.02
        else 'Useless'
    )
    return iv_df


def normalize_features(df, feature_cols):
    """
    Standardize numerical features to mean=0, std=1.
    Returns normalized DataFrame and fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler


def build_customer_features(df):
    """
    Full feature engineering pipeline:
    1. Extract time features
    2. Encode categoricals
    3. Aggregate to customer level
    Returns customer-level DataFrame
    """
    df = create_time_features(df)
    df = encode_categoricals(df)
    customer_df = create_aggregate_features(df)
    return customer_df


def compute_rfm(df):
    """Compute raw RFM values for clustering"""
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime',
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    return rfm


if __name__ == '__main__':
    import os
    os.chdir(r'C:\Users\bamla\OneDrive\Desktop\credit-risk-model')

    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    # Step 1: Load
    print("\n[1] Loading raw data...")
    df = load_data('data/raw/data.csv')
    print(f"    Raw data shape: {df.shape}")

    # Step 2: Build customer features
    print("\n[2] Building customer-level features...")
    customer_df = build_customer_features(df)
    print(f"    Customer features shape: {customer_df.shape}")
    print(f"    Columns: {customer_df.columns.tolist()}")

    # Step 3: IV Analysis (using Fraud_Rate as proxy target for now)
    print("\n[3] Computing Information Value (WoE/IV)...")
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Amount',
                    'Std_Amount', 'Max_Amount', 'Unique_Products',
                    'Unique_Channels', 'Fraud_Rate']
    iv_results = compute_all_iv(customer_df, feature_cols, 'Total_Fraud')
    print(f"\n    IV Results:")
    print(iv_results.to_string())

    # Step 4: Normalize
    print("\n[4] Normalizing features...")
    num_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Amount',
                'Std_Amount', 'Max_Amount', 'Min_Amount',
                'Total_Value', 'Avg_Value', 'Unique_Products',
                'Unique_Channels', 'Unique_Categories']
    customer_scaled, scaler = normalize_features(customer_df, num_cols)
    print(f"    Normalized {len(num_cols)} features")
    print(f"    Sample means after scaling: {customer_scaled[num_cols].mean().round(3).values[:3]}")

    # Step 5: Save
    print("\n[5] Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    customer_df.to_csv('data/processed/customer_features.csv', index=False)
    customer_scaled.to_csv('data/processed/customer_features_scaled.csv', index=False)
    print("    Saved customer_features.csv")
    print("    Saved customer_features_scaled.csv")

    print("\n✅ Feature engineering pipeline complete!")
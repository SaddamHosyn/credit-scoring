import os
import gc
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("feature-engineering")

RAW_DIR = "home-credit-default-risk"
OUTPUT_DIR = "output/processed_data"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def reduce_mem_usage(df):
    """
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def get_bureau_features():
    logger.info("Processing bureau and bureau_balance...")
    bb_path = os.path.join(RAW_DIR, 'bureau_balance.csv')
    bureau_path = os.path.join(RAW_DIR, 'bureau.csv')
    
    if not os.path.exists(bureau_path):
        logger.warning("bureau.csv not found. Skipping bureau features.")
        return pd.DataFrame()

    if os.path.exists(bb_path):
        bb = pd.read_csv(bb_path)
        bb = reduce_mem_usage(bb)
        bb = pd.get_dummies(bb, columns=['STATUS'], dummy_na=False)
        
        # Aggregate by SK_ID_BUREAU
        bb_agg = bb.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            **{c: ['mean', 'sum'] for c in bb.columns if c.startswith('STATUS_')}
        })
        bb_agg.columns = pd.Index([f'BB_{c[0]}_{c[1].upper()}' for c in bb_agg.columns.tolist()])
        
        # Merge back to bureau
        bureau = pd.read_csv(bureau_path)
        bureau = reduce_mem_usage(bureau)
        bureau = bureau.merge(bb_agg, how='left', on='SK_ID_BUREAU')
        del bb, bb_agg
        gc.collect()
    else:
        bureau = pd.read_csv(bureau_path)
        bureau = reduce_mem_usage(bureau)

    # OHE categorical columns in bureau
    categorical_cols = bureau.select_dtypes(include=['object']).columns.tolist()
    bureau = pd.get_dummies(bureau, columns=categorical_cols, dummy_na=False)

    # Core aggregations by SK_ID_CURR
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
    }
    
    # include BB columns in num_aggregations
    for col in bureau.columns:
        if col.startswith('BB_'):
            num_aggregations[col] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(num_aggregations)
    bureau_agg.columns = pd.Index([f'BUREAU_{c[0]}_{c[1].upper()}' for c in bureau_agg.columns.tolist()])
    
    # Count of credits
    bureau_agg['BUREAU_CREDITS_COUNT'] = bureau.groupby('SK_ID_CURR').size().astype(np.int32)
    
    del bureau
    gc.collect()
    return bureau_agg

def get_previous_app_features():
    logger.info("Processing previous applications...")
    prev_path = os.path.join(RAW_DIR, 'previous_application.csv')
    if not os.path.exists(prev_path):
        logger.warning("previous_application.csv not found. Skipping prev features.")
        return pd.DataFrame()

    prev = pd.read_csv(prev_path)
    prev = reduce_mem_usage(prev)
    
    # Replace some anomalies in DAYS_
    for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev[col].replace(365243, np.nan, inplace=True)
        
    # OHE categorical columns
    categorical_cols = prev.select_dtypes(include=['object']).columns.tolist()
    prev = pd.get_dummies(prev, columns=categorical_cols, dummy_na=False)
    
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    # Add status and type sum/means
    for col in prev.columns:
        if col.startswith('NAME_CONTRACT_STATUS_') or col.startswith('NAME_CONTRACT_TYPE_') or col.startswith('CODE_REJECT_REASON_'):
            num_aggregations[col] = ['mean', 'sum']
            
    prev_agg = prev.groupby('SK_ID_CURR').agg(num_aggregations)
    prev_agg.columns = pd.Index([f'PREV_{c[0]}_{c[1].upper()}' for c in prev_agg.columns.tolist()])
    
    # Count of prev applications
    prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size().astype(np.int32)
    
    del prev
    gc.collect()
    return prev_agg

def get_pos_cash_features():
    logger.info("Processing POS_CASH_balance...")
    pos_path = os.path.join(RAW_DIR, 'POS_CASH_balance.csv')
    if not os.path.exists(pos_path):
        logger.warning("POS_CASH_balance.csv not found. Skipping POS features.")
        return pd.DataFrame()

    pos = pd.read_csv(pos_path)
    pos = reduce_mem_usage(pos)
    pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na=False)
    
    num_aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'CNT_INSTALMENT': ['mean', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'max'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for col in pos.columns:
        if col.startswith('NAME_CONTRACT_STATUS_'):
            num_aggregations[col] = ['mean']
            
    pos_agg = pos.groupby('SK_ID_CURR').agg(num_aggregations)
    pos_agg.columns = pd.Index([f'POS_{c[0]}_{c[1].upper()}' for c in pos_agg.columns.tolist()])
    
    del pos
    gc.collect()
    return pos_agg

def get_installments_features():
    logger.info("Processing installments_payments...")
    ins_path = os.path.join(RAW_DIR, 'installments_payments.csv')
    if not os.path.exists(ins_path):
        logger.warning("installments_payments.csv not found. Skipping installments features.")
        return pd.DataFrame()

    ins = pd.read_csv(ins_path)
    ins = reduce_mem_usage(ins)
    
    # Compute lateness (positive = late)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DPD'] = ins['DPD'].clip(lower=0)
    
    # Compute underpayment
    ins['UNDERPAY'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['UNDERPAY'] = ins['UNDERPAY'].clip(lower=0)
    
    num_aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['max', 'mean', 'sum'],
        'DPD': ['max', 'mean', 'sum'],
        'UNDERPAY': ['max', 'mean', 'sum']
    }
    
    ins_agg = ins.groupby('SK_ID_CURR').agg(num_aggregations)
    ins_agg.columns = pd.Index([f'INS_{c[0]}_{c[1].upper()}' for c in ins_agg.columns.tolist()])
    ins_agg['INS_PAYMENT_COUNT'] = ins.groupby('SK_ID_CURR').size().astype(np.int32)
    
    del ins
    gc.collect()
    return ins_agg

def get_credit_card_features():
    logger.info("Processing credit_card_balance...")
    cc_path = os.path.join(RAW_DIR, 'credit_card_balance.csv')
    if not os.path.exists(cc_path):
        logger.warning("credit_card_balance.csv not found. Skipping credit card features.")
        return pd.DataFrame()

    cc = pd.read_csv(cc_path)
    cc = reduce_mem_usage(cc)
    cc = pd.get_dummies(cc, columns=['NAME_CONTRACT_STATUS'], dummy_na=False)
    
    # Credit card utilization
    cc['UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
    
    num_aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'AMT_BALANCE': ['max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean'],
        'UTILIZATION': ['max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    cc_agg = cc.groupby('SK_ID_CURR').agg(num_aggregations)
    cc_agg.columns = pd.Index([f'CC_{c[0]}_{c[1].upper()}' for c in cc_agg.columns.tolist()])
    
    del cc
    gc.collect()
    return cc_agg

def build_secondary_features_lookup():
    logger.info("=== STARTING AGGREGATION OF SECONDARY TABLES ===")
    
    # Get all features
    bureau = get_bureau_features()
    prev = get_previous_app_features()
    pos = get_pos_cash_features()
    ins = get_installments_features()
    cc = get_credit_card_features()
    
    # Merge sequentially to minimize memory footprint
    logger.info("Merging secondary features...")
    lookup = None
    
    for df in [bureau, prev, pos, ins, cc]:
        if df.empty:
            continue
        if lookup is None:
            lookup = df
        else:
            lookup = lookup.join(df, how='outer')
            del df
            gc.collect()
            
    if lookup is not None:
        lookup = reduce_mem_usage(lookup)
        lookup.index.name = 'SK_ID_CURR'
        lookup_path = os.path.join(OUTPUT_DIR, 'secondary_features_lookup.csv')
        logger.info(f"Saving secondary features lookup ({lookup.shape}) to {lookup_path}...")
        lookup.to_csv(lookup_path)
    else:
        lookup = pd.DataFrame(index=pd.Index([], name='SK_ID_CURR'))
        
    return lookup

def preprocess_application_data(df, is_train=True):
    logger.info(f"Preprocessing application {'train' if is_train else 'test'} basic features...")
    
    # Anomaly DAYS_EMPLOYED
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'].isna().astype(int)
    
    # Ratios
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    # EXT_SOURCE aggregates
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCES_PROD'] = df[ext_cols].prod(axis=1)
    
    return df

def main():
    # 1. Build and save the secondary features lookup
    lookup = build_secondary_features_lookup()
    
    # 2. Process application_train
    logger.info("Loading application_train.csv...")
    train = pd.read_csv(os.path.join(RAW_DIR, 'application_train.csv'))
    train = reduce_mem_usage(train)
    train = preprocess_application_data(train, is_train=True)
    
    # 3. Process application_test
    logger.info("Loading application_test.csv...")
    test = pd.read_csv(os.path.join(RAW_DIR, 'application_test.csv'))
    test = reduce_mem_usage(test)
    test = preprocess_application_data(test, is_train=False)
    
    # 4. Merge secondary features
    if not lookup.empty:
        logger.info("Merging secondary features with train & test applications...")
        train = train.merge(lookup, on='SK_ID_CURR', how='left')
        test = test.merge(lookup, on='SK_ID_CURR', how='left')
        
    del lookup
    gc.collect()
    
    # 5. Handle Categorical Encoding (Combined OHE to ensure aligned columns)
    logger.info("Applying Categorical Encoding...")
    train_target = train['TARGET'].copy()
    train_features = train.drop(columns=['TARGET']).copy()
    
    combined = pd.concat([train_features, test], axis=0, ignore_index=True)
    del train, test
    gc.collect()
    
    # OHE all categorical fields
    cat_cols = combined.select_dtypes(include=['object']).columns.tolist()
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)
    
    # Re-split
    train_rows = len(train_features)
    train_eng = combined.iloc[:train_rows, :].copy()
    test_eng = combined.iloc[train_rows:, :].copy()
    
    train_eng['TARGET'] = train_target.values
    
    # Align and clean names (replacing spaces, special symbols)
    import re
    def clean_col_name(col):
        new_col = re.sub(r"[^A-Za-z0-9_]+", "", str(col))
        return new_col if new_col != "" else "feature"
        
    train_eng.columns = [clean_col_name(c) for c in train_eng.columns]
    test_eng.columns = [clean_col_name(c) for c in test_eng.columns]
    
    # Save final engineered tables
    train_out_path = os.path.join(OUTPUT_DIR, 'application_train_engineered.csv')
    test_out_path = os.path.join(OUTPUT_DIR, 'application_test_engineered.csv')
    
    logger.info(f"Saving final train engineered ({train_eng.shape}) to {train_out_path}...")
    train_eng.to_csv(train_out_path, index=False)
    
    logger.info(f"Saving final test engineered ({test_eng.shape}) to {test_out_path}...")
    test_eng.to_csv(test_out_path, index=False)
    
    logger.info("=== FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()

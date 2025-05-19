"""
Data validation script for checking the quality of saved parquet files.
"""

import polars as pl
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_validator')

def check_dataframe(df, name):
    """Check a DataFrame for common data quality issues."""
    logger.info(f"\nChecking {name}:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")
    
    # Check for nulls
    null_counts = df.null_count()
    if not null_counts.is_empty() and null_counts.select(pl.all().sum()).row(0)[0] > 0:
        logger.warning(f"Found null values in {name}:")
        logger.warning(null_counts)
    
    # Check for infinite values in numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64]]
    if numeric_cols:
        inf_counts = df.select([
            pl.col(col).filter(pl.col(col).is_infinite()).count().alias(f"{col}_inf")
            for col in numeric_cols
        ])
        if not inf_counts.is_empty() and inf_counts.select(pl.all().sum()).row(0)[0] > 0:
            logger.warning(f"Found infinite values in {name}:")
            logger.warning(inf_counts)
    
    # Check date ranges
    if 'date' in df.columns:
        date_range = df.select([
            pl.col('date').min().alias('min_date'),
            pl.col('date').max().alias('max_date')
        ])
        logger.info(f"Date range: {date_range}")
    
    # Check for duplicate date-symbol pairs
    if 'date' in df.columns and 'symbol' in df.columns:
        duplicates = df.group_by(['date', 'symbol']).count().filter(pl.col('count') > 1)
        if not duplicates.is_empty():
            logger.warning(f"Found duplicate date-symbol pairs in {name}:")
            logger.warning(duplicates)
    
    # Check value ranges for numeric columns
    for col in numeric_cols:
        stats = df.select([
            pl.col(col).min().alias('min'),
            pl.col(col).max().alias('max'),
            pl.col(col).mean().alias('mean'),
            pl.col(col).std().alias('std')
        ])
        logger.info(f"\nStats for {col}:")
        logger.info(stats)

def main():
    """Main function to check all data files."""
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist!")
        return
    
    # Check each parquet file
    files = {
        'returns_df': 'returns_df.parquet',
        'mkt_cap_df': 'mkt_cap_df.parquet',
        'sector_df': 'sector_df.parquet',
        'value_df': 'value_df.parquet'
    }
    
    for name, file in files.items():
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist!")
            continue
            
        try:
            df = pl.read_parquet(file_path)
            if df.is_empty():
                logger.warning(f"{name} is empty!")
                continue
            check_dataframe(df, name)
        except Exception as e:
            logger.error(f"Error checking {name}: {str(e)}")

if __name__ == "__main__":
    main() 
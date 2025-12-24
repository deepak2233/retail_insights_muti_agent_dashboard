"""
Production-Grade Data Ingestion Pipeline
Handles real Amazon/E-commerce sales data with robust error handling
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Production pipeline for ingesting and processing e-commerce sales data"""
    
    def __init__(self, data_dir: str = "data/Sales Dataset"):
        self.data_dir = Path(data_dir)
        self.processed_data = None
        self.stats = {}
        
    def ingest_all_data(self) -> pd.DataFrame:
        """
        Main entry point: Ingest and process all data files
        Returns unified, cleaned DataFrame
        """
        logger.info("🚀 Starting production data ingestion pipeline...")
        
        try:
            # Step 1: Load Amazon sales data (main dataset)
            amazon_df = self._load_amazon_sales()
            
            # Step 2: Load supplementary data
            sale_report_df = self._load_sale_report()
            international_df = self._load_international_sales()
            
            # Step 3: Process and standardize
            amazon_processed = self._process_amazon_data(amazon_df)
            international_processed = self._process_international_data(international_df)
            
            # Step 4: Merge datasets
            unified_df = self._merge_datasets(amazon_processed, international_processed, sale_report_df)
            
            # Step 5: Data quality checks
            unified_df = self._data_quality_checks(unified_df)
            
            # Step 6: Feature engineering
            unified_df = self._feature_engineering(unified_df)
            
            # Step 7: Save processed data
            self._save_processed_data(unified_df)
            
            self.processed_data = unified_df
            
            logger.info(f"✅ Pipeline complete! Processed {len(unified_df):,} records")
            return unified_df
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _load_amazon_sales(self) -> pd.DataFrame:
        """Load Amazon sales report with error handling"""
        try:
            file_path = self.data_dir / "Amazon Sale Report.csv"
            logger.info(f"📂 Loading Amazon sales data from {file_path}...")
            
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"   ✅ Loaded {len(df):,} records")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading Amazon data: {e}")
            raise
    
    def _load_sale_report(self) -> Optional[pd.DataFrame]:
        """Load sale report with fallback"""
        try:
            file_path = self.data_dir / "Sale Report.csv"
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"   ✅ Loaded sale report: {len(df):,} records")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load sale report: {e}")
            return None
    
    def _load_international_sales(self) -> Optional[pd.DataFrame]:
        """Load international sales with fallback"""
        try:
            file_path = self.data_dir / "International sale Report.csv"
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"   ✅ Loaded international sales: {len(df):,} records")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load international sales: {e}")
            return None
    
    def _process_amazon_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize Amazon sales data"""
        logger.info("🔄 Processing Amazon sales data...")
        
        df = df.copy()
        
        # 1. Drop unnecessary columns
        columns_to_drop = ['index', 'Unnamed: 22']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # 2. Standardize column names
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            'Order ID': 'order_id',
            'Date': 'date',
            'Status': 'status',
            'Fulfilment': 'fulfilment',
            'Sales Channel': 'sales_channel',
            'ship-service-level': 'service_level',
            'Style': 'style',
            'SKU': 'sku',
            'Category': 'category',
            'Size': 'size',
            'ASIN': 'asin',
            'Courier Status': 'courier_status',
            'Qty': 'quantity',
            'currency': 'currency',
            'Amount': 'amount',
            'ship-city': 'city',
            'ship-state': 'state',
            'ship-postal-code': 'postal_code',
            'ship-country': 'country',
            'promotion-ids': 'promotions',
            'B2B': 'is_b2b',
            'fulfilled-by': 'fulfilled_by'
        })
        
        # 3. Parse dates with multiple format support
        df['date'] = self._parse_dates(df['date'])
        
        # 4. Handle missing amounts
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # 5. Clean quantity
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
        
        # 6. Standardize status values
        df['status'] = df['status'].str.strip().fillna('Unknown')
        
        # 7. Extract year, month, quarter
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B')
        df['quarter'] = df['date'].dt.quarter
        df['quarter_name'] = 'Q' + df['quarter'].astype(str)
        
        # 8. Clean categorical data
        df['category'] = df['category'].str.strip().fillna('Unknown')
        df['state'] = df['state'].str.strip().fillna('Unknown')
        df['city'] = df['city'].str.strip().fillna('Unknown')
        
        # 9. Calculate revenue (only for completed orders)
        df['revenue'] = df.apply(
            lambda row: row['amount'] if row['status'] not in ['Cancelled', 'Returned'] else 0,
            axis=1
        )
        
        # 10. Add data source
        df['data_source'] = 'Amazon India'
        
        logger.info(f"   ✅ Processed Amazon data: {len(df):,} records")
        return df
    
    def _process_international_data(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Process international sales data"""
        if df is None:
            return None
            
        logger.info("🔄 Processing international sales data...")
        
        df = df.copy()
        
        # Standardize columns
        df = df.rename(columns={
            'DATE': 'date',
            'CUSTOMER': 'customer',
            'Style': 'style',
            'SKU': 'sku',
            'Size': 'size',
            'PCS': 'quantity',
            'RATE': 'unit_price',
            'GROSS AMT': 'amount'
        })
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean numeric fields
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
        
        # Add metadata
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['revenue'] = df['amount']
        df['data_source'] = 'International'
        df['status'] = 'Shipped'
        df['country'] = 'International'
        
        logger.info(f"   ✅ Processed international data: {len(df):,} records")
        return df
    
    def _merge_datasets(self, amazon_df: pd.DataFrame, 
                        international_df: Optional[pd.DataFrame],
                        sale_report_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets with common schema"""
        logger.info("🔗 Merging datasets...")
        
        # Common columns for all datasets
        common_cols = ['date', 'sku', 'quantity', 'amount', 'revenue', 'year', 'month', 'quarter', 'data_source']
        
        # Start with Amazon data (main dataset)
        unified = amazon_df.copy()
        
        # Add international data if available
        if international_df is not None:
            # Ensure common columns exist
            for col in common_cols:
                if col not in international_df.columns:
                    international_df[col] = None
            
            # Append international data
            unified = pd.concat([unified, international_df], ignore_index=True, sort=False)
            logger.info(f"   ✅ Merged international data")
        
        logger.info(f"   ✅ Final unified dataset: {len(unified):,} records")
        return unified
    
    def _parse_dates(self, date_series: pd.Series) -> pd.Series:
        """Parse dates with multiple format support and error handling"""
        
        # Try multiple date formats
        formats = ['%m-%d-%y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
        
        parsed_dates = None
        for fmt in formats:
            try:
                parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
                if parsed.notna().sum() > 0:
                    if parsed_dates is None:
                        parsed_dates = parsed
                    else:
                        # Fill NaT values with newly parsed dates
                        parsed_dates = parsed_dates.fillna(parsed)
            except:
                continue
        
        # Final fallback: flexible parsing
        if parsed_dates is None or parsed_dates.isna().sum() > 0:
            parsed_dates = pd.to_datetime(date_series, errors='coerce')
        
        # Fill remaining NaT with a default date
        default_date = pd.Timestamp('2022-01-01')
        parsed_dates = parsed_dates.fillna(default_date)
        
        return parsed_dates
    
    def _data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data quality checks and fixes"""
        logger.info("🔍 Running data quality checks...")
        
        initial_count = len(df)
        
        # 1. Remove duplicates
        df = df.drop_duplicates(subset=['order_id'] if 'order_id' in df.columns else None)
        if len(df) < initial_count:
            logger.info(f"   ℹ️  Removed {initial_count - len(df)} duplicate records")
        
        # 2. Fix negative amounts
        negative_amounts = (df['amount'] < 0).sum()
        if negative_amounts > 0:
            logger.warning(f"   ⚠️  Found {negative_amounts} negative amounts, setting to 0")
            df.loc[df['amount'] < 0, 'amount'] = 0
        
        # 3. Fix invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"   ⚠️  Found {invalid_dates} invalid dates")
        
        # 4. Validate quantity
        df['quantity'] = df['quantity'].clip(lower=0)
        
        # 5. Check for outliers in amount
        if len(df) > 0:
            q99 = df['amount'].quantile(0.99)
            outliers = (df['amount'] > q99 * 3).sum()
            if outliers > 0:
                logger.warning(f"   ⚠️  Found {outliers} potential outliers in amount")
        
        logger.info(f"   ✅ Data quality checks complete")
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analytics"""
        logger.info("⚡ Engineering features...")
        
        # 1. Order value category
        df['order_value_category'] = pd.cut(
            df['amount'],
            bins=[0, 300, 600, 1000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # 2. Is cancelled
        df['is_cancelled'] = df['status'].str.contains('Cancel', case=False, na=False)
        
        # 3. Is shipped
        df['is_shipped'] = df['status'].str.contains('Shipped', case=False, na=False)
        
        # 4. Has promotion
        if 'promotions' in df.columns:
            df['has_promotion'] = df['promotions'].notna() & (df['promotions'] != '')
        
        # 5. Day of week
        df['day_of_week'] = df['date'].dt.day_name()
        
        # 6. Is weekend
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
        
        # 7. Profit margin (estimate: 30% for successful orders)
        df['estimated_profit'] = df['revenue'] * 0.30
        
        logger.info(f"   ✅ Added {7} new features")
        return df
    
    def _save_processed_data(self, df: pd.DataFrame):
        """Save processed data in multiple formats"""
        try:
            # Save as CSV
            csv_path = "data/processed_sales_data.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"   💾 Saved CSV: {csv_path}")
            
            # Save as Parquet (compressed)
            parquet_path = "data/processed_sales_data.parquet"
            df.to_parquet(parquet_path, index=False, compression='snappy')
            logger.info(f"   💾 Saved Parquet: {parquet_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary"""
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'total_revenue': float(df['revenue'].sum()),
            'total_orders': len(df[df['revenue'] > 0]),
            'cancelled_orders': len(df[df['is_cancelled']]),
            'cancellation_rate': float((df['is_cancelled'].sum() / len(df)) * 100),
            'avg_order_value': float(df[df['revenue'] > 0]['revenue'].mean()),
            'total_quantity': int(df['quantity'].sum()),
            'unique_products': int(df['sku'].nunique()),
            'unique_categories': int(df['category'].nunique()) if 'category' in df.columns else 0,
            'data_sources': df['data_source'].value_counts().to_dict(),
        }
        
        return summary


def main():
    """Run the ingestion pipeline"""
    pipeline = DataIngestionPipeline()
    
    # Ingest and process all data
    df = pipeline.ingest_all_data()
    
    # Get summary
    summary = pipeline.get_data_summary()
    
    print("\n" + "="*80)
    print("📊 DATA INGESTION SUMMARY")
    print("="*80)
    for key, value in summary.items():
        print(f"{key:.<50} {value}")
    print("="*80)
    
    return df


if __name__ == "__main__":
    main()

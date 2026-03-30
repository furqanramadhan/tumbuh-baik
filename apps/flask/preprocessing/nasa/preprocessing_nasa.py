from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import logging 
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NasaPreprocessingError(Exception):
    """Custom exception for NASA POWER preprocessing errors"""
    pass

class NasaDataValidator:
    """Validated NASA POWER data before preprocessing"""
    
    def validate_dataset(self, db, collection_name: str) -> Dict[str, Any]:
        """
        Validates that the dataset contains required columns and is suitable for preprocessing
        Returns a dictionary with validation results
        """
        try:
            # Check if collection exists
            if collection_name not in db.list_collection_names():
                return{
                    'valid': False,
                    'errors': [f"Collection {collection_name} does not exist"]
                }
            # GET sample document to check schema
            sample=db[collection_name].find_one()
            if not sample:
                return{
                    'valid': False,
                    'errors': [f"Collection {collection_name} is empty"]
                }
            # Define required fields for NASA POWER dataset
            required_fields = [
                'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 
                'WS10M', 'WS10M_MAX', 'WD10M'
            ]
            # Check if all required fields are exists
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing required fields: {', '.join(missing_fields)}"]
                }
            # Check if date field exists
            date_fields = ['Date', 'Year', 'month', 'day']
            missing_date_fields = [field for field in date_fields if field not in sample]
            
            if missing_date_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing date fields: {', '.join(missing_date_fields)}"]
                }
            # All validations passed
            return {
                'valid': True,
                'message': "NASA POWER dataset is valid for preprocessing!"
            }
        except Exception as e:
            logger.error(f"Error during dataset validation: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
            
class NasaDataLoader:
    """Loads NASA POWER data from MongoDB into pandas DataFrame"""
    
    def load_data(self, db, collection_name: str) -> pd.DataFrame:
        """Load all data from MongoDB into pd.DataFrame"""
        try:
            # Load all records from MongoDB collection, no pagination
            cursor = db[collection_name].find({})
            df = pd.DataFrame(list(cursor))
            
            if len(df) == 0:
                raise NasaPreprocessingError(f"No data found in collection '{collection_name}'")
            
            # Convert ObjectId into string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            # Ensure date columns is datetime type
            if 'Date' in df.columns and not pd.api.types.is_datetime64_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.warning(f"Failed to convert 'Date' column to datetime: {str(e)}")
            
            # Sort by Date chronologically order
            if 'Date' in df.columns:
                df = df.sort_values('Date')
            
            logger.info(f"Successfully loaded {len(df)} records from '{collection_name}'")
            return df
        except Exception as e:
            error_msg = f"Error loading data from collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
        
class NasaDataSaver:
    """Saves preprocessed NASA POWER data back to new collection_name MongoDB"""
    
    def save_preprocessed_data(
        self,
        db,
        preprocessed_data: pd.DataFrame,
        original_collection_name: str,
    ) -> Dict[str, Any]:
        """Save preprocessed to a new collection, update dataset-meta"""
        try:
            # Generate cleaned collection name
            cleaned_collection_name = f"{original_collection_name}_cleaned"
            
            # Drop the existing cleaned collection if it exists
            if cleaned_collection_name in db.list_collection_names():
                logger.info(f"Dropping existing collection: {cleaned_collection_name}")
                db[cleaned_collection_name].drop()
            
            # Generate new _id values instead of keeping the original ones
            # Create a new DataFrame without the _id column
            if '_id' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('_id', axis=1)
                
            # Remove any existing __v column
            if '__v' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('__v', axis=1)
                logger.info("Dropped '__v' column from preprocessed data")
            
            # Convert DataFrame records for MongoDB insertion
            # MongoDB will generate new _id values automatically
            records = preprocessed_data.to_dict('records')
            if records:
                db[cleaned_collection_name].insert_many(records)
                logger.info(f"Inserted {len(records)} records into '{cleaned_collection_name}'")
            
            # Update dataset-meta collection 
            meta_info = self._update_dataset_metadata(
                db,
                original_collection_name,
                cleaned_collection_name,
                len(records)
            )
            return {
                "preprocessedCollections": [cleaned_collection_name],
                "recordsInserted": {
                    "original": db[original_collection_name].count_documents({}),
                    "cleaned": len(records)
                },
                "metadata": meta_info
            }
        except Exception as e:
            error_msg = f"Error saving preprocessed data: {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
    
    def _update_dataset_metadata(
        self,
        db,
        original_collection_name: str,
        cleaned_collection_name: str,
        record_count: int
    ) -> Dict[str, Any]:
        """Update dataset-meta after preprocessing - Point to cleaned collection"""
        try:
            # Find the metadata collection
            meta_collection = None
            if "dataset_meta" in db.list_collection_names():
                meta_collection = "dataset_meta"
            elif "DatasetMeta" in db.list_collection_names():
                meta_collection = "DatasetMeta"
            else:
                logger.warning("No metadata collection found!")
                return {"status": "no_meta_collection"}
            
            # Get original metadata to preserve other fields
            original_meta = db[meta_collection].find_one({"collectionName": original_collection_name})
            
            if not original_meta:
                logger.warning(f"No metadata found for collection '{original_collection_name}'")
                return {"status": "no_original_metadata"}
            
            # Get cleaned collection columns
            sample_doc = db[cleaned_collection_name].find_one()
            if sample_doc:
                all_columns = list(sample_doc.keys())
                cleaned_columns = [
                    col for col in all_columns
                    if col not in ['_id', '__v'] and not col.startswith('_')
                ]
            else:
                cleaned_columns = []
            
            # cleaned_columns = list(sample_doc.keys()) if sample_doc else []
            
            # Update metadata to point to cleaned collection
            update_fields = {
                "status": "preprocessed",
                "collectionName": cleaned_collection_name,  
                "totalRecords": record_count,               
                "columns": cleaned_columns,
                "lastUpdated": datetime.now(),
                "name": f"{original_meta.get('name', original_collection_name)} (Cleaned)"
            }
            
            # Only include apiConfig if it's an API dataset
            if original_meta.get('isAPI', False):
                update_fields["apiConfig"] = original_meta.get('apiConfig', {})
            
            result = db[meta_collection].update_one(
                {"collectionName": original_collection_name},
                {"$set": update_fields}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated metadata to point to cleaned collection: {cleaned_collection_name}")
            else:
                logger.warning("No metadata was updated")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {"status": "error", "error": str(e)}
        
class NasaPreprocessor:
    """Main class for preprocessing NASA POWER datasets"""
    
    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.validator = NasaDataValidator()
        self.loader = NasaDataLoader()
        self.original_data = None
        self.saver = NasaDataSaver()
        self.preprocessing_report = {
            "dataset_type": "nasa",
            "missing_data": {},
            "outliers": {},
            "smoothing": {},
            "smoothing_validation": {},
            "gaps": {},
            "model_coverage": {},
            "quality_metrics": {},
            "warnings": [],
            "decomposition": {}  # Decomposition unified report
        }
        
        
    def _sanitize_for_mongodb(self, obj):
        """
        Recursively convert numpy types to native Python types for MongoDB serialization
        """
        if isinstance(obj, dict):
            return {key: self._sanitize_for_mongodb(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_mongodb(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj

    def preprocess(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess NASA POWER dataset with separated report storage
        
        Args:
            options: Dictionary of preprocessing options
            
        Returns:
            Dictionary with preprocessing results and processed dataframe
        """
        try:
            # Default options (unchanged)
            default_options = {
                "smoothing_method": "exponential",
                "window_size": 5,
                "exponential_alpha": 0.05,
                "adaptive_alpha_selection": False,
                "alpha_optimization_range": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                "trend_window_size": 7,
                "drop_outliers": True,
                "outlier_methods": ["iqr", "zscore"],
                "iqr_multiplier": 2.0,
                "zscore_threshold": 3.5,
                "outlier_treatment": "interpolate",
                "fill_missing": True,
                "detect_gaps": True,
                "exclude_tail_data": True,
                "max_gap_interpolate": 90,
                "calculate_coverage": True,
                "columns_to_process": [
                    'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 
                    'WS10M', 'WS10M_MAX', 'WD10M'
                ],
                "parameter_configs": {
                    "T2M": {
                        "smoothing_method": "adaptive_exponential",
                        "alpha_range": [0.30, 0.65],
                        "volatility_window": 30,
                        "reason": "Temperature stable - adaptive smoothing to preserve long-term trend (avoid alpha < 0.30)"
                    },
                    "RH2M": {
                        "smoothing_method": "adaptive_exponential",
                        "alpha_range": [0.25, 0.65],
                        "volatility_window": 30,
                        "reason": "Humidity high variability - adaptive smoothing preserves trends"
                    },
                    "ALLSKY_SFC_SW_DWN": {
                        "smoothing_method": None,
                        "apply_outlier_detection": False,
                        "reason": "Solar radiation - no smoothing to preserve daily (day-night) signal and strong seasonality; outliers likely cloud events (signal)"
                    },
                    "PRECTOTCORR": {
                        "smoothing_method": None,
                        "apply_outlier_detection": False,
                        "reason": "Precipitation events are signal - retain extreme values"
                    },
                    "T2M_MAX": {
                        "smoothing_method": "adaptive_exponential",
                        "alpha_range": [0.20, 0.50],
                        "volatility_window": 30,
                        "reason": "Max temperature - adaptive for weather pattern changes"
                    },
                    "T2M_MIN": {
                        "smoothing_method": "adaptive_exponential",
                        "alpha_range": [0.20, 0.50],
                        "volatility_window": 30,
                        "reason": "Min temperature - adaptive for weather pattern changes"
                    },
                    # Wind variables - no smoothing
                    "WS10M": {"smoothing_method": None},
                    "WS10M_MAX": {"smoothing_method": None},
                    "WD10M": {"smoothing_method": None}
                }
            }
            
            # Merge with provided options
            if options is None:
                options = {}
            self.options = {**default_options, **options}
            
            # Validate dataset
            validation_result = self.validator.validate_dataset(self.db, self.collection_name)
            if not validation_result.get('valid', False):
                raise NasaPreprocessingError(f"Validation failed: {validation_result.get('errors', ['Unknown error'])}")
            
            # Load data - this gets all records, not just one page
            df = self.loader.load_data(self.db, self.collection_name)
            logger.info(f"Loaded {len(df)} records from collection '{self.collection_name}'")
            
            # Apply preprocessing 
            processed_df = self._apply_preprocessing(df)
            
            # Save processed data
            save_result = self.saver.save_preprocessed_data(
                self.db, 
                processed_df, 
                self.collection_name
            )
            
            cleaned_collection = save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None
            
            # STEP 1: Save preprocessing_report FIRST (to get preprocessing_id)
            report_save_result = self._save_preprocessing_report(
                cleaned_collection or self.collection_name
            )
            
            preprocessing_id = report_save_result.get("report_id")  # ObjectId
            
            # STEP 2: Calculate and save decomposition (uses preprocessing_id as reference)
            if preprocessing_id and report_save_result.get("status") == "success":
                try:
                    self._calculate_and_save_decomposition(processed_df, preprocessing_id)
                    decomposition_saved = True
                    logger.info("Decomposition data saved to decomposition_report collection")
                except Exception as e:
                    logger.error(f"Failed to save decomposition: {str(e)}")
                    decomposition_saved = False
            else:
                decomposition_saved = False
                logger.warning("Skipping decomposition save - preprocessing report failed")
            
            return {
                "status": "success",
                "message": "NASA POWER dataset preprocessed successfully",
                "collection": self.collection_name,
                "preprocessedData": processed_df.head(10).to_dict('records'),
                "recordCount": len(processed_df),
                "originalRecordCount": len(df),
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": cleaned_collection,
                "preprocessing_report_id": str(preprocessing_id) if preprocessing_id else None,
                "report_saved": report_save_result.get("status") == "success",
                "decomposition_saved": decomposition_saved,
                "decomposition_collection": "decomposition_report" if decomposition_saved else None
            }
            
        except Exception as e:
            error_msg = f"Error preprocessing NASA POWER data: {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
    
    def _apply_preprocessing(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply preprocessing steps to the data with progress tracking"""
        logger.info("Starting NASA POWER DATA preprocessing...")
        
        processed_df = df.copy()
        total_steps = 10
        current_step = 0
        
        self.original_data = df.copy()
        
        # Helper function untuk log progress
        def log_progress(stage, message):
            nonlocal current_step
            current_step += 1
            percentage = int((current_step / total_steps) * 100)
            logger.info(f"PROGRESS:{percentage}:{stage}:{message}")
        
        # STEP 1-5: Existing preprocessing steps...
        log_progress("fill_values", "Replacing fill values with NaN...")
        processed_df = self._replace_fill_values(processed_df)
        
        log_progress("tail_data", "Checking for tail data to exclude...")
        if self.options.get("exclude_tail_data", True):
            processed_df = self._exclude_tail_data(processed_df)
        
        log_progress("gap_detection", "Detecting gaps in time series...")
        if self.options.get("detect_gaps", True):
            self._detect_gaps(processed_df)
        
        log_progress("imputation", "Imputing missing values...")
        if self.options.get("fill_missing", True):
            processed_df = self._impute_missing_values(processed_df)
        
        log_progress("outliers", "Detecting and handling outliers...")
        if self.options.get("drop_outliers", True):
            processed_df = self._handle_outliers(processed_df)
        
        # STEP 6: Alpha optimization for regular exponential parameters
        if self.options.get("adaptive_alpha_selection", False):
            log_progress("alpha_optimization", "Optimizing alpha parameters for better trend preservation...")
            self._optimize_parameter_alphas(processed_df)
        
        # STEP 7: Volatility optimization for adaptive parameters (INDEPENDENT)
        log_progress("volatility_optimization", "Optimizing volatility parameters for adaptive smoothing...")
        self._optimize_volatility_parameters(processed_df)
        
        # Save data state BEFORE smoothing
        logger.info("SAVING PRE-SMOOTHING DATA STATE for correct trend validation...")
        self.pre_smoothing_data = processed_df.copy()
        
        # Log verification
        for param in ['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN']:
            if param in self.pre_smoothing_data.columns:
                stats = self.pre_smoothing_data[param].describe()
                logger.info(f"PRE-SMOOTHING {param}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, count={stats['count']}")
        
        log_progress("smoothing", "Applying smoothing methods...")
        processed_df = self._apply_smoothing(processed_df)
        
        # Log verification after smoothing
        for param in ['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN']:
            if param in processed_df.columns:
                stats = processed_df[param].describe()
                logger.info(f"POST-SMOOTHING {param}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # STEP 8: Smoothing validation (SINGLE INSTANCE - KEEP THIS)
        log_progress("smoothing_validation", "Validating smoothing quality (GCV + Trend Preservation)...")
        self._validate_smoothing_method(processed_df)
        
        # STEP 9: Model coverage analysis
        log_progress("model_coverage", "Calculating model coverage analysis...")
        if self.options.get("calculate_coverage", True):
            self._calculate_model_coverage(processed_df)
            
        self.log_detailed_coverage_analysis(processed_df)
        
        # STEP 10: Quality metrics
        log_progress("quality_metrics", "Generating quality metrics...")
        self._generate_quality_metrics(df, processed_df) 
        
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df
    
    def _replace_fill_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NASA POWER fill values (-999, -99, -9999) with NaN"""
        logger.info("Replacing fill values with NaN...")
        
        fill_values = [-999, -99, -9999]
        params = self.options.get("columns_to_process", [])
        
        replaced_count = {}
        for param in params:
            if param in df.columns:
                mask = df[param].isin(fill_values)
                count = mask.sum()
                if count > 0:
                    df.loc[mask, param] = np.nan
                    replaced_count[param] = int(count)
                    
        self.preprocessing_report["missing_data"]["fill_values_replaced"] = replaced_count
        logger.info(f"Replaced fill values: {replaced_count}")
        return df
    
    def _exclude_tail_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude last 5-7 days due to NASA POWER data lag"""
        logger.info("Checking for tail data to exclude...")
        
        if 'Date' not in df.columns:
            logger.warning("No Date column found, skipping tail exclusion")
            return df
        
        # Find latest complete date (all parameters have valid data)
        latest_complete = self._find_latest_complete_date(df)
        
        if latest_complete is None:
            logger.warning("Could not determine latest complete date")
            return df
        
        # Count excluded records
        excluded_mask = df['Date'] > latest_complete
        excluded_count = excluded_mask.sum()
        
        if excluded_count > 0:
            excluded_dates = df[excluded_mask]['Date'].tolist()
            df_complete = df[~excluded_mask].copy()
            
            self.preprocessing_report["missing_data"]["tail_data_excluded"] = {
                "count": int(excluded_count),
                "latest_complete_date": str(latest_complete),
                "excluded_dates": [str(d) for d in excluded_dates[:10]],  # First 10 dates
                "reason": "NASA POWER data lag (~5 days)"
            }
            
            logger.info(f"Excluded {excluded_count} tail records after {latest_complete}")
            return df_complete
        
        return df
    
    def _find_latest_complete_date(self, df: pd.DataFrame) -> Optional[datetime]:
        """Find the last date where all parameters have valid data"""
        params = self.options.get("columns_to_process", [])
        
        # Sort by date descending
        df_sorted = df.sort_values('Date', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            # Check if all parameters are valid (not NaN)
            all_valid = all(
                pd.notna(row.get(param)) for param in params if param in df.columns
            )
            
            if all_valid:
                return row['Date']
        
        return None
    
    def _detect_gaps(self, df: pd.DataFrame) -> None:
        """Detect and classify gaps in the time series"""
        logger.info("Detecting gaps in time series data...")    
        
        if 'Date' not in df.columns:
            logger.warning("No Date column found, skipping gap detection")
            return

        df_sorted = df.sort_values('Date').reset_index(drop=True)
        dates = pd.to_datetime(df_sorted['Date'])
        
        # Find expected date range
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
        missing_dates = date_range.difference(dates)
        
        if len(missing_dates) == 0:
            logger.info("No date gaps found")
            self.preprocessing_report["gaps"] = {
                "total_gaps": 0,
                "gap_details": []
            }
            return
        
        # Group consecutive missing dates into gaps
        gaps = []
        current_gap = [missing_dates[0]]
        
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i-1]).days == 1:
                current_gap.append(missing_dates[i])
            else:
                gaps.append(current_gap)
                current_gap = [missing_dates[i]]
        gaps.append(current_gap)
        
        # Classify gaps
        gap_details = []
        small_gaps = medium_gaps = large_gaps = 0
        
        for gap in gaps:
            duration = len(gap)
            gap_info = {
                "start_date": str(gap[0]),
                "end_date": str(gap[-1]),
                "duration_days": duration
            }
            
            if duration <= 7:
                gap_info["type"] = "small"
                gap_info["imputation_method"] = "linear"
                small_gaps += 1
            elif duration <= 30:
                gap_info["type"] = "medium"
                gap_info["imputation_method"] = "spline"
                medium_gaps += 1
            else:
                gap_info["type"] = "large"
                gap_info["imputation_method"] = "seasonal" if duration <= 90 else "none"
                large_gaps += 1
                
                if duration > 90:
                    self.preprocessing_report["warnings"].append(
                        f"Large gap detected: {gap_info['start_date']} to {gap_info['end_date']} "
                        f"({duration} days) - not interpolated"
                    )
            
            gap_details.append(gap_info)
            
        self.preprocessing_report["gaps"] = {
            "total_gaps": len(gaps),
            "small_gaps": small_gaps,
            "medium_gaps": medium_gaps,
            "large_gaps": large_gaps,
            "gap_details": gap_details[:10] # First 10 gaps for report
        }
        logger.info(f"Detected {len(gaps)} gaps: {small_gaps} small, {medium_gaps} medium, {large_gaps} large")
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using appropriate methods"""
        logger.info("Imputing missing values...")
        
        params = self.options.get("columns_to_process", [])
        imputation_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Get parameter-specific config
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            
            # Special handling for precipitation
            if param == "PRECTOTCORR":
                df, imputed_count = self._impute_precipitation(df, param)
            else:
                df, imputed_count = self._impute_standard_parameter(df, param)
            
            if imputed_count > 0:
                imputation_summary[param] = imputed_count
        
        self.preprocessing_report["missing_data"]["imputed_values"] = imputation_summary
        logger.info(f"Imputation summary: {imputation_summary}")
        
        return df
    
    def _impute_precipitation(self, df: pd.DataFrame, param: str) -> tuple:
        """Special imputation for precipitation data"""
        missing_mask = df[param].isna()
        imputed_count = 0
        
        if not missing_mask.any():
            return df, imputed_count
        
        # For small gaps, use context-aware imputation
        missing_indices = np.where(missing_mask)[0]
        
        for idx in missing_indices:
            # Get surrounding context
            before_idx = max(0, idx - 1)
            after_idx = min(len(df) - 1, idx + 1)
            
            before_val = df.iloc[before_idx][param] if before_idx < idx else np.nan
            after_val = df.iloc[after_idx][param] if after_idx > idx else np.nan
            
            # Context-aware imputation
            if pd.notna(before_val) and pd.notna(after_val):
                if before_val == 0 and after_val == 0:
                    # No rain before and after - assume no rain
                    df.loc[df.index[idx], param] = 0.0
                else:
                    # Linear interpolation
                    df.loc[df.index[idx], param] = (before_val + after_val) / 2
                imputed_count += 1
            elif pd.notna(before_val):
                # Forward fill
                df.loc[df.index[idx], param] = before_val
                imputed_count += 1
            elif pd.notna(after_val):
                # Backward fill
                df.loc[df.index[idx], param] = after_val
                imputed_count += 1
        
        return df, imputed_count
    
    def _impute_standard_parameter(self, df: pd.DataFrame, param: str) -> tuple:
        """Standard imputation for non-precipitation parameters"""
        missing_mask = df[param].isna()
        original_missing_count = missing_mask.sum()
        
        if original_missing_count == 0:
            return df, 0
        
        # Linear interpolation for small gaps
        df[param] = df[param].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline interpolation for remaining gaps
        remaining_missing = df[param].isna().sum()
        if remaining_missing > 0 and remaining_missing < original_missing_count:
            df[param] = df[param].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Forward/backward fill for edge cases
        df[param] = df[param].fillna(method='ffill', limit=3)
        df[param] = df[param].fillna(method='bfill', limit=3)
        
        final_missing = df[param].isna().sum()
        imputed_count = original_missing_count - final_missing
        
        return df, int(imputed_count)
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using IQR and Z-score methods"""
        logger.info("Detecting and handling outliers...")
        
        params = self.options.get("columns_to_process", [])
        outlier_methods = self.options.get("outlier_methods", ["iqr", "zscore"])
        outlier_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Skip outlier detection for precipitation
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            if not param_config.get("apply_outlier_detection", True):
                continue
            
            # Detect outliers using both methods
            outliers_iqr = self._detect_outliers_iqr(df, param) if "iqr" in outlier_methods else pd.Series([False] * len(df))
            outliers_zscore = self._detect_outliers_zscore(df, param) if "zscore" in outlier_methods else pd.Series([False] * len(df))
            
            # Intersection: both methods must agree
            outliers_mask = outliers_iqr & outliers_zscore
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                # Handle outliers
                df = self._treat_outliers(df, param, outliers_mask)
                outlier_summary[param] = int(outlier_count)
        
        self.preprocessing_report["outliers"] = {
            "total_outliers": sum(outlier_summary.values()),
            "by_parameter": outlier_summary,
            "methods_used": outlier_methods,
            "treatment": self.options.get("outlier_treatment", "interpolate")
        }
        
        logger.info(f"Outlier detection summary: {outlier_summary}")
        
        return df
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, param: str) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = df[param].quantile(0.25)
        Q3 = df[param].quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = self.options.get("iqr_multiplier", 2.0)
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (df[param] < lower_bound) | (df[param] > upper_bound)
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, param: str) -> pd.Series:
        """Detect outliers using Z-score method"""
        mean = df[param].mean()
        std = df[param].std()
        
        if std == 0:
            return pd.Series([False] * len(df))
        
        z_scores = np.abs((df[param] - mean) / std)
        threshold = self.options.get("zscore_threshold", 3.5)
        
        return z_scores > threshold
    
   
    
    def _treat_outliers(self, df: pd.DataFrame, param: str, outliers_mask: pd.Series) -> pd.DataFrame:
        """Treat detected outliers"""
        treatment = self.options.get("outlier_treatment", "interpolate")
        
        if treatment == "interpolate":
            # Replace outliers with NaN then interpolate
            df.loc[outliers_mask, param] = np.nan
            df[param] = df[param].interpolate(method='linear', limit=3, limit_direction='both')
        
        elif treatment == "cap":
            # Winsorization - cap at boundary values
            Q1 = df[param].quantile(0.25)
            Q3 = df[param].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.options.get("iqr_multiplier", 2.0)
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df.loc[df[param] < lower_bound, param] = lower_bound
            df.loc[df[param] > upper_bound, param] = upper_bound
        
        elif treatment == "remove":
            # Remove rows with outliers (not recommended)
            df = df[~outliers_mask]
        
        return df
    
    def _calculate_model_coverage(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Calculate model applicability coverage for Holt-Winters and LSTM models
        Analyzes data segments that are suitable/unsuitable for each model type
        """
        logger.info("Calculating model coverage analysis...")
        
        params = self.options.get("columns_to_process", [])
        coverage_results = {
            "holt_winters": {"coverage_percentage": 0, "uncovered_breakdown": {}},
            "lstm": {"coverage_percentage": 0, "uncovered_breakdown": {}},
            "per_parameter": {}
        }
        
        total_data_points = len(df)
        
        for param in params:
            if param not in df.columns:
                continue
            
            try:
                param_coverage = self._analyze_parameter_coverage(df, param, total_data_points)
                coverage_results["per_parameter"][param] = param_coverage
            except Exception as e:
                logger.error(f"Error calculating coverage for parameter {param}: {str(e)}")
                coverage_results["per_parameter"][param] = {
                    "holt_winters_coverage": 0,
                    "lstm_coverage": 0,
                    "error": str(e)
                }
        
        # Calculate overall coverage percentages
        if coverage_results["per_parameter"]:
            hw_coverage = [
                result.get("holt_winters_coverage", 0)
                for result in coverage_results["per_parameter"].values()
                if isinstance(result, dict) and "holt_winters_coverage" in result
            ]
            lstm_coverage = [
                result.get("lstm_coverage", 0)
                for result in coverage_results["per_parameter"].values()
                if isinstance(result, dict) and "lstm_coverage" in result
            ]
            
            # Overall coverage (average across parameters)
            overall_hw_coverage = sum(hw_coverage) / len(hw_coverage) if hw_coverage else 0
            overall_lstm_coverage = sum(lstm_coverage) / len(lstm_coverage) if lstm_coverage else 0
            
            # Aggregate uncovered breakdown
            hw_uncovered = self._aggregate_uncovered_breakdown(coverage_results["per_parameter"], "holt_winters")
            lstm_uncovered = self._aggregate_uncovered_breakdown(coverage_results["per_parameter"], "lstm")
            
            coverage_results["holt_winters"] = {
                "coverage_percentage": round(overall_hw_coverage, 2),
                "uncovered_breakdown": hw_uncovered,
                "model_suitability": self._determine_model_suitability(overall_hw_coverage)
            }
            
            coverage_results["lstm"] = {
                "coverage_percentage": round(overall_lstm_coverage, 2),
                "uncovered_breakdown": lstm_uncovered,
                "model_suitability": self._determine_model_suitability(overall_lstm_coverage)
            }
        
        # Store coverage results in preprocessing report
        self.preprocessing_report["model_coverage"] = coverage_results
        logger.info(f"Model coverage - Holt Winters: {coverage_results['holt_winters']['coverage_percentage']:.1f}%" )
        logger.info(f"Model coverage - LSTM: {coverage_results['lstm']['coverage_percentage']:.1f}%")
        
    def _get_smoothing_quality(self, param: str) -> Dict[str, Any]:
        """
        Retrieve smoothing validation results for parameter
        
        Returns:
            dict: GCV score, quality status, trend preservation, and penalties
        """
        validation = self.preprocessing_report.get("smoothing_validation", {})
        
        if param not in validation:
            return {
                "quality": "unknown",
                "gcv": None,
                "trend": None,
                "trend_value": 0.0,
                "penalty": 0.0
            }
        
        param_validation = validation[param]
        quality_status = param_validation.get("quality_status", "unknown")
        trend_pct = param_validation.get("trend_preservation_pct", 0)
        trend_value = trend_pct / 100.0 
        
        quality_penalty_map = {
            "excellent": 0.0,
            "good": 5.0,
            "fair": 10.0,
            "poor": 20.0,
            "unknown": 0.0
        }
        
        return {
            "quality": quality_status,
            "gcv": param_validation.get("gcv_score"),
            "trend": param_validation.get("trend_preservation_pct"),
            "trend_value": trend_value,
            "penalty": quality_penalty_map.get(quality_status, 0.0)
        }
        
    def _analyze_parameter_coverage(
        self,
        df: pd.DataFrame,
        param: str,
        total_points: int
    ) -> Dict[str, Any]:
        """
        Analyze coverage for a specific parameter for both model types
        Remove verbose analysis_details from output
        """
        series = df[param].dropna()
        if len(series) < 30: # Need minimum data for analysis
            return {
                "holt_winters_coverage": 0,
                "lstm_coverage": 0,
                "insufficient_data": True
            }
        
        # Analysis results (for internal calculation only)
        coverage_analysis = {
            "data_points": len(series),
            "missing_ratio": (total_points - len(series)) / total_points
        }
        
        # Perform all analysis (same as before)
        large_gaps_impact = self._analyze_large_gaps(df, param)
        extreme_outliers_impact = self._analyze_extreme_outliers(series)
        seasonality_analysis = self._analyze_seasonality(series)
        stationarity_analysis = self._analyze_stationarity(series)
        precipitation_analysis = self._analyze_precipitation_extremes(series, param) if param == "PRECTOTCORR" else {}
        
        # Check if parameter was smoothed
        smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})
        was_smoothed = smoothing_summary.get(param, "none") != "none"

        # Calculate coverage (same logic)
        if was_smoothed:
            hw_coverage = self._calculate_holt_winters_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                seasonality_analysis,
                stationarity_analysis,
                coverage_analysis,
                param
            )
            
            lstm_coverage = self._calculate_lstm_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                precipitation_analysis,
                stationarity_analysis,
                coverage_analysis,
                param  
            )
        else:
            non_smoothed_coverage = self._calculate_non_smoothed_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                coverage_analysis,
                param
            )
            hw_coverage = lstm_coverage = non_smoothed_coverage
        
        # Remove analysis_details entirely
        result = {
            "holt_winters_coverage": hw_coverage["coverage_percentage"],
            "lstm_coverage": lstm_coverage["coverage_percentage"]
        }
        
        # ADD: keep lightweight details for detailed logger
        result["analysis_details"] = {
            "data_points": len(series),
            "missing_ratio": coverage_analysis.get("missing_ratio", 0),
            "large_gaps": large_gaps_impact,
            "extreme_outliers": extreme_outliers_impact,
            "seasonality": seasonality_analysis,
            "stationarity": stationarity_analysis,
        }

        if precipitation_analysis:
            result["analysis_details"]["precipitation"] = precipitation_analysis
        
        # Only add uncovered reasons if they exist (avoid empty objects)
        if hw_coverage["uncovered_reasons"]:
            result["holt_winters_uncovered"] = hw_coverage["uncovered_reasons"]
        
        if lstm_coverage["uncovered_reasons"]:
            result["lstm_uncovered"] = lstm_coverage["uncovered_reasons"]
        
        return result
        
    def _analyze_large_gaps(self, df: pd.DataFrame, param: str) -> Dict[str, Any]:
        """Analyze impact of large gaps (>90 days)"""
        if 'Date' not in df.columns:
            return {"impact_percentage": 0, "large_gaps_count": 0}
        
        # Get gaps from preprocessing report
        gaps_info = self.preprocessing_report.get("gaps", {})
        large_gaps = [gap for gap in gaps_info.get("gap_details", []) if gap.get("duration_days", 0) > 90]
        
        if not large_gaps:
            return {"impact_percentage": 0, "large_gaps_count": 0}
        
        # Calculate impact
        total_gap_days = sum(gap["duration_days"] for gap in large_gaps)
        total_days = (df['Date'].max() - df['Date'].min()).days + 1
        impact_percentage = (total_gap_days / total_days) * 100
        
        return {
            "impact_percentage": round(impact_percentage, 2),
            "large_gaps_count": len(large_gaps),
            "total_gap_days": total_gap_days
        }
    
    def _analyze_extreme_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze extreme outliers that survived double-detection"""
        if len(series) < 10:
            return {"impact_percentage": 0}
        
        # More stringent outlier detection (3.5 sigma)
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return {"impact_percentage": 0}
        
        z_scores = np.abs((series - mean) / std)
        extreme_outliers = z_scores > 3.5  # More stringent than preprocessing (3.0)
        
        impact_percentage = (extreme_outliers.sum() / len(series)) * 100
        
        return {
            "impact_percentage": round(impact_percentage, 2),
            "extreme_outliers_count": int(extreme_outliers.sum()),
            "max_z_score": round(float(z_scores.max()), 2)
        }

    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns (critical for Holt-Winters) using STL"""
        try:
            if len(series) < 730:  # Need at least 2 years for seasonal analysis
                return {"seasonal_strength": 0, "has_clear_seasonality": False, "insufficient_data": True}

            # STL decomposition (consistent with decomposition pipeline)
            stl = STL(series, period=365, robust=True)
            decomposition = stl.fit()

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())

            if seasonal_var + residual_var == 0:
                seasonal_strength = 0
            else:
                seasonal_strength = seasonal_var / (seasonal_var + residual_var)

            has_clear_seasonality = seasonal_strength > 0.3  # Threshold for clear seasonality

            return {
                "seasonal_strength": round(float(seasonal_strength), 3),
                "has_clear_seasonality": has_clear_seasonality,
                "seasonal_variance": round(float(seasonal_var), 3),
                "residual_variance": round(float(residual_var), 3)
            }

        except Exception as e:
            logger.warning(f"Error in seasonal analysis: {str(e)}")
            return {"seasonal_strength": 0, "has_clear_seasonality": False, "error": str(e)}

    def _analyze_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze stationarity using Augmented Dickey-Fuller test"""
        try:
            if len(series) < 50:
                return {"is_stationary": False, "insufficient_data": True}
            
            # Perform ADF test
            adf_result = adfuller(series.dropna())
            
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05 indicates stationarity
            
            return {
                "is_stationary": is_stationary,
                "adf_statistic": round(float(adf_result[0]), 4),
                "p_value": round(float(adf_result[1]), 4),
                "critical_values": {k: round(v, 4) for k, v in adf_result[4].items()}
            }
            
        except Exception as e:
            logger.warning(f"Error in stationarity analysis: {str(e)}")
            return {"is_stationary": False, "error": str(e)}
        
    def _analyze_precipitation_extremes(self, series: pd.Series, param: str) -> Dict[str, Any]:
        """Analyze extreme precipitation events (0 vs 500mm range)"""
        if param != "PRECTOTCORR":
            return {}
        
        zero_ratio = (series == 0).sum() / len(series)
        extreme_high = (series > 100).sum() / len(series)  # >100mm per day is extreme
        
        # Calculate range ratio (wide range affects LSTM)
        data_range = series.max() - series.min()
        range_impact = min(data_range / 500, 1.0)  # Normalize to 0-1
        
        return {
            "zero_precipitation_ratio": round(float(zero_ratio), 3),
            "extreme_precipitation_ratio": round(float(extreme_high), 3),
            "range_impact": round(float(range_impact), 3),
            "max_precipitation": round(float(series.max()), 2)
        }
        
    def log_detailed_coverage_analysis(self, processed_df: pd.DataFrame) -> None:
        """
        Log detailed coverage analysis for debugging and verification.
        Outputs comprehensive breakdown of penalties, thresholds, and quality metrics.
        """
        
        logger.info("DETAILED COVERAGE ANALYSIS REPORT")
        
        
        # Get coverage data
        coverage_data = self.preprocessing_report.get("model_coverage", {})
        per_param = coverage_data.get("per_parameter", {})
        
        if not per_param:
            logger.warning("No per-parameter coverage data available")
            return
        
        # Overall Summary
        logger.info("OVERALL COVERAGE SUMMARY:")
        logger.info(f"  Holt-Winters: {coverage_data.get('holt_winters', {}).get('coverage_percentage', 0):.2f}%")
        logger.info(f"  LSTM: {coverage_data.get('lstm', {}).get('coverage_percentage', 0):.2f}%")
        logger.info(f"  HW Suitability: {coverage_data.get('holt_winters', {}).get('model_suitability', 'unknown').upper()}")
        logger.info(f"  LSTM Suitability: {coverage_data.get('lstm', {}).get('model_suitability', 'unknown').upper()}")
        
        # Per-Parameter Detailed Breakdown
        
        logger.info("PER-PARAMETER DETAILED ANALYSIS:")
        
        
        params = self.options.get("columns_to_process", [])
        
        for param in params:
            if param not in per_param:
                logger.info(f"{param}: No coverage data available")
                continue
            
            param_data = per_param[param]

            # Basic Coverage
            hw_coverage = param_data.get("holt_winters_coverage", 0)
            lstm_coverage = param_data.get("lstm_coverage", 0)
            logger.info(f"Coverage:")
            logger.info(f"Holt-Winters: {hw_coverage:.2f}%")
            logger.info(f"LSTM: {lstm_coverage:.2f}%")
            # Analysis Details
            analysis = param_data.get("analysis_details", {})
            
            # 1. Smoothing Quality (if available)
            smoothing_validation = self.preprocessing_report.get("smoothing_validation", {})
            if param in smoothing_validation:
                smooth_data = smoothing_validation[param]
                logger.info(f"SMOOTHING QUALITY:")
                logger.info(f"Status: {smooth_data.get('quality_status', 'unknown').upper()}")
                logger.info(f"GCV Score: {smooth_data.get('gcv_score', 'N/A')}")
                logger.info(f"Trend Preservation: {smooth_data.get('trend_preservation_pct', 'N/A')}%")
                logger.info(f"Method: {smooth_data.get('smoothing_method', 'none')}")
                logger.info(f"Data Points: {smooth_data.get('data_points', 0)}")
            
            # 2. Large Gaps Analysis
            large_gaps = analysis.get("large_gaps", {})
            if large_gaps:
                logger.info(f"LARGE GAPS ANALYSIS:")
                logger.info(f"Impact: {large_gaps.get('impact_percentage', 0):.2f}%")
                logger.info(f"Count: {large_gaps.get('large_gaps_count', 0)}")
                logger.info(f"Total Gap Days: {large_gaps.get('total_gap_days', 0)}")
            else:
                logger.info(f"LARGE GAPS: None detected")
            
            # 3. Extreme Outliers Analysis
            outliers = analysis.get("extreme_outliers", {})
            if outliers:
                logger.info(f"EXTREME OUTLIERS:")
                logger.info(f"Impact: {outliers.get('impact_percentage', 0):.2f}%")
                logger.info(f"Count: {outliers.get('extreme_outliers_count', 0)}")
                logger.info(f"Max Z-Score: {outliers.get('max_z_score', 'N/A')}")
            else:
                logger.info(f"EXTREME OUTLIERS: None detected")
            
            # 4. Seasonality Analysis
            seasonality = analysis.get("seasonality", {})
            if seasonality and not seasonality.get("insufficient_data", False):
                logger.info(f"SEASONALITY ANALYSIS:")
                logger.info(f"Seasonal Strength: {seasonality.get('seasonal_strength', 0):.3f}")
                logger.info(f"Has Clear Seasonality: {'YES' if seasonality.get('has_clear_seasonality', False) else 'NO'}")
                logger.info(f"Seasonal Variance: {seasonality.get('seasonal_variance', 'N/A')}")
                logger.info(f"Residual Variance: {seasonality.get('residual_variance', 'N/A')}")
                if seasonality.get("error"):
                    logger.info(f"Error: {seasonality.get('error')}")
            else:
                logger.info(f"SEASONALITY: Insufficient data or not analyzed")
            
            # 5. Stationarity Analysis
            stationarity = analysis.get("stationarity", {})
            if stationarity and not stationarity.get("insufficient_data", False):
                logger.info(f"STATIONARITY ANALYSIS:")
                logger.info(f"Is Stationary: {'YES' if stationarity.get('is_stationary', False) else 'NO'}")
                logger.info(f"ADF Statistic: {stationarity.get('adf_statistic', 'N/A')}")
                logger.info(f"P-Value: {stationarity.get('p_value', 'N/A')}")
                if stationarity.get("critical_values"):
                    logger.info(f"Critical Values:")
                    for level, value in stationarity.get("critical_values", {}).items():
                        logger.info(f"{level}: {value}")
                if stationarity.get("error"):
                    logger.info(f"Error: {stationarity.get('error')}")
            else:
                logger.info(f"STATIONARITY: Insufficient data or not analyzed")
            
            # 6. Precipitation Extremes (if applicable)
            precipitation = analysis.get("precipitation", {})
            if precipitation:
                logger.info(f"PRECIPITATION EXTREMES:")
                logger.info(f"Zero Precipitation Ratio: {precipitation.get('zero_precipitation_ratio', 0):.3f}")
                logger.info(f"Extreme Precipitation Ratio: {precipitation.get('extreme_precipitation_ratio', 0):.3f}")
                logger.info(f"Range Impact: {precipitation.get('range_impact', 0):.3f}")
                logger.info(f"Max Precipitation: {precipitation.get('max_precipitation', 'N/A')} mm")
            
            # 7. Holt-Winters Penalty Breakdown
            hw_uncovered = param_data.get("holt_winters_uncovered", {})
            if hw_uncovered:
                logger.info(f"HOLT-WINTERS PENALTY BREAKDOWN:")
                total_penalty = sum(hw_uncovered.values())
                logger.info(f"Total Penalty: {total_penalty:.2f}%")
                
                # Sort penalties by magnitude for better readability
                for reason, penalty in sorted(hw_uncovered.items(), key=lambda x: x[1], reverse=True):
                    # Better formatting for trend preservation
                    if reason == "trend_preservation_loss":
                        display_name = "Trend Preservation Loss"
                    else:
                        display_name = reason.replace('_', ' ').title()
                    
                    logger.info(f"{display_name}: -{penalty:.2f}%")
            else:
                logger.info(f"HOLT-WINTERS: No penalties applied")

            # 8. LSTM Penalty Breakdown
            lstm_uncovered = param_data.get("lstm_uncovered", {})
            if lstm_uncovered:
                logger.info(f"LSTM PENALTY BREAKDOWN:")
                total_penalty = sum(lstm_uncovered.values())
                logger.info(f"Total Penalty: {total_penalty:.2f}%")
                
                for reason, penalty in sorted(lstm_uncovered.items(), key=lambda x: x[1], reverse=True):
                    # Better formatting
                    if reason == "trend_preservation_loss":
                        display_name = "Trend Preservation Loss"
                    else:
                        display_name = reason.replace('_', ' ').title()
                    
                    logger.info(f"{display_name}: -{penalty:.2f}%")
            else:
                logger.info(f"LSTM: No penalties applied")
            
            # 9. Data Quality Indicators
            logger.info(f"DATA QUALITY INDICATORS:")
            logger.info(f"Data Points: {analysis.get('data_points', len(processed_df))}")
            logger.info(f"Missing Ratio: {analysis.get('missing_ratio', 0)*100:.2f}%")
            if analysis.get("insufficient_data"):
                logger.info(f"WARNING: Insufficient data for full analysis")
        
        # Global Uncovered Breakdown
        logger.info("GLOBAL PENALTY AGGREGATION:")
        hw_global = coverage_data.get("holt_winters", {}).get("uncovered_breakdown", {})
        lstm_global = coverage_data.get("lstm", {}).get("uncovered_breakdown", {})
        
        if hw_global:
            logger.info(f"Holt-Winters Average Penalties Across Parameters:")
            for reason, penalty in sorted(hw_global.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{reason.replace('_', ' ').title()}: -{penalty:.2f}%")
        
        if lstm_global:
            logger.info(f"LSTM Average Penalties Across Parameters:")
            for reason, penalty in sorted(lstm_global.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{reason.replace('_', ' ').title()}: -{penalty:.2f}%")
        
        # Warnings Summary
        warnings = self.preprocessing_report.get("warnings", [])
        if warnings:
            logger.info("WARNINGS SUMMARY:")
            
            for i, warning in enumerate(warnings, 1):
                logger.info(f"  {i}. {warning}")
        logger.info("END OF DETAILED COVERAGE ANALYSIS")
    
    def _optimize_volatility_parameters(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Optimize volatility window and alpha ranges for adaptive exponential smoothing
    
        Tests different configurations to find optimal settings for each parameter
        """
        
        logger.info("Optimizing volatility parameters for adaptive smoothing...")
        
        # Parameters that use adaptive smoothing
        adaptive_params = []
        for param in self.options.get("columns_to_process", []):
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            if param_config.get("smoothing_method") == "adaptive_exponential":
                adaptive_params.append(param)
        
        if not adaptive_params:
            logger.info("No parameters configured for adaptive smoothing - skipping optimization")
            return
        
        # Test configurations
        test_windows = [15, 30, 45]  # Volatility window sizes to test
        test_ranges = {
            "conservative": [0.15, 0.35],  # Low volatility parameters
            "moderate": [0.20, 0.50],      # Medium volatility parameters  
            "aggressive": [0.25, 0.65]     # High volatility parameters
        }
        
        optimization_results = {}
        
        for param in adaptive_params:
            if param not in df.columns:
                continue
            
            logger.info(f"Optimizing {param}...")
            
            series = df[param].dropna()
            if len(series) < 100:
                logger.warning(f"{param}: Insufficient data for optimization ({len(series)} < 100)")
                continue
            
            best_config = None
            best_score = -np.inf
            
            # Test all combinations
            for window in test_windows:
                for range_name, alpha_range in test_ranges.items():
                    try:
                        # Temporarily set config for testing
                        test_config = {
                            "volatility_window": window,
                            "alpha_range": alpha_range
                        }
                        
                        # Test this configuration
                        score = self._test_adaptive_config(series, param, test_config)
                        
                        logger.info(f"  {param}: window={window}, range={alpha_range} → score={score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_config = test_config
                            
                    except Exception as e:
                        logger.warning(f"Error testing {param} config window={window}, range={alpha_range}: {str(e)}")
                        continue
            
            # Update parameter config with best settings
            if best_config:
                param_configs = self.options.setdefault("parameter_configs", {})
                param_configs.setdefault(param, {}).update(best_config)
                
                optimization_results[param] = {
                    "optimal_window": best_config["volatility_window"],
                    "optimal_range": best_config["alpha_range"],
                    "score": best_score
                }
                
                logger.info(f"{param}: Optimal config → window={best_config['volatility_window']}, "
                        f"range={best_config['alpha_range']}, score={best_score:.3f}")
            else:
                logger.warning(f"{param}: No valid configuration found, keeping defaults")
        
        # Store results in preprocessing report
        self.preprocessing_report["adaptive_optimization"] = optimization_results
        
        logger.info(f"Volatility optimization completed for {len(optimization_results)} parameters")
    
    def _test_adaptive_config(self, series: pd.Series, param: str, test_config: dict) -> float:
        """
        Test a specific adaptive configuration and return quality score
        
        Args:
            series: Parameter data
            param: Parameter name
            test_config: Dictionary with 'volatility_window' and 'alpha_range'
            
        Returns:
            Combined score (higher = better)
        """
        try:
            # Temporarily update config for testing
            original_config = self.options.get("parameter_configs", {}).get(param, {}).copy()
            
            # Apply test config
            if "parameter_configs" not in self.options:
                self.options["parameter_configs"] = {}
            if param not in self.options["parameter_configs"]:
                self.options["parameter_configs"][param] = {}
                
            self.options["parameter_configs"][param].update(test_config)
            
            # Apply adaptive smoothing with test config
            smoothed_series = self._smooth_exponential_adaptive(series, param)
            
            # Calculate quality metrics
            original_values = series.values
            smoothed_values = smoothed_series.values
            
            # Align arrays (in case of length mismatch)
            min_len = min(len(original_values), len(smoothed_values))
            original_values = original_values[:min_len]
            smoothed_values = smoothed_values[:min_len]
            
            # Calculate GCV (will use adaptive EDF calculation)
            gcv_score = self._calculate_gcv(original_values, smoothed_values, param)
            
            # Calculate trend preservation
            trend_preservation = self._calculate_trend_preservation(original_values, smoothed_values)
            
            # Combined score (prioritize trend preservation for NASA data)
            combined_score = (trend_preservation * 100 * 1.5) - (gcv_score * 0.3)
            
            # Restore original config
            self.options["parameter_configs"][param] = original_config
            
            return combined_score
            
        except Exception as e:
            logger.warning(f"Error testing adaptive config for {param}: {str(e)}")
            return -1000.0  # Very low score for failed tests
    
    def _calculate_trend_penalty(self, trend_preservation: float) -> float:
        """
        Calculate penalty based on trend preservation loss
        
        Args:
            trend_preservation: Trend agreement ratio (0.0 to 1.0)
        
        Returns:
            Penalty percentage (0.0 to 20.0)
        
        Rationale:
        - Trend preservation measures how well smoothing maintains directional patterns
        - Low preservation means forecasting models will learn from distorted patterns
        - Critical for both Holt-Winters (trend component) and LSTM (sequential learning)
        
        Thresholds:
        - ≥90%: Excellent preservation, no penalty
        - 85-90%: Minor loss, models still reliable (-3%)
        - 80-85%: Moderate loss, noticeable forecast degradation (-7%)
        - 75-80%: Significant loss, high forecast uncertainty (-12%)
        - <75%: Severe loss, forecasting unreliable (-20%)
        """
        if trend_preservation >= 0.90:
            return 0.0
        elif trend_preservation >= 0.85:
            return 3.0
        elif trend_preservation >= 0.80:
            return 7.0
        elif trend_preservation >= 0.75:
            return 12.0
        else:
            return 20.0
    
    def _calculate_seasonality_penalty(
        self, 
        seasonal_strength: float
    ) -> float:
        """
         Calculate gradient-based seasonality penalty for Holt-Winters
    
        Args:
            seasonal_strength: Seasonal variance ratio (0.0 to 1.0)
        
        Returns:
            Penalty percentage (0.0 to 25.0)
        
        Rationale:
        - Holt-Winters Seasonal method requires clear seasonal patterns
        - Weak/absent seasonality causes poor seasonal component estimation
        - Model may overfit noise or fail to capture true patterns
        
        Thresholds (based on seasonal decomposition literature):
        - <0.1: No seasonality, HW inappropriate (-25%)
        - 0.1-0.2: Very weak, HW struggles (-20%)
        - 0.2-0.3: Weak but usable with caution (-12%)
        - 0.3-0.5: Acceptable, not optimal (-5%)
        - ≥0.5: Strong seasonality, HW ideal (0%)
        
        Note: LSTM is less affected by seasonality, so this penalty
        is applied with reduced weight for LSTM coverage calculation
        """
        if seasonal_strength < 0.1:
            return 25.0
        elif seasonal_strength < 0.2:
            return 20.0
        elif seasonal_strength < 0.3:
            return 12.0
        elif seasonal_strength < 0.5:
            return 5.0
        else:
            return 0.0
        
    def _optimize_parameter_alphas(self, df: pd.DataFrame) -> None:
        logger.info("Starting adaptive alpha optimization...")

        params = self.options.get("columns_to_process", [])
        alpha_range = self.options.get(
            "alpha_optimization_range",
            [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        )

        optimization_results = {}

        # Logging overview
        logger.info(f"Alpha range for testing: {alpha_range}")
        logger.info("Focus parameters: T2M (target trend preservation >75%)")

        for param in params:
            if param not in df.columns:
                continue

            param_config = self.options.get("parameter_configs", {}).get(param, {})
            smoothing_method = param_config.get("smoothing_method", self.options.get("smoothing_method", "exponential"))

            # SKIP parameters that use adaptive_exponential (they use volatility optimization instead)
            if smoothing_method == "adaptive_exponential":
                logger.info(f"{param}: Skipping (uses adaptive optimization)")
                continue
                
            # Skip parameters without exponential smoothing
            if smoothing_method is None or smoothing_method != "exponential":
                logger.info(f"{param}: Skipping (no exponential smoothing)")
                continue

            # Current alpha
            current_alpha = param_config.get(
                "exponential_alpha",
                self.options.get("exponential_alpha", 0.20)
            )

            # Find optimal alpha
            optimal_alpha = self._find_optimal_alpha_for_parameter(
                df,
                param,
                alpha_range
            )

            # Ensure config structure exists
            if "parameter_configs" not in self.options:
                self.options["parameter_configs"] = {}

            if param not in self.options["parameter_configs"]:
                self.options["parameter_configs"][param] = {}

            # Update alpha
            self.options["parameter_configs"][param]["exponential_alpha"] = optimal_alpha

            optimization_results[param] = {
                "original_alpha": current_alpha,
                "optimized_alpha": optimal_alpha,
                "improvement": f"alpha {current_alpha:.3f} -> {optimal_alpha:.3f}"
            }

            # Logging results
            if optimal_alpha > current_alpha:
                logger.info(
                    f"{param}: alpha {current_alpha:.3f} -> {optimal_alpha:.3f} "
                    "(increased for better trend preservation)"
                )
            elif optimal_alpha < current_alpha:
                logger.info(
                    f"{param}: alpha {current_alpha:.3f} -> {optimal_alpha:.3f} "
                    "(reduced for better smoothing balance)"
                )
            else:
                logger.info(
                    f"{param}: alpha {current_alpha:.3f} unchanged (optimal)"
                )

        # Store results in report
        self.preprocessing_report["alpha_optimization"] = optimization_results

        logger.info(
            f"Alpha optimization complete: {len(optimization_results)} parameters optimized"
        )
        
    def _find_optimal_alpha_for_parameter(
        self,
        df: pd.DataFrame,
        param: str,
        alpha_range: List[float]
    ) -> float:
        """
        Test multiple alpha values and return the optimal one.

        Scoring:
        - Standard parameters: (trend_preservation * 1.0) - (GCV * 0.5)
        - Problematic parameters (RH2M, ALLSKY_SFC_SW_DWN):
        (trend_preservation * 1.5) - (GCV * 0.3)
        """

        if param not in df.columns:
            logger.warning(f"Parameter {param} not in dataframe - using default alpha")
            return self.options.get("exponential_alpha", 0.20)

        series = df[param].dropna()

        # Minimum data requirement
        if len(series) < 100:
            logger.warning(
                f"Parameter {param}: insufficient data ({len(series)} points < 100) - using default alpha"
            )
            return self.options.get("exponential_alpha", 0.20)

        # Higher defaults for problematic parameters
        if param in ["RH2M", "ALLSKY_SFC_SW_DWN"]:
            best_alpha = 0.30
        else:
            best_alpha = self.options.get("exponential_alpha", 0.20)

        best_score = -np.inf
        test_results = []

        logger.info(f"     Testing {len(alpha_range)} alpha values for {param}...")

        for alpha in alpha_range:

            performance = self._test_alpha_performance(series.values, alpha)

            if performance is None:
                continue

            gcv_score = performance.get("gcv_score", 10.0)
            trend_pct = performance.get("trend_preservation_pct", 0.0)

            # Enhanced scoring
            if param in ["RH2M", "ALLSKY_SFC_SW_DWN"]:
                combined_score = (trend_pct * 1.5) - (gcv_score * 0.3)
            else:
                combined_score = (trend_pct * 1.0) - (gcv_score * 0.5)

            test_results.append({
                "alpha": alpha,
                "gcv": gcv_score,
                "trend": trend_pct,
                "score": combined_score
            })

            logger.info(
                f"{param} α={alpha}: GCV={gcv_score:.4f}, Trend={trend_pct:.1f}%, Score={combined_score:.2f}"
            )

            if combined_score > best_score:
                best_score = combined_score
                best_alpha = alpha

        if test_results:
            best_result = max(test_results, key=lambda x: x["score"])
            logger.info(
                f"BEST for {param}: α={best_result['alpha']}, Trend={best_result['trend']:.1f}%"
            )

        return best_alpha

    def _test_alpha_performance(
        self,
        original_series: np.ndarray,
        alpha: float
    ) -> Optional[Dict[str, Any]]:
        """
        Test a single alpha value and return GCV + trend preservation metrics
        """
        
        try:
            # Validate input size
            if len(original_series) < 100:  # Match minimum from _find_optimal_alpha_for_parameter
                return None
            
            # Check for NaN issues - both all NaN and too many NaN
            nan_count = np.isnan(original_series).sum()
            if nan_count == len(original_series):
                logger.warning(f"All-NaN data detected for α={alpha}")
                return None
            
            if nan_count > len(original_series) * 0.5:  # More than 50% NaN
                logger.warning(f"Too many NaN values ({nan_count}/{len(original_series)}) for α={alpha}")
                return None

            # Validate alpha range (should already be validated, but defensive)
            if alpha <= 0 or alpha >= 1:
                logger.warning(f"Invalid alpha value: {alpha} (must be between 0 and 1)")
                return None
            
            # Apply exponential smoothing on cleaned series
            series_pd = pd.Series(original_series).dropna()  # Drop NaN before smoothing
            
            if len(series_pd) < 100:  # Check again after dropping NaN
                logger.warning(f"Insufficient valid data after dropping NaN for α={alpha}")
                return None
                
            smoothed_series = series_pd.ewm(alpha=alpha, adjust=False).mean().values
            original_clean = original_series[~np.isnan(original_series)]
            
            # Ensure same length
            min_len = min(len(original_clean), len(smoothed_series))
            original_clean = original_clean[:min_len]
            smoothed_series = smoothed_series[:min_len]
            
            gcv_score = self._calculate_gcv(original_clean, smoothed_series, param=None)
            trend_preservation = self._calculate_trend_preservation(original_clean, smoothed_series)
            
            return {
                "alpha": alpha,
                "gcv_score": gcv_score,
                "trend_preservation_pct": trend_preservation * 100,
                "combined_score": (trend_preservation * 100) - (gcv_score * 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Error testing α={alpha}: {str(e)}")
            return None
        
    def _calculate_holt_winters_coverage(
        self, 
        large_gaps, 
        extreme_outliers, 
        seasonality, 
        stationarity, 
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate Holt-Winters model coverage with refined penalties
        
        Phase 1 Improvements:
        1. Added independent trend preservation penalty
        2. Gradient-based seasonality penalty
        3. Integrated GCV smoothing quality
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality
        smoothing_quality = self._get_smoothing_quality(param)
        gcv_penalty = smoothing_quality["penalty"]
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation ( Phase 1)
        trend_value = smoothing_quality.get("trend_value", 0.0)
        
        if trend_value > 0:  # Only if smoothing was applied
            trend_penalty = self._calculate_trend_penalty(trend_value)
            
            if trend_penalty > 0:
                base_coverage -= trend_penalty
                uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
                
                # Log warning for severe trend loss
                if trend_penalty >= 12.0:
                    logger.warning(
                        f"Parameter {param}: Severe trend preservation loss "
                        f"({trend_value*100:.1f}%) - High forecast uncertainty"
                    )
        
        # PENALTY 3: Large Gaps
        large_gap_penalty = large_gaps.get("impact_percentage", 0)
        
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 4: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 5: Weak Seasonality (REFINED - Phase 1)
        seasonal_strength = seasonality.get("seasonal_strength", 0)
        has_clear_seasonality = seasonality.get("has_clear_seasonality", False)
        
        # Use refined gradient-based penalty
        seasonal_penalty = self._calculate_seasonality_penalty(seasonal_strength)
        
        if seasonal_penalty > 0:
            base_coverage -= seasonal_penalty
            uncovered_reasons["weak_seasonality"] = round(seasonal_penalty, 2)
            
            # Log info about seasonality quality
            if seasonal_penalty >= 20.0:
                logger.warning(
                    f"Parameter {param}: Very weak seasonality "
                    f"(strength={seasonal_strength:.3f}) - Holt-Winters may be inappropriate"
                )
            elif seasonal_penalty >= 12.0:
                logger.info(
                    f"Parameter {param}: Weak seasonality "
                    f"(strength={seasonal_strength:.3f}) - Holt-Winters suboptimal"
                )
        
        # PENALTY 6: Non-Stationarity
        if not stationarity.get("is_stationary", False):
            stationarity_penalty = 5.0
            base_coverage -= stationarity_penalty
            uncovered_reasons["non_stationary"] = round(stationarity_penalty, 2)
        
        # PENALTY 7: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        # PENALTY 8: Compound Penalty (if multiple serious issues)
        issue_count = 0
        
        if large_gap_penalty > 5:
            issue_count += 1
        if outlier_penalty > 3:
            issue_count += 1
        if seasonal_penalty >= 12:  # refined threshold
            issue_count += 1
        if gcv_penalty > 10:
            issue_count += 1
        if trend_penalty >= 12:  #  count trend as serious issue
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(
                f"Parameter {param}: {issue_count} serious issues detected "
                f"- Compound penalty applied (forecasting highly uncertain)"
            )
        
        # FINAL COVERAGE
        final_coverage = max(0, base_coverage)
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
    
    
    def _calculate_lstm_coverage(
        self, 
        large_gaps, 
        extreme_outliers, 
        precipitation, 
        stationarity, 
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate LSTM model coverage with refined penalties
        
        Phase 1 Improvements:
        1. Added trend preservation penalty (slightly reduced weight vs HW)
        2. Integrated GCV smoothing quality
        
        Note: LSTM is less sensitive to seasonality than Holt-Winters,
        so seasonality penalty is not applied here
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality (80% weight vs HW)
        smoothing_quality = self._get_smoothing_quality(param)
        gcv_penalty = smoothing_quality["penalty"] * 0.8  # Slightly less critical than HW
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation (Phase 1, 80% weight)
        trend_value = smoothing_quality.get("trend_value", 0.0)
        
        if trend_value > 0:  # Only if smoothing was applied
            trend_penalty_full = self._calculate_trend_penalty(trend_value)
            trend_penalty = trend_penalty_full * 0.8  # 80% weight for LSTM
            
            if trend_penalty > 0:
                base_coverage -= trend_penalty
                uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
        
        # PENALTY 3: Large Gaps (80% weight vs HW)
        large_gap_penalty = large_gaps.get("impact_percentage", 0) * 0.8
        
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 4: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 5: Precipitation Extremes (if applicable)
        precip_penalty = 0
        
        if precipitation:
            precip_penalty = precipitation.get("range_impact", 0) * 10
            
            if precip_penalty > 0:
                base_coverage -= precip_penalty
                uncovered_reasons["precipitation_extremes"] = round(precip_penalty, 2)
        
        # PENALTY 6: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        # PENALTY 7: Compound Penalty (if multiple serious issues)
        issue_count = 0
        
        if large_gap_penalty > 4:  # Slightly lower threshold than HW
            issue_count += 1
        if outlier_penalty > 3:
            issue_count += 1
        if precip_penalty > 5:
            issue_count += 1
        if gcv_penalty > 8:
            issue_count += 1
        if trend_penalty >= 10:  #  count trend as serious issue (80% of 12)
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(
                f"Parameter {param}: {issue_count} serious issues detected "
                f"- Compound penalty applied (LSTM forecasting uncertain)"
            )
        
        # FINAL COVERAGE
        final_coverage = max(0, base_coverage)
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }

    def _aggregate_uncovered_breakdown(self, per_parameter_results: Dict, model_type: str) -> Dict[str, float]:
        """Aggregate uncovered breakdown across all parameters"""
        
        uncovered_key = f"{model_type}_uncovered"
        all_reasons = {}
        
        for param_result in per_parameter_results.values():
            if isinstance(param_result, dict) and uncovered_key in param_result:
                for reason, percentage in param_result[uncovered_key].items():
                    if reason not in all_reasons:
                        all_reasons[reason] = []
                    all_reasons[reason].append(percentage)
        
        # Average the percentages across parameters
        aggregated = {}
        for reason, percentages in all_reasons.items():
            aggregated[reason] = round(sum(percentages) / len(percentages), 2)
        
        return aggregated

    def _determine_model_suitability(self, coverage_percentage: float) -> str:
        """Determine model suitability based on coverage percentage"""
        if coverage_percentage >= 85:
            return "excellent"
        elif coverage_percentage >= 75:
            return "good"
        elif coverage_percentage >= 60:
            return "fair"
        else:
            return "poor"
    
    def _calculate_adaptive_alpha(
        self,
        series: pd.Series,
        param: str
    )-> np.ndarray:
        """
        Calculate adaptive alpha values based on local volatility
        
        Args:
            series: Time series data
            param: Parameter name for configuration
            
        Returns:
            Array of alpha values (one per data point)
        """
        try:
            # Get parameter specific config
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            alpha_range = param_config.get("alpha_range", [0.15, 0.45])  # Default range
            volatility_window = param_config.get("volatility_window", 30)  # Default 30 days
            alpha_min, alpha_max = alpha_range
            
            # Calculate rolling volatility (standard deviation)
            volatility = series.rolling(window=volatility_window, min_periods=1).std()
            
            # Fill initial NaN values with backward fill
            volatility = volatility.fillna(method='bfill')
            
            # Normalize volatility to [0, 1] using percentiles to handle outliers
            v_10th = volatility.quantile(0.1)
            v_90th = volatility.quantile(0.9)
            
            # Avoid division by zero
            if v_90th == v_10th:
                volatility_norm = pd.Series([0.5] * len(volatility), index=volatility.index)
            else:
                volatility_norm = (volatility - v_10th) / (v_90th - v_10th)
                volatility_norm = volatility_norm.clip(0, 1)
            
            # Map to alpha range: low volatility = low alpha, high volatility = high alpha
            alphas = alpha_min + volatility_norm * (alpha_max - alpha_min)
            
            # Ensure alphas are within valid range [0.01, 0.99]
            alphas = alphas.clip(0.01, 0.99)
            
            logger.info(f"{param}: Adaptive alpha range [{alpha_min:.2f}, {alpha_max:.2f}], "
                    f"actual range [{alphas.min():.3f}, {alphas.max():.3f}]")
            
            return alphas.values
        except Exception as e:
            logger.warning(f"Error calculating adaptive alpha for {param}: {str(e)}")
            # Fallback to fixed alpha
            fallback_alpha = param_config.get("exponential_alpha", 0.25)
            return np.full(len(series), fallback_alpha)
        
    def _smooth_exponential_adaptive(self, series: pd.Series, param: str) -> pd.Series:
        """
        Apply exponential smoothing with time-varying alpha values
        
        Args:
            series: Time series data
            param: Parameter name
            
        Returns:
            Adaptively smoothed time series
        """
        try:
            # Get adaptive alpha array
            alphas = self._calculate_adaptive_alpha(series, param)
            
            # Initialize result array
            smoothed = np.zeros(len(series))
            smoothed[0] = series.iloc[0]  # First value unchanged
            
            # Apply time-varying EWMA: s_t = alpha_t * x_t + (1-alpha_t) * s_{t-1}
            for i in range(1, len(series)):
                if pd.notna(series.iloc[i]):  # Only smooth non-NaN values
                    smoothed[i] = alphas[i] * series.iloc[i] + (1 - alphas[i]) * smoothed[i-1]
                else:
                    smoothed[i] = smoothed[i-1]  # Carry forward for NaN
            
            # Log adaptive smoothing stats
            avg_alpha = alphas.mean()
            alpha_std = alphas.std()
            
            logger.info(f"{param}: Adaptive smoothing applied - "
                    f"avg_alpha={avg_alpha:.3f}, alpha_std={alpha_std:.3f}")
            
            return pd.Series(smoothed, index=series.index)
            
        except Exception as e:
            logger.error(f"Error in adaptive smoothing for {param}: {str(e)}")
            # Fallback to regular exponential smoothing
            logger.info(f"{param}: Falling back to regular exponential smoothing")
            return self._smooth_exponential(series, param)

    def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to appropriate parameters"""
        logger.info("Applying conditional smoothing methods...")
        params = self.options.get("columns_to_process", [])
        smoothing_method = self.options.get("smoothing_method", "exponential")
        smoothing_summary = {}
        
        # Track smoothing decisions for better reporting
        smoothing_decisions = {
            "smoothed": [],
            "skipped": [],
            "reasons": {}
        }
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Check parameter-specific config
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            param_smoothing = param_config.get("smoothing_method", smoothing_method)
            
            # Get reason for detailed logging
            reason = param_config.get("reason", "default configuration")
            
            if param_smoothing is None:
                # Skip smoothing for this parameter
                logger.info(f"{param}: Skipping smoothing - {reason}")
                smoothing_summary[param] = "none"
                smoothing_decisions["skipped"].append(param)
                smoothing_decisions["reasons"][param] = reason
                continue
            
            # Apply smoothing based on method
            if param_smoothing == "exponential":
                alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
                logger.info(f"  {param}: Smoothing (α={alpha:.3f}) - {reason}")
                
                df[param] = self._smooth_exponential(df[param], param)
                smoothing_summary[param] = f"exponential (α={alpha:.3f})"
                smoothing_decisions["smoothed"].append(param)
                smoothing_decisions["reasons"][param] = f"α={alpha:.3f}, {reason}"
                
            elif param_smoothing == "adaptive_exponential":
                # Adaptive exponential smoothing
                alpha_range = param_config.get("alpha_range", [0.15, 0.45])
                volatility_window = param_config.get("volatility_window", 30)
                
                logger.info(f"  {param}: Adaptive smoothing (α_range={alpha_range}, window={volatility_window}) - {reason}")
                
                df[param] = self._smooth_exponential_adaptive(df[param], param)
                smoothing_summary[param] = f"adaptive_exponential (α={alpha_range[0]:.2f}-{alpha_range[1]:.2f})"
                smoothing_decisions["smoothed"].append(param)
                smoothing_decisions["reasons"][param] = f"adaptive α={alpha_range}, {reason}"
                
            elif param_smoothing == "moving_average":
                window = self.options.get("window_size", 5)
                logger.info(f"  {param}: Moving average (window={window}) - {reason}")
                
                df[param] = self._smooth_moving_average(df[param])
                smoothing_summary[param] = f"moving_average (w={window})"
                smoothing_decisions["smoothed"].append(param)
                smoothing_decisions["reasons"][param] = f"window={window}, {reason}"
        
        # Enhanced reporting with decision tracking
        self.preprocessing_report["smoothing"] = {
            "method": "conditional",  # Changed from single method
            "parameters_smoothed": smoothing_summary,
            "decisions": smoothing_decisions,
            "summary": {
                "total_parameters": len(params),
                "smoothed_count": len(smoothing_decisions["smoothed"]),
                "skipped_count": len(smoothing_decisions["skipped"])
            }
        }
        
        # Summary logging
        logger.info(f"Conditional smoothing completed:")
        logger.info(f"  - Smoothed: {len(smoothing_decisions['smoothed'])} parameters")
        logger.info(f"  - Skipped: {len(smoothing_decisions['skipped'])} parameters")
        
        return df
    
    def _validate_smoothing_method(self, df: pd.DataFrame) -> None:
        """
        Validate smoothing quality using GCV + Trend Preservation
        SIMPLIFIED - Remove redundant fields
        """
        logger.info("Validating smoothing quality...")

        params = self.options.get("columns_to_process", [])
        validation_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Check smoothing_summary instead of config
            smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})
            param_smoothing_applied = smoothing_summary.get(param, "none")
            
            # Skip parameters where smoothing was NOT applied
            if param_smoothing_applied == "none":
                # SIMPLIFIED: Don't store "no_smoothing" entries
                continue
            
            # Get baseline data
            if hasattr(self, 'pre_smoothing_data') and self.pre_smoothing_data is not None and param in self.pre_smoothing_data.columns:
                original = self.pre_smoothing_data[param].dropna()
                baseline_source = "pre_smoothing_data"
            else:
                logger.warning(f"Pre-smoothing data not available for {param} - using raw original data")
                if self.original_data is None or param not in self.original_data.columns:
                    logger.warning(f"Original data not available for {param}")
                    continue
                original = self.original_data[param].dropna()
                baseline_source = "original_data"
            
            # Post-smoothing data
            smoothed = df[param].dropna()
            
            # Align indices 
            common_idx = original.index.intersection(smoothed.index)
            if len(common_idx) < 30:
                continue
                
            original_aligned = original.loc[common_idx]
            smoothed_aligned = smoothed.loc[common_idx]
            
            # Calculate metrics
            gcv_score = self._calculate_gcv(
                original_aligned.values,
                smoothed_aligned.values,
                param
            )
            
            trend_agreement = self._calculate_trend_preservation(
                original_aligned.values,
                smoothed_aligned.values
            )
            
            quality_status = self._determine_smoothing_quality(
                gcv_score,
                trend_agreement
            )
            
            # SIMPLIFIED STRUCTURE - Only 3 essential fields
            validation_results[param] = {
                "gcv_score": round(float(gcv_score), 4),
                "trend_preservation_pct": round(float(trend_agreement * 100), 2),
                "quality_status": quality_status
                # REMOVED: smoothing_method (redundant with preprocessing_summary.smoothing)
                # REMOVED: data_points (same for all parameters)
                # REMOVED: baseline_source (internal detail)
            }
            
            logger.info(
                f"Parameter {param}: "
                f"GCV={gcv_score:.4f}, "
                f"Trend={trend_agreement*100:.2f}%, "
                f"Quality={quality_status.upper()}"
            )
            
            # Add warnings if quality is poor
            if quality_status == "poor":
                self.preprocessing_report["warnings"].append(
                    f"Parameter {param}: smoothing quality is poor "
                    f"(GCV={gcv_score:.3f}, Trend={trend_agreement*100:.1f}%)"
                )
        
        # Store in report
        self.preprocessing_report["smoothing_validation"] = validation_results
        logger.info(f"Smoothing validation completed for {len(validation_results)} parameters")
        
    def _determine_smoothing_quality(self, gcv: float, trend_preservation: float) -> str:
        """
        Determine overall smoothing quality based on GCV and windowed trend preservation
        
        Updated thresholds for windowed trend agreement (typically higher values):
        - Windowed trend preservation focuses on weekly patterns vs daily noise
        - Expect 85-95% agreement for good smoothing vs 60-70% with daily comparison
        
        Args:
            gcv: Generalized Cross-Validation score (lower = better)  
            trend_preservation: Windowed trend agreement ratio (0.0 to 1.0, higher = better)
        
        Returns:
            Quality status: "excellent", "good", "fair", "poor"
        """
        # Higher thresholds for windowed trend preservation
        if gcv < 2.0 and trend_preservation > 0.85:
            return "excellent"
        elif gcv < 4.0 and trend_preservation > 0.80:
            return "good"
        elif gcv < 10.0 and trend_preservation > 0.75:
            return "fair"
        else:
            return "poor"
        
    def _calculate_gcv(self, original: np.ndarray, smoothed: np.ndarray, param: str) -> float:
        """
        Calculate Generalized Cross-Validation score
        Lower GCV = better smoothing balance between fit and complexity
        
        GCV penalizes both:
        - Poor fit (high MSE)
        - Over-smoothing (high effective degrees of freedom)
        
        Update for support adaptive exponential smoothing EDF calculation
        """
        n = len(original)
            
        # Calculate Mean Squared Error
        mse = np.mean((original - smoothed) ** 2)
        
        # Get parameter-specific smoothing method (not global!)
        param_config = self.options.get("parameter_configs", {}).get(param, {})
        smoothing_method = param_config.get(
            "smoothing_method", 
            self.options.get("smoothing_method", "exponential")  # Fallback to global
        )
    
        # Handle case where smoothing_method is None (defensive programming)
        if smoothing_method is None:
            logger.warning(f"GCV called for parameter {param} with no smoothing method")
            return 0.0  # Return 0 to indicate no smoothing was applied
        
        # Estimate effective degrees of freedom based on smoothing method
        if smoothing_method == "moving_average":
            window_size = self.options.get("window_size", 5)
            edf = window_size
            
        elif smoothing_method == "exponential":
            # Get parameter-specific alpha correctly
            if param:
                param_config = self.options.get("parameter_configs", {}).get(param, {})
                alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
            else:
                alpha = self.options.get("exponential_alpha", 0.15)
            
            # Validate alpha range
            alpha = max(0.01, min(0.99, alpha))  # Clamp to valid range
            
            # CORRECTED FORMULA for Exponential Weighted Moving Average (EWMA)
            edf = 1.0 / (2.0 - alpha)
            
            # Ensure reasonable bounds
            edf = max(1.0, min(edf, n / 3.0))  # EDF should be between 1 and n/3
            
        elif smoothing_method == "adaptive_exponential":
            # Calculate EDF for adaptive exponential smoothing
            alpha_range = param_config.get("alpha_range", [0.15, 0.45])
            alpha_min, alpha_max = alpha_range
            
            # Use average alpha for EDF estimation
            alpha_avg = (alpha_min + alpha_max) / 2.0
            
            # Apply same formula as regular exponential
            edf_min = 1.0 / (2.0 - alpha_max)  # Min EDF (high alpha)
            edf_max = 1.0 / (2.0 - alpha_min)  # Max EDF (low alpha) 
            edf = (edf_min + edf_max) / 2.0    # Average EDF
            
            # Alternative: Could use actual alpha values if stored during smoothing
            # For now, using range average is simpler and sufficient
            
            # Ensure reasonable bounds
            edf = max(1.0, min(edf, n / 3.0))
            
            logger.debug(f"Adaptive GCV: α_range={alpha_range}, α_avg={alpha_avg:.3f}, edf={edf:.2f}")
            
        else:
            edf = 5  # Default conservative estimate
            if smoothing_method:
                logger.warning(f"Unknown smoothing method '{smoothing_method}' for parameter {param}, using default EDF=5")

        # GCV formula: MSE / (1 - edf/n)²
        denominator = (1.0 - edf / n) ** 2

        # Ensure denominator is not too small (prevents division explosion)
        if denominator <= 0.01:
            logger.warning(f"GCV denominator too small ({denominator:.4f}), clamping to 0.01")
            denominator = 0.01

        gcv = mse / denominator

        return gcv
    
    def _calculate_non_smoothed_coverage(
        self,
        large_gaps,
        extreme_outliers,
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate coverage for parameters that don't need smoothing
        
        Args:
            large_gaps: Large gaps analysis
            extreme_outliers: Extreme outliers analysis  
            coverage_analysis: Basic coverage data
            param: Parameter name
            
        Returns:
            Coverage dictionary for non-smoothed parameters
        """
        base_coverage = 95.0  # High base coverage - no smoothing needed
        uncovered_reasons = {}
        
        # PENALTY 1: Large Gaps
        large_gap_penalty = large_gaps.get("impact_percentage", 0)
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 2: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 3: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        final_coverage = max(0, base_coverage)
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
    
    def _calculate_trend_preservation(self, original: np.ndarray, smoothed: np.ndarray) -> float:
        """
        Calculate windowed trend direction agreement between original and smoothed data.
        
        Uses moving average windows to compare trend directions, making the metric
        less sensitive to daily fluctuations and more focused on meaningful patterns.
        
        Args:
            original: Original data array (after imputation & outliers, before smoothing)
            smoothed: Smoothed data array
            
        Returns:
            Trend agreement ratio (0.0 to 1.0) - higher values indicate better preservation
            
        Algorithm:
        1. Apply 7-day moving average to both series to smooth daily noise
        2. Calculate direction changes on the smoothed moving averages  
        3. Compare direction agreement, ignoring flat periods
        4. Return percentage of matching directions as ratio (0-1)
        """
        # Validate input lengths
        if len(original) != len(smoothed):
            logger.warning(f"Trend preservation: length mismatch (original={len(original)}, smoothed={len(smoothed)})")
            return np.nan
        
        if len(original) < 2:  # Need at least 2 points for diff
            logger.warning(f"Trend preservation: insufficient data (n={len(original)})")
            return np.nan
        
        window = self.options.get("trend_window_size", 7)  # Default 7 days

        if len(original) < window + 1:
            logger.debug(f"Trend preservation: data too short for {window}-day window (n={len(original)}) - assuming perfect")
            return 1.0

        # Calculate moving averages to smooth daily fluctuations
        orig_ma = pd.Series(original).rolling(window, center=True, min_periods=1).mean().values
        smooth_ma = pd.Series(smoothed).rolling(window, center=True, min_periods=1).mean().values
        
        # Remove any NaN values that might occur at edges
        valid = ~(np.isnan(orig_ma) | np.isnan(smooth_ma))
        orig_ma = orig_ma[valid]
        smooth_ma = smooth_ma[valid]
        
        if len(orig_ma) < 2:
            logger.debug(f"Trend preservation: insufficient valid data after windowing (n={len(orig_ma)})")
            return 1.0  # Assume perfect for very short series
        
        # Calculate trend directions on moving averages (less noisy than daily changes)
        original_diff = np.diff(orig_ma)
        smoothed_diff = np.diff(smooth_ma)
        
        # Get trend directions (+1 for increase, -1 for decrease, 0 for flat)
        original_direction = np.sign(original_diff)
        smoothed_direction = np.sign(smoothed_diff)
        
        # Exclude flat periods where both directions are zero
        non_zero_mask = (original_direction != 0) & (smoothed_direction != 0)
        
        if non_zero_mask.sum() == 0:
            # No clear trends detected in windowed data (all flat periods)
            # This indicates very stable data - smoothing preserved stability perfectly
            logger.debug(f"Trend preservation: No clear trends in {window}-day windows (stable data) - returning 1.0 (perfect)")
            return 1.0
        
        if non_zero_mask.sum() < 5:  # Very few trend changes in windowed data
            logger.debug(f"Trend preservation: Very few windowed trend changes ({non_zero_mask.sum()}) - may indicate stable data")
        
        # Calculate agreement between trend directions
        agreement = (original_direction[non_zero_mask] == smoothed_direction[non_zero_mask])
        trend_preservation = agreement.mean()
        
        # Log debug info for verification
        logger.debug(f"Windowed trend analysis: {non_zero_mask.sum()} trend changes, {trend_preservation:.3f} agreement")
        
        return trend_preservation
    
    def _smooth_moving_average(self, series: pd.Series) -> pd.Series:
        """Apply moving average smoothing"""
        window_size = self.options.get("window_size", 5)
        return series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    def _smooth_exponential(
        self,
        series: pd.Series,
        param: str = None
    ) -> pd.Series:
        """Apply exponential smoothing with parameter-specific alpha"""
        
        if param:
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
        else:
            alpha = self.options.get("exponential_alpha", 0.15)
        return series.ewm(alpha=alpha, adjust=True).mean()
    
    def _generate_quality_metrics(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
        """Generate quality metrics for preprocessing report"""
        logger.info("Generating quality metrics...")
        
        params = self.options.get("columns_to_process", [])
        
        # Calculate completeness
        total_cells = len(processed_df) * len(params)
        missing_cells = sum(processed_df[param].isna().sum() for param in params if param in processed_df.columns)
        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate data modification percentage
        original_count = len(original_df)
        processed_count = len(processed_df)
        records_removed = original_count - processed_count
        
        self.preprocessing_report["quality_metrics"] = {
            "original_records": int(original_count),
            "processed_records": int(processed_count),
            "records_removed": int(records_removed),
            "completeness_percentage": round(completeness, 2),
            "data_quality": "high" if completeness > 95 else "medium" if completeness > 90 else "low"
        }
        
        logger.info(f"Quality metrics: {completeness:.2f}% complete, {records_removed} records removed")
    
    def _calculate_and_save_decomposition(
        self,
        df: pd.DataFrame,
        preprocessing_id: ObjectId
    ) -> None:
        """
        Calculate seasonal decomposition and save ONE document per preprocessing
        containing all parameter.
        """
        logger.info("Calculating seasonal decomposition for all parameters...")
        params = self.options.get("columns_to_process", [])
        smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})

        parameters_dict = {}
        total_data_points = 0
        successful_params = []

        for param in params:
            if param not in df.columns:
                logger.warning(f"{param}: Not found in dataframe, skipping")
                continue

            if len(df) < 730:
                logger.warning(f"{param}: Insufficient data for decomposition ({len(df)} days < 730)")
                continue

            try:
                series = df[param].dropna()
                if len(series) < 730:
                    logger.warning(f"{param}: Insufficient valid data after dropping NaN ({len(series)} < 730)")
                    continue

                param_smoothing_applied = smoothing_summary.get(param, "none")
                data_type = "smoothed" if param_smoothing_applied != "none" else "original"
                logger.info(f"{param}: Decomposing {data_type} data with STL...")

                stl = STL(series, period=365, robust=True)
                decomposition = stl.fit()

                seasonal_var = np.var(decomposition.seasonal.dropna())
                residual_var = np.var(decomposition.resid.dropna())
                seasonal_strength = (
                    seasonal_var / (seasonal_var + residual_var)
                    if (seasonal_var + residual_var) != 0
                    else 0.0
                )

                data_array = []
                for i in range(len(series)):
                    idx_val = series.index[i]

                    # Resolve Date safely
                    if "Date" in df.columns:
                        try:
                            date_val = df.loc[idx_val, "Date"]
                        except Exception:
                            date_val = idx_val
                    else:
                        date_val = idx_val

                    if isinstance(date_val, (np.datetime64, pd.Timestamp)):
                        date_val = pd.Timestamp(date_val).to_pydatetime()

                    data_array.append({
                        "Date": date_val,
                        "original": float(series.iloc[i]) if pd.notna(series.iloc[i]) else None,
                        "trend": float(decomposition.trend.iloc[i]) if pd.notna(decomposition.trend.iloc[i]) else None,
                        "seasonal": float(decomposition.seasonal.iloc[i]) if pd.notna(decomposition.seasonal.iloc[i]) else None,
                        "residual": float(decomposition.resid.iloc[i]) if pd.notna(decomposition.resid.iloc[i]) else None
                    })

                parameters_dict[param] = {
                    "seasonal_strength": round(float(seasonal_strength), 3),
                    "metadata": {
                        "data_type": data_type,
                        "smoothing_method": param_smoothing_applied if param_smoothing_applied != "none" else None
                    },
                    "data": data_array
                }

                total_data_points += len(data_array)
                successful_params.append(param)
                logger.info(f"{param}: Decomposition successful ({len(data_array)} points)")

            except Exception as e:
                logger.error(f"{param}: Decomposition failed - {str(e)}")
                logger.error(traceback.format_exc())
                self.preprocessing_report["warnings"].append(f"Failed to decompose {param}: {str(e)}")
                continue

        if not parameters_dict:
            logger.warning("No parameters successfully decomposed. Skipping save.")
            self.preprocessing_report["decomposition_summary"] = {
                "parameters_decomposed": [],
                "total_documents": 0,
                "total_data_points": 0,
                "collection": "decomposition_report",
                "status": "no_data"
            }
            return

        combined_doc = {
            "preprocessing_id": preprocessing_id,
            "dataset_name": self.collection_name,
            "dataset_type": "nasa",
            "decomposition_method": "STL",
            "timestamp": datetime.now(),
            "parameters": parameters_dict,
        }

        try:
            sanitized_doc = self._sanitize_for_mongodb(combined_doc)
            result = self.db["decomposition_report"].insert_one(sanitized_doc)

            logger.info(
                f"Saved decomposition document: {len(parameters_dict)} parameters, "
                f"{total_data_points} points (ID: {result.inserted_id})"
            )

            self.preprocessing_report["decomposition_summary"] = {
                "parameters_decomposed": successful_params,
                "total_documents": 1,
                "total_data_points": total_data_points,
                "collection": "decomposition_report",
                "document_id": str(result.inserted_id),
                "status": "success",
                "storage_method": "one_document_per_preprocessing"
            }

        except Exception as e:
            logger.error(f"Failed to save decomposition to MongoDB: {str(e)}")
            logger.error(traceback.format_exc())

            self.preprocessing_report["decomposition_summary"] = {
                "parameters_decomposed": successful_params,
                "total_documents": 0,
                "total_data_points": total_data_points,
                "collection": "decomposition_report",
                "status": "save_failed",
                "error": str(e)
            }
        
    def _save_preprocessing_report(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        Save simplified preprocessing report to preprocessing_report collection
        Move optimization results and remove redundancy
        """
        try:
            # Prepare optimization summary (simplified)
            optimization_summary = {}
            
            # Adaptive optimization (simplified structure)
            adaptive_opt = self.preprocessing_report.get("adaptive_optimization", {})
            if adaptive_opt:
                optimization_summary["adaptive"] = {
                    param: {
                        "window": config["optimal_window"],
                        "range": config["optimal_range"]
                        # REMOVED: score (internal metric)
                    }
                    for param, config in adaptive_opt.items()
                }
            
            # Alpha optimization (if any)
            alpha_opt = self.preprocessing_report.get("alpha_optimization", {})
            if alpha_opt:
                optimization_summary["alpha"] = {
                    param: {
                        "original": config["original_alpha"],
                        "optimized": config["optimized_alpha"]
                        # REMOVED: improvement string (redundant)
                    }
                    for param, config in alpha_opt.items()
                }
            
            # Prepare simplified report document
            report_doc = {
                "dataset_type": "nasa",
                "original_collection_name": self.collection_name,
                "cleaned_collection_name": collection_name,
                "preprocessing_timestamp": datetime.now(),
                
                # ENHANCED Preprocessing Summary (with optimization)
                "preprocessing_summary": {
                    "missing_data": {
                        "tail_data_excluded": self.preprocessing_report.get("missing_data", {}).get("tail_data_excluded"),
                        "imputed_values": self.preprocessing_report.get("missing_data", {}).get("imputed_values", {})
                    },
                    "outliers": self.preprocessing_report.get("outliers", {}),
                    "smoothing": {
                        **self.preprocessing_report.get("smoothing", {}),
                        # MOVED: Add optimization results here
                        "optimization": optimization_summary
                    },
                    "gaps_summary": {
                        "total_gaps": self.preprocessing_report.get("gaps", {}).get("total_gaps", 0),
                        "small_gaps": self.preprocessing_report.get("gaps", {}).get("small_gaps", 0),
                        "medium_gaps": self.preprocessing_report.get("gaps", {}).get("medium_gaps", 0),
                        "large_gaps": self.preprocessing_report.get("gaps", {}).get("large_gaps", 0)
                    }
                },
                
                # Quality Metrics (keep full)
                "quality_metrics": self.preprocessing_report.get("quality_metrics", {}),
                
                # SIMPLIFIED Smoothing Validation (3 fields only per parameter)
                "smoothing_validation": self.preprocessing_report.get("smoothing_validation", {}),
                
                # SIMPLIFIED Model Coverage (no analysis_details, no empty uncovered objects)
                "model_coverage": self.preprocessing_report.get("model_coverage", {}),
                
                # REMOVED: alpha_optimization and adaptive_optimization (moved to preprocessing_summary)
                
                # Decomposition Summary ONLY (not the data)
                "decomposition_summary": self.preprocessing_report.get("decomposition_summary", {}),
                
                # Warnings (filtered to important ones)
                "warnings": [w for w in self.preprocessing_report.get("warnings", []) 
                            if "Large gap" in w or "quality is poor" in w],
                
                # Status
                "status": "success",
                
                # Record counts
                "record_count": {
                    "original": len(self.original_data) if self.original_data is not None else 0,
                    "processed": self.preprocessing_report.get("quality_metrics", {}).get("processed_records", 0)
                }
            }
            
            # Sanitize for MongoDB 
            sanitized_document = self._sanitize_for_mongodb(report_doc)
            
            # Insert into preprocessing_report collection
            result = self.db["preprocessing_report"].insert_one(sanitized_document)
            
            logger.info(f"SIMPLIFIED preprocessing report saved with ID: {result.inserted_id}")
            
            return {
                "status": "success",
                "report_id": result.inserted_id,
                "collection": "preprocessing_report"
            }
            
        except Exception as e:
            error_msg = f"Error saving preprocessing report: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": str(e)
            }
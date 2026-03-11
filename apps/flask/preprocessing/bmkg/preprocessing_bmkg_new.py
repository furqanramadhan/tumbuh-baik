from typing import Dict, List, Any, Optional
from certifi import where
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from pymongo import MongoClient
from bson import ObjectId
import logging
import traceback
import warnings

warnings.filterwarnings("ignore")

# config logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BmkgPreprocessingError(Exception):
    """Custom exception for BMKG preprocessing errors"""
    pass
class BmkgDataValidator:
    """Validate BMKG data before preprocessing"""
    
    def validate_dataset(
        self,
        db,
        collection_name: str,
    )-> Dict[str, Any]:
        """Validate BMKG dataset with criteria:
        
        Validates:
        - Collection existence and non-empty
        - Required fields presence
        - Physical relationship 
        - Temporal continuity
        """
        try:
            # Check collection exist
            if collection_name not in db.list_collection_names():
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} does not exist."],
                }
                
            # Get total records 
            total_records = db[collection_name].count_documents({})
            if total_records == 0:
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} is empty."],
                }
            logger.info(f"Validating {total_records} records in {collection_name}...")
            
            # Sample validation (100 records)
            sample_size = min(100, total_records)
            sample_docs = list(db[collection_name].find().limit(sample_size))
            
            # Define required fields with specified requirements
            field_requirements = {
                'Date': {
                    'type': (str, pd.Timestamp, datetime), 
                    'required': True, 
                    'description': 'Date'
                },
                'TN': {
                    'type': (int, float),
                    'range': (-10, 35),
                    'required': True,
                    'description': 'Minimum Temperature (°C)'
                },
                'TX': {
                    'type': (int, float), 
                    'range': (-5, 45), 
                    'required': True, 
                    'description': 'Maximum Temperature (°C)'
                },
                'TAVG': {
                    'type': (int, float), 
                    'range': (-5, 40), 
                    'required': True, 
                    'description': 'Average Temperature (°C)'
                },
                'RH_AVG': {
                    'type': (int, float), 
                    'range': (0, 100), 
                    'required': True, 
                    'description': 'Average Humidity (%)'
                },
                'RR': {
                    'type': (int, float), 
                    'range': (0, 500), 
                    'required': True, 
                    'description': 'Rainfall (mm)'
                },
                'SS': {
                    'type': (int, float), 
                    'range': (0, 14), 
                    'required': True, 
                    'description': 'Sunshine Duration (hours)'
                },
                'FF_X': {
                    'type': (int, float), 
                    'range': (0, 50), 
                    'required': True, 
                    'description': 'Maximum Wind Speed (m/s)'
                },
                'DDD_X': {
                    'type': (int, float), 
                    'range': (0, 360), 
                    'required': True, 
                    'description': 'Wind Direction (degrees)'
                },
                'FF_AVG': {
                    'type': (int, float), 
                    'range': (0, 30), 
                    'required': True, 
                    'description': 'Average Wind Speed (m/s)'
                },
                'DDD_CAR': {
                    'type': str, 
                    'required': True, 
                    'description': 'Wind Direction (cardinal)'
                }
            }
            
            # Validation results
            validation_errors = []
            validation_warnings = []
            field_stats = {}
            
            # Check schema
            first_doc = sample_docs[0]
            missing_fields = [
                field for field in field_requirements
                if field not in first_doc
            ]
            
            if missing_fields:
                validation_errors.append(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )
            
            # Validate each field
            for field, requirements in field_requirements.items():
                if field not in first_doc:
                    continue
                
                field_values = []
                invalid_types = 0
                out_of_range = 0
                missing_values = 0
                
                for doc in sample_docs:
                    value = doc.get(field)
                    
                    # Check missing/null
                    if value is None or value == '' or \
                       (isinstance(value, float) and np.isnan(value)):
                        missing_values += 1
                        continue

                    # Check BMKG fill values (8888, 9999)
                    if isinstance(value, (int, float)) and \
                       value in [8888.0, 9999.0, 8888, 9999]:
                        missing_values += 1
                        continue

                    # Check data type
                    if not isinstance(value, requirements['type']):
                        invalid_types += 1
                        continue

                    # Check range
                    if 'range' in requirements and \
                       isinstance(value, (int, float)):
                        min_val, max_val = requirements['range']
                        if not (min_val <= value <= max_val):
                            out_of_range += 1

                    field_values.append(value)
                
                # calculate stats
                missing_pct = (missing_values / sample_size) * 100
                field_stats[field] = {
                    'missing_count': missing_values,
                    'missing_percentage': round(missing_pct, 2),
                    'invalid_types': invalid_types,
                    'out_of_range': out_of_range,
                    'valid_values': len(field_values),
                    'description': requirements.get('description', '')
                }
                
                # Generate warnings
                if missing_pct > 50:
                    validation_warnings.append(
                        f"Field '{field}' has {missing_pct:.1f}% missing values (high!)"
                    )
                elif missing_pct > 20:
                    validation_warnings.append(
                        f"Field '{field}' has {missing_pct:.1f}% missing values"
                    )

                if invalid_types > 0:
                    validation_errors.append(
                        f"Field '{field}' has {invalid_types} records with invalid data types"
                    )

                if out_of_range > sample_size * 0.1:
                    validation_warnings.append(
                        f"Field '{field}' has {out_of_range} outliers in sample"
                    )
            
            # Physical relationship validation
            relationship_warnings = self._validate_physical_relationships(sample_docs)
            validation_warnings.extend(relationship_warnings)
            
            # Temporal continuity validation
            date_validation = self._validate_temporal_continuity(
                db, collection_name, total_records
            )
            
            if date_validation.get('warnings'):
                validation_warnings.extend(date_validation['warnings'])
            
            # Minimum data requirement for (STL needs 2+ years)
            if total_records < 730:  # 2 years
                validation_warnings.append(
                    f"Dataset has only {total_records} records (<2 years). "
                    "STL decomposition may not work optimally."
                )
            
            # Determine validation status
            is_valid = len(validation_errors) == 0

            return {
                'valid': is_valid,
                'total_records': total_records,
                'sample_size': sample_size,
                'errors': validation_errors,
                'warnings': validation_warnings,
                'fields_statistics': field_stats,
                'temporal_info': date_validation.get('temporal_info', {}),
                'message': "Dataset validation completed" if is_valid 
                          else "Dataset validation failed"
            }
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
            }
    def _validate_temporal_continuity(
        self,
        db,
        collection_name: str,
        total_records: int,
    ) -> Dict[str,Any]:
        """Validate temporal continuity and coverage"""
    
        try:
            # Get first and last dates
            first_record = db[collection_name].find_one(sort=[('Date', 1)])
            last_record = db[collection_name].find_one(sort=[('Date', -1)])
            
            if not first_record or not last_record:
                return {
                    'valid': False,
                    'warnings': ["Cannot determine date range from dataset."]
                }
            
            # Extract dates
            first_date = first_record.get('Date')
            last_date = last_record.get('Date')
            
            # Convert to datetime
            if isinstance(first_date, str):
                first_date = pd.to_datetime(first_date)
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
                
            # calculate date range
            date_range = (last_date - first_date).days + 1
            warnings = []
            
              # Check for gaps
            if total_records < date_range * 0.8:
                gap_pct = ((date_range - total_records) / date_range) * 100
                warnings.append(
                    f"Potential data gaps: Expected ~{date_range} records "
                    f"but found {total_records} ({gap_pct:.1f}% gap)"
                )
            
            # Check duplicates
            pipeline = [
                {'$group': {'_id': '$Date', 'count': {'$sum': 1}}},
                {'$match': {'count': {'$gt': 1}}}
            ]
            duplicates = list(db[collection_name].aggregate(pipeline))
            
            if duplicates:
                warnings.append(
                    f"Found {len(duplicates)} dates with duplicate records"
                )
            
            temporal_info = {
                'start_date': first_date.strftime('%Y-%m-%d') 
                              if hasattr(first_date, 'strftime') 
                              else str(first_date),
                'end_date': last_date.strftime('%Y-%m-%d') 
                            if hasattr(last_date, 'strftime') 
                            else str(last_date),
                'date_range_days': date_range,
                'total_records': total_records,
                'coverage_percentage': round((total_records / date_range) * 100, 2),
                'duplicate_dates': len(duplicates)
            }
            
            return {
                'valid': True,
                'warnings': warnings,
                'temporal_info': temporal_info
            }
        except Exception as e:
            logger.error(f"Error validating temporal continuity: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Temporal validation error: {str(e)}"]
            }
    def _validate_physical_relationships(
        self, 
        sample_docs: List[Dict]
    )-> List[str]:
        """Validate physical relationships between fields (TX ≥ TAVG ≥ TN, FF_X ≥ FF_AVG)"""
        
        warnings = []
        
        temp_violations = 0
        wind_violations = 0
        
        for doc in sample_docs:
            # Temperature: TX ≥ TAVG ≥ TN
            tx =  doc.get('TX')
            tn = doc.get('TN')
            tavg = doc.get('TAVG')
            
            if all(v is not None and not pd.isna(v) for v in [tx, tn, tavg]):
                if not (tn <= tavg <= tx):
                    temp_violations += 1
            
            # Wind speed: FF_X ≥ FF_AVG
            ff_x = doc.get('FF_X')
            ff_avg = doc.get('FF_AVG')
            
            if all(v is not None and not pd.isna(v) for v in [ff_x, ff_avg]):
                if ff_avg > ff_x:
                    wind_violations += 1
        
        if temp_violations > 0:
            warnings.append(
                f"Found {temp_violations} temperature relationship violations "
                f"(TX ≥ TAVG ≥ TN)"
            )
        
        if wind_violations > 0:
            warnings.append(
                f"Found {wind_violations} wind speed relationship violations "
                f"(FF_X ≥ FF_AVG)"
            )
        
        return warnings
class BmkgDataLoader:
    """Loads BMKG data from MongoDB into pandas DF"""
    
    def load_data(
        self,
        db,
        collection_name: str,
    )-> pd.DataFrame:
        """
        Load all data from MongoDB collection
        
        Returns:
            DataFrame with Date as index, sorted chronologically
        """
        
        try:
            # Load all records 
            cursor = db[collection_name].find({})
            df = pd.DataFrame(list(cursor))
            
            if len(df) == 0:
                raise BmkgPreprocessingError(
                    f"No data found in collection '{collection_name}'"
                )
            
            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)

            # Ensure Date is datetime
            if 'Date' in df.columns:
                if not pd.api.types.is_datetime64_dtype(df['Date']):
                    try:
                        df['Date'] = pd.to_datetime(df['Date'])
                    except Exception as e:
                        logger.warning(f"Failed to convert Date: {str(e)}")
            else:
                # Create Date from Year/Month/Day if needed
                if all(col in df.columns for col in ['Year', 'Month', 'Day']):
                    df['Date'] = pd.to_datetime(
                        df[['Year', 'Month', 'Day']]
                    )
                else:
                    raise BmkgPreprocessingError(
                        "No 'Date' column or Year/Month/Day columns found"
                    )

            # Sort by Date
            if 'Date' in df.columns:
                df = df.sort_values('Date')

            logger.info(f"Successfully loaded {len(df)} records from '{collection_name}'")
            return df
            
            
        except Exception as e:
            error_msg = f"Error loading data from {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise BmkgPreprocessingError(error_msg) from e
        
class BmkgDataSaver:
    """Saves preprocessed BMKG data back to MongoDB"""
    
    def save_preprocessed_data(
        self,
        db,
        preprocessed_data: pd.DataFrame,
        original_collection: str,
    ) -> Dict[str, Any]:
        """
        Save preprocessed data to new collection with '_cleaned' suffix
        Update dataset metadata
        
        Returns:
            Dictionary with save results and metadata info
        """
        try:
            # Generate cleaned collection name
            cleaned_collection = f"{original_collection}_cleaned"

            # Drop existing cleaned collection
            if cleaned_collection in db.list_collection_names():
                logger.info(f"Dropping existing collection: {cleaned_collection}")
                db[cleaned_collection].drop()

            # Prepare DataFrame for insertion
            df_to_save = preprocessed_data.copy()
            

            # KEEP Date as column (reset index)
            if df_to_save.index.name == 'Date' or 'Date' not in df_to_save.columns:
                df_to_save = df_to_save.reset_index()  # Convert Date index to column

            # Remove MongoDB internal fields
            if '_id' in df_to_save.columns:
                df_to_save = df_to_save.drop('_id', axis=1)

            if '__v' in df_to_save.columns:
                df_to_save = df_to_save.drop('__v', axis=1)
                logger.info("Dropped '__v' column")
            
            # Drop temporary preprocessing columns
            temp_columns = ['Season', 'is_RR_missing', 
                           'Month', 'day_of_year', 'max_daylight']
            columns_to_drop = [
                col for col in temp_columns 
                if col in df_to_save.columns
            ]
            
            if columns_to_drop:
                df_to_save = df_to_save.drop(columns_to_drop, axis=1)
                logger.info(f"Dropped temporary columns: {columns_to_drop}")

            # Convert to records for MongoDB
            records = df_to_save.to_dict('records')
            
            if records:
                db[cleaned_collection].insert_many(records)
                logger.info(
                    f"Inserted {len(records)} records into '{cleaned_collection}'"
                )

            # Update metadata
            meta_info = self._update_dataset_metadata(
                db,
                original_collection,
                cleaned_collection,
                len(records)
            )

            return {
                "preprocessedCollections": [cleaned_collection],
                "recordsInserted": {
                    "original": db[original_collection].count_documents({}),
                    "cleaned": len(records)
                },
                "metadata": meta_info
            }
            
        except Exception as e:
            error_msg = f"Error saving preprocessed data: {str(e)}"
            logger.error(error_msg)
            raise BmkgPreprocessingError(error_msg)
    def _update_dataset_metadata(
        self,
        db,
        original_collection: str,
        cleaned_collection: str,
        record_count: int,
    )->  Dict[str, Any]:
        """Update dataset-meta collection to point to cleaned data"""
        try:
               # Find metadata collection
            meta_collection = None
            if "dataset_meta" in db.list_collection_names():
                meta_collection = "dataset_meta"
            elif "DatasetMeta" in db.list_collection_names():
                meta_collection = "DatasetMeta"
            else:
                logger.warning("No metadata collection found!")
                return {"status": "no_meta_collection"}
            
            # Get original metadata
            original_meta = db[meta_collection].find_one(
                {"collectionName": original_collection}
            )
            
            if not original_meta:
                logger.warning(
                    f"No metadata found for '{original_collection}'"
                )
                return {"status": "no_original_metadata"}
            
            # Get cleaned collection columns
            sample_doc = db[cleaned_collection].find_one()
            if sample_doc:
                all_columns = list(sample_doc.keys())
                cleaned_columns = [
                    col for col in all_columns
                    if col not in ['_id', '__v'] and not col.startswith('_')
                ]
            else:
                cleaned_columns = []
            
            # Update metadata to point to cleaned collection
            update_fields = {
                "status": "preprocessed",
                "collectionName": cleaned_collection,
                "totalRecords": record_count,
                "columns": cleaned_columns,
                "lastUpdated": datetime.now(),
                "name": f"{original_meta.get('name', original_collection)} (Cleaned)"
            }
            
            result = db[meta_collection].update_one(
                {"collectionName": original_collection},
                {"$set": update_fields}
            )
            
            if result.modified_count > 0:
                logger.info(
                    f"Updated metadata to point to: {cleaned_collection}"
                )
            else:
                logger.warning("No metadata was updated")
            
            return {"status": "success"}
        
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {"status": "error", "error": str(e)}
        
class BmkgPreprocessor:
    """
    Main orchestrator class for preprocessing approach
    Implements smoothing-based imputation without ML
    Optimized for GCV validation and time-series modeling (Holt-Winters, LSTM)
    
    Key differences from preprocessing_bmkg.py:
    - NO machine learning models (RandomForest, KMeans)
    - Focus on smoothing methods (STL, cubic spline, moving average)
    - Seasonal median instead of mean
    - Adaptive imputation strategies
    """
    
    def __init__(
        self,
        db,
        collection_name: str,
    ):
        """
        Initialize preprocessor
        
        Args:
            db: MongoDB database instance
            collection_name: Name of the collection to preprocess
        """
        self.db = db
        self.collection_name = collection_name
        
        # Initialize helper classes
        self.validator = BmkgDataValidator()
        self.loader = BmkgDataLoader()
        self.saver = BmkgDataSaver()
        
        # Store original data for validation
        self.original_data = None
        
        # Configuration: Season definitions
        self.season_config = {
            "wet_months": [9, 10, 11, 12, 1, 2, 3],  # Sep-Mar
            "dry_months": [4, 5, 6, 7, 8],            # Apr-Aug
            "dry_season_peak": [5, 6, 7]              # Core dry months for RR=0
        }
        
        # Two-stage rain classification config
        self.rain_classification_config = {
            "probability_thresholds": {
                "wet_season_base": 0.65,      # 65% chance in wet season
                "dry_season_base": 0.15,      # 15% chance in dry season
                "dry_peak_base": 0.05,        # 5% in peak dry months
            },
            "markov_weights": {
                "rain_to_rain": 0.75,         # If yesterday rained → 75% today
                "dry_to_dry": 0.85,          # If yesterday dry → 85% dry today
                "rain_to_dry": 0.25,         # If yesterday rained → 25% dry today
                "dry_to_rain": 0.15,         # If yesterday dry → 15% rain today
            },
            "context_thresholds": {
                "rh_high": 85,               # RH_AVG > 85% suggests rain
                "rh_very_high": 95,          # RH_AVG > 95% strong rain signal
                "wind_calm": 2.5,            # FF_X < 2.5 m/s during rain
                "wind_moderate": 5.0,        # FF_X < 5.0 m/s light rain
            },
            "amount_categories": {
                "light": {"min": 0.1, "max": 5.0},      # Light rain
                "moderate": {"min": 5.0, "max": 20.0},   # Moderate rain
                "heavy": {"min": 20.0, "max": 100.0},    # Heavy rain
                "extreme": {"min": 100.0, "max": 300.0}  # Extreme rain
            }
        }
        
        # Configuration: Valid ranges for outlier detection
        self.valid_ranges = {
            'TX': (-5, 45),      # Max temperature (°C)
            'TN': (-10, 35),     # Min temperature (°C)
            'TAVG': (-5, 40),    # Avg temperature (°C)
            'RH_AVG': (0, 100),  # Relative humidity (%)
            'RR': (0, 500),      # Rainfall (mm)
            'FF_X': (0, 50),     # Max wind speed (m/s)
            'FF_AVG': (0, 30),   # Avg wind speed (m/s)
            'DDD_X': (0, 360),   # Wind direction (degrees)
            'SS': (0, 14)        # Sunshine duration (hours)
        } 
        
        # Configuration: Missing value codes
        self.missing_codes = [8888.0, 9999.0, 8888, 9999]
        
        # Configuration: Parameters to process
        self.params_to_process = [
            'RR', 'TX', 'TN', 'TAVG', 'RH_AVG', 
            'FF_X', 'FF_AVG', 'DDD_X', 'SS', 'DDD_CAR'
        ]
        
        # Track imputation confidence
        self.imputation_confidence = {}
        
        # Preprocessing report (NASA-style structure)
        self.preprocessing_report = {
            "missing_data": {},
            "outliers": {},
            "imputation": {},
            "physical_constraints": {},
            "smoothing_validation": {},
            "model_coverage": {},
            "quality_metrics": {},
            "warnings": []
        }
    
    def _get_imputation_method(self, param: str) -> str:
        """Get imputation method name for reporting"""
        method_map = {
            'RR': 'two_stage_binary_conditional',
            'TX': 'mathematical_relationships_cubic_spline',
            'TN': 'mathematical_relationships_cubic_spline',
            'TAVG': 'mathematical_relationships_cubic_spline',
            'RH_AVG': 'dewpoint_method_cubic_spline',
            'FF_X': 'linear_regression_cubic_spline',
            'FF_AVG': 'linear_regression_cubic_spline',
            'DDD_X': 'circular_mean',
            'SS': 'cubic_spline',
            'DDD_CAR': 'mode_based'
        }
        return method_map.get(param, 'unknown') 
 
    def preprocess(
        self,
        options: Dict[str, Any] = None
    )->  Dict[str, Any]:
        """
        Main preprocessing pipeline
        
        Args:
            options: Optional configuration overrides
            
        Returns:
            Dictionary with preprocessing results
        """
        
        try:
            start_time = datetime.now()
            logger.info("PROGRESS: Starting BMKG preprocessing pipeline...")
            self.options = self._get_default_options()
            if options: 
                self.options.update(options)
            
            # Step 1: Validate dataset
            logger.info("\n[1/6] Validating dataset...")
            validation_result = self.validator.validate_dataset(
                self.db, self.collection_name
            )
            
            if not validation_result.get('valid', False):
                errors = validation_result.get('errors', ['Unknown error'])
                raise BmkgPreprocessingError(f"Validation failed: {errors}")
            
            logger.info(f"Validation passed - {validation_result['total_records']} records")
            
            # Log warnings if any
            warnings = validation_result.get('warnings', [])
            if warnings:
                logger.warning(f"Validation warnings ({len(warnings)}):")
                for warning in warnings[:5]:  # Show first 5
                    logger.warning(f"  - {warning}")
                    
            # Step 2: Load data
            logger.info("\n[2/6] Loading data from MongoDB...")
            df = self.loader.load_data(self.db, self.collection_name)
            original_record_count = len(df)
            
            # Don't save original data here (has RangeIndex)
            # self.original_data = df.copy()  # ❌ Index is RangeIndex, not DatetimeIndex
            
            # NEW CODE:
            # self.original_data will be set in _apply_preprocessing() 
            # after _prepare_temporal_features() runs (DatetimeIndex)
            self.original_data = None  # Initialize as None
            
            logger.info(f"Loaded {original_record_count} records")
                
            # Step 3: Apply preprocessing
            logger.info("\n[3/6] Applying preprocessing pipeline...")
            processed_df = self._apply_preprocessing(df)
            logger.info("Preprocessing completed")
            
            # Step 4: Validate smoothing quality (GCV + Trend)
            logger.info("\n[4/6] Validating preprocessing quality...")
            self._validate_preprocessing_quality(processed_df)
            logger.info("Quality validation completed")
            
            # Step 5: Calculate model coverage
            logger.info("\n[5/6] Calculating model coverage (HW & LSTM)...")
            self._calculate_model_coverage(processed_df)
            logger.info("Coverage analysis completed")
            
            # Step 6: Save preprocessed data
            logger.info("\n[6/6] Saving preprocessed data...")
            save_result = self.saver.save_preprocessed_data(
                self.db,
                processed_df,
                self.collection_name
            )            
            cleaned_collection = save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None
            
            report_save_result = self._save_preprocessing_report(
                cleaned_collection or self.collection_name
            )
            
            # Generate final report 
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
             # Generate quality metrics
            self._generate_quality_metrics(df, processed_df)
            
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Records processed: {len(processed_df)}")
            logger.info(f"Cleaned collection: {cleaned_collection}")
            
            return {
                "status": "success",
                "message": "BMKG dataset preprocessed successfully",
                "collection": self.collection_name,
                "preprocessedData": processed_df.head(10).to_dict('records'),
                "recordCount": len(processed_df),
                "originalRecordCount": original_record_count,
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": cleaned_collection,
                "processingTime": round(processing_time, 2),
                "preprocessing_report": self._sanitize_for_mongodb(self.preprocessing_report),  # Sanitized report
                "validation_result": validation_result,
                "metadata": save_result.get("metadata", {}),
                "report_saved": report_save_result.get("status") == "success",
                "report_id": report_save_result.get("report_id")
            }
        except Exception as e:
            error_msg = f"Error during preprocessing: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise BmkgPreprocessingError(error_msg)
    
    def _get_default_options(self)-> Dict[str, Any]:
        """Get default preprocessing options with gap-aware wind parameter configs"""
        return {
            # Imputation settings
            "fill_missing": True,
            "detect_gaps": True,
            "max_gap_interpolate": 30,  # 30 days max for interpolation
            
            # Outlier detection settings
            "detect_outliers": True,
            "outlier_methods": ["iqr", "zscore"],
            "iqr_multiplier": 2.5,
            "zscore_threshold": 3.5,
            "outlier_treatment": "interpolate",
            
            # Smoothing settings (NO ML!)
            "apply_smoothing": False,  # focuses on imputation, not smoothing
            
            # Coverage analysis
            "calculate_coverage": True,
            
            # Parameters to process
            "columns_to_process": self.params_to_process,
            
            # NEW: Parameter-specific configurations
            "parameter_configs": {
                # FORECASTING CRITICAL - Full imputation effort
                "TAVG": {
                    "forecasting_critical": True,
                    "max_gap_interpolate": 7,
                    "reason": "Temperature is critical for forecasting"
                },
                "RR": {
                    "forecasting_critical": True,
                    "max_gap_interpolate": 7,
                    "reason": "Rainfall is critical for forecasting"
                },
                "RH_AVG": {
                    "forecasting_critical": True,
                    "max_gap_interpolate": 7,
                    "reason": "Humidity is critical for forecasting"
                },
                
                # SUPPORTING - Standard imputation
                "TX": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 14,
                    "reason": "Supporting parameter for forecasting"
                },
                "TN": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 14,
                    "reason": "Supporting parameter for forecasting"
                },
                "SS": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 14,
                    "reason": "Supporting parameter for forecasting"
                },
                
                # METADATA - Minimal/No forced imputation
                "FF_X": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 3,
                    "skip_long_gaps": True,
                    "reason": "Wind data - metadata only, don't force completion"
                },
                "FF_AVG": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 3,
                    "skip_long_gaps": True,
                    "reason": "Wind data - metadata only, don't force completion"
                },
                "DDD_X": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 3,
                    "skip_long_gaps": True,
                    "reason": "Wind direction - metadata only"
                },
                "DDD_CAR": {
                    "forecasting_critical": False,
                    "max_gap_interpolate": 3,
                    "skip_long_gaps": True,
                    "reason": "Wind direction cardinal - metadata only"
                }
            }
        }
        
    def _apply_preprocessing(
        self, 
        df: pd.DataFrame
    )-> pd.DataFrame:
        """
        Apply preprocessing pipeline with gap aware wind parameter handling
        
        Pipeline steps:
        1. Prepare temporal features
        2. Replace fill values
        3. Detect gaps
        4. Detect and handle outliers
        5. Impute missing values
        6. Apply physical constraints
        """
        logger.info("PROGRESS: Starting preprocessing steps...")
        logger.info("Starting preprocessing steps")
        processed_df = df.copy()
            
        # Step 1: Prepare temporal features
        logger.info("[1/7] Preparing temporal features...")
        processed_df = self._prepare_temporal_features(processed_df)
        
        # Step 2: Replace fill values
        logger.info("[2/7] Replacing fill values...")
        processed_df = self._replace_fill_values(processed_df)

        # Step 2.5: Detect and fix suspicious zeros in wind data (NEW)
        logger.info("[2.5/7] Detecting suspicious zeros...")
        processed_df = self._detect_and_fix_suspicious_zeros(processed_df)

        # Save original_data AFTER fill values replaced and suspicious zeros fixed
        cols_to_keep = [col for col in processed_df.columns 
                        if col not in ['Season', 'Month', 'is_RR_missing']]
        self.original_data = processed_df[cols_to_keep].copy()
        logger.info(f"Saved original data reference with DatetimeIndex (shape: {self.original_data.shape})")
        
        # Step 3: Detect gaps
        if self.options.get("detect_gaps", True):
            logger.info("[3/7] Detecting gaps...")
            self._detect_gaps(processed_df)
        
        # Step 4: Detect and handle outliers
        if self.options.get("detect_outliers", True):
            logger.info("[4/7] Detecting and handling outliers...")
            processed_df = self._handle_outliers(processed_df)
        
        # Step 5: Impute missing values with gap-aware wind handling (MODIFIED)
        if self.options.get("fill_missing", True):
            logger.info("[5/7] Imputing missing values with gap-aware logic")
            processed_df = self._impute_missing_values(processed_df)
        
        # Step 6: Apply physical constraints
        logger.info("[6/7] Applying physical constraints...")
        processed_df = self._apply_physical_constraints(processed_df)
        
        # Step 7 : Calculate and embed STL decomposition 
        logger.info("[7/7] Calculating and embedding STL decomposition...")
        self._calculate_and_embed_decomposition(processed_df)
        
        logger.info("preprocessing pipeline completed")
        return processed_df
    
    
    def _prepare_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare temporal features (Date index, Season, Month)"""
        
        # Ensure Date is datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
        elif df.index.name != 'Date':
            # Try to create Date from Year/Month/Day
            if all(col in df.columns for col in ['Year', 'Month', 'Day']):
                df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
                df = df.set_index('Date').sort_index()
        
        # Add Season column
        df['Season'] = df.index.month.map(
            lambda m: 'Wet' if m in self.season_config['wet_months'] else 'Dry'
        )
        
        # Add Month name
        df['Month'] = df.index.month_name()
        
        logger.info(f"    Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"    Temporal features added: Season, Month")
        
        return df
    
    def _replace_fill_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace BMKG fill values (8888, 9999) with NaN"""
        
        replaced_count = {}
        
        for param in self.params_to_process:
            if param not in df.columns:
                continue
            
            mask = df[param].isin(self.missing_codes)
            count = mask.sum()
            
            if count > 0:
                df.loc[mask, param] = np.nan
                replaced_count[param] = int(count)
        
        self.preprocessing_report["missing_data"]["fill_values_replaced"] = replaced_count
        
        # FF_AVG Zero Analysis
        if 'FF_AVG' in df.columns:
            ff_avg_zeros = (df['FF_AVG'] == 0.0).sum()
            ff_avg_missing = df['FF_AVG'].isna().sum()
            ff_avg_valid = ((df['FF_AVG'] > 0) & df['FF_AVG'].notna()).sum()
            total = len(df)
            
            # Suspicious zeros: FF_X > 0 but FF_AVG = 0
            if 'FF_X' in df.columns:
                suspicious_zeros = (
                    (df['FF_X'] > 0) & 
                    (df['FF_AVG'] == 0.0) & 
                    df['FF_AVG'].notna()
                ).sum()
            else:
                suspicious_zeros = 0
            
            logger.info("FF_AVG ZERO-INFLATION DIAGNOSTIC (BEFORE FIX)")
            logger.info(f"Total records: {total}")
            logger.info(f"FF_AVG = 0.0: {ff_avg_zeros} ({ff_avg_zeros/total*100:.1f}%)")
            logger.info(f"FF_AVG missing: {ff_avg_missing} ({ff_avg_missing/total*100:.1f}%)")
            logger.info(f"FF_AVG > 0 (valid): {ff_avg_valid} ({ff_avg_valid/total*100:.1f}%)")
            logger.info(f"Suspicious zeros (FF_X>0 but FF_AVG=0): {suspicious_zeros} ({suspicious_zeros/total*100:.1f}%)")
        
        if replaced_count:
            logger.info(f"Replaced fill values: {replaced_count}")
        else:
            logger.info("No fill values found")
        
        # PHASE 1 FIX: FF_AVG SUSPICIOUS ZERO REPLACEMENT
        # Replace suspicious zeros with NaN BEFORE outlier detection
        # This ensures two-tier imputation handles them correctly
        # instead of treating them as valid calm wind conditions
        
        if 'FF_AVG' in df.columns and 'FF_X' in df.columns:
            
            # Identify suspicious zeros
            suspicious_mask = (
                (df['FF_AVG'] == 0.0) &      # FF_AVG is zero
                (df['FF_X'] > 0) &            # But FF_X is positive
                df['FF_AVG'].notna()          # And FF_AVG is not NaN
            )
            
            suspicious_count = suspicious_mask.sum()
            
            if suspicious_count > 0:
                # Store original values for reporting
                suspicious_dates = df.index[suspicious_mask]
                sample_dates = suspicious_dates[:5]  # First 5 for logging
                
                logger.info(f"Detected {suspicious_count} suspicious FF_AVG zeros")
                
                logger.info("Sample suspicious zeros:")
                logger.info(f"{'Date':<12} {'FF_X':>8} {'FF_AVG (orig)':>15} {'Action':<20}")
                
                for idx in sample_dates:
                    ff_x_val = df.loc[idx, 'FF_X']
                    ff_avg_val = df.loc[idx, 'FF_AVG']
                    logger.info(
                        f"{idx.strftime('%Y-%m-%d'):<12} "
                        f"{ff_x_val:>8.1f} "
                        f"{ff_avg_val:>15.1f} "
                        f"{'→ Replace with NaN':<20}"
                    )
                
                if suspicious_count > 5:
                    logger.info(f"... and {suspicious_count - 5} more")
                
                # Replace suspicious zeros with NaN
                df.loc[suspicious_mask, 'FF_AVG'] = np.nan
                
                # Update statistics
                ff_avg_zeros_after = (df['FF_AVG'] == 0.0).sum()
                ff_avg_missing_after = df['FF_AVG'].isna().sum()
                
                logger.info("Fix applied successfully:")
                logger.info(f"  Suspicious zeros replaced: {suspicious_count}")
                logger.info(f"  FF_AVG = 0.0 remaining: {ff_avg_zeros_after} (legitimate calm wind)")
                logger.info(f"  FF_AVG missing (total): {ff_avg_missing_after} (will be imputed)")
                
                
                # Store in preprocessing report
                if "suspicious_zeros" not in self.preprocessing_report:
                    self.preprocessing_report["suspicious_zeros"] = {}
                
                self.preprocessing_report["suspicious_zeros"]["FF_AVG"] = {
                    "count": int(suspicious_count),
                    "percentage": round((suspicious_count / total) * 100, 2),
                    "action": "replaced_with_nan",
                    "reason": "physical_impossibility_FF_X_greater_than_zero"
                }
                                
            else:
                logger.info("No suspicious FF_AVG zeros detected (all zeros are legitimate)")
        return df
    
    def _detect_and_fix_suspicious_zeros(
        self,
        df: pd.DataFrame
    )-> pd.DataFrame:
        """ Detect and fix suspicious FF_AVG=0 where FF_X>0"""
        suspicious_count = 0
        
        if 'FF_AVG' in df.columns and 'FF_X' in df.columns:
            # Find suspicious cases: FF_X > 0 but FF_AVG = 0
            suspicious_mask = (df['FF_X'] > 0) & (df['FF_AVG'] == 0.0)
            suspicious_count = suspicious_mask.sum()
            
            if suspicious_count > 0:
                logger.info(f"Detected {suspicious_count} suspicious FF_AVG zeros ({suspicious_count/len(df)*100:.1f}%)")
                
                # Replace suspicious zeros with NaN
                df.loc[suspicious_mask, 'FF_AVG'] = np.nan
                
                # Store in report
                self.preprocessing_report["suspicious_zeros"] = {
                    "ff_avg_zeros_detected": int(suspicious_count),
                    "percentage": round((suspicious_count/len(df))*100, 1),
                    "action": "replaced_with_nan"
                }
        return df
        
    
    def _detect_gaps(self, df: pd.DataFrame) -> None:
        """Detect and classify gaps in time series uses same logic as preprocessing_bmkg.py"""
        
        dates = df.index
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
        missing_dates = date_range.difference(dates)
        
        if len(missing_dates) == 0:
            logger.info("    No date gaps found")
            self.preprocessing_report["gaps"] = {
                "total_gaps": 0,
                "gap_details": []
            }
            return
        
        # Group consecutive missing dates
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
                "start_date": str(gap[0].date()),
                "end_date": str(gap[-1].date()),
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
                    warning = (f"Large gap detected: {gap_info['start_date']} to "
                              f"{gap_info['end_date']} ({duration} days)")
                    self.preprocessing_report["warnings"].append(warning)
            
            gap_details.append(gap_info)
        
        self.preprocessing_report["gaps"] = {
            "total_gaps": len(gaps),
            "small_gaps": small_gaps,
            "medium_gaps": medium_gaps,
            "large_gaps": large_gaps,
            "gap_details": gap_details[:10]  # Store first 10
        }
        
        logger.info(f"    Detected {len(gaps)} gaps: {small_gaps} small, "
                   f"{medium_gaps} medium, {large_gaps} large")
    def _handle_outliers(
        self,
        df: pd.DataFrame,
    )-> pd.DataFrame:
        """
        Detect and handle outliers using IQR and Z-score methods
        
        Strategy:
        - IQR method: Q1 - k*IQR, Q3 + k*IQR (k=2.0 default)
        - Z-score method: |z| > threshold (3.5 default)
        - Treatment: Set to NaN for later imputation
        """
        
        outlier_stats = {}
        methods = self.options.get("outlier_methods", ["iqr", "zscore"])
        
        for param in self.params_to_process:
            if param not in df.columns:
                continue
        
            # Skip non-numeric parameters
            if param == 'DDD_CAR':
                continue
            
            # Get valid range for this parameter
            if param not in self.valid_ranges:
                continue
            
            min_val, max_val = self.valid_ranges[param]
            outliers_detected = 0
            outlier_mask = pd.Series(False, index=df.index)
            
            # Suspicious zero detection: FF_AVG=0 tapi FF_X>0
            if param == 'FF_AVG' and 'FF_X' in df.columns:
                suspicious_zero_mask = (
                    (df['FF_AVG'] == 0.0) &
                    (df['FF_X'] > 0) &
                    df['FF_AVG'].notna()
                )
                outlier_mask |= suspicious_zero_mask
                logger.info(f"    Detected {suspicious_zero_mask.sum()} suspicious FF_AVG zeros (FF_X>0 but FF_AVG=0)")
            
            # Method 1: IQR
            if "iqr" in methods:
                Q1 = df[param].quantile(0.25)
                Q3 = df[param].quantile(0.75)
                IQR = Q3 - Q1
                
                k = self.options.get("iqr_multiplier", 2.5)
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR
                
                iqr_outliers = (df[param] < lower_bound) | (df[param] > upper_bound)
                outlier_mask |= iqr_outliers
            
            # Method 2: Z-score
            if "zscore" in methods:
                threshold = self.options.get("zscore_threshold", 3.5)
                mean = df[param].mean()
                std = df[param].std()
                
                if std > 0:
                    z_scores = np.abs((df[param] - mean) / std)
                    zscore_outliers = z_scores > threshold
                    outlier_mask |= zscore_outliers
            
            # Method 3: Physical range (always applied)
            range_outliers = (df[param] < min_val) | (df[param] > max_val)
            outlier_mask |= range_outliers
            
            # Count outliers 
            outliers_detected = outlier_mask.sum()
            
            if outliers_detected > 0:
                # Apply treatment
                treatment = self.options.get("outlier_treatment", "interpolate")
                
                if treatment == "interpolate":
                    df.loc[outlier_mask, param] = np.nan
                    
                elif treatment == "cap":
                    df.loc[df[param] < min_val, param] = min_val
                    df.loc[df[param] > max_val, param] = max_val
                    
                outlier_stats[param] = {
                    "count": int(outliers_detected),
                    "percentage": round((outliers_detected / len(df)) * 100, 2),
                    "treatment": treatment
                }
        
        self.preprocessing_report["outliers"] = outlier_stats
            
        if outlier_stats:
            total_outliers = sum(v["count"] for v in outlier_stats.values())
            logger.info(f"    Detected and handled {total_outliers} outliers across {len(outlier_stats)} parameters")
        else:
            logger.info("    No outliers detected")
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using smoothing-based methods (NO ML)
        Uses gap-aware logic for wind parameters only
        """
        
        imputation_stats = {}
        
        # Track missing values before imputation
        missing_before = {}
        for param in self.params_to_process:
            if param in df.columns:
                missing_before[param] = int(df[param].isna().sum())
        
        # 1. RAINFALL (RR) - Most complex
        if 'RR' in df.columns:
            logger.info("Imputing RR (Rainfall)...")
            df = self._impute_rainfall(df)
            
        # 2. TEMPERATURE (TX, TN, TAVG)
        temp_params = ['TX', 'TN', 'TAVG']
        if any(p in df.columns for p in temp_params):
            logger.info("Imputing TX/TN/TAVG (Temperature)...")
            df = self._impute_temperature(df)
        
        # 3. HUMIDITY (RH_AVG)
        if 'RH_AVG' in df.columns:
            logger.info("Imputing RH_AVG (Humidity)...")
            df = self._impute_humidity(df)
        
        # 4. WIND SPEED (FF_X, FF_AVG) - GAP-AWARE
        if 'FF_X' in df.columns or 'FF_AVG' in df.columns:
            logger.info("Imputing FF_X/FF_AVG (Wind Speed - Gap Aware)...")
            df = self._impute_wind_speed_gap_aware(df)
        
        # 5. WIND DIRECTION (DDD_X)
        if 'DDD_X' in df.columns:
            logger.info("Imputing DDD_X (Wind Direction)...")
            df = self._impute_wind_direction(df)
        
        # 6. SUNSHINE DURATION (SS)
        if 'SS' in df.columns:
            logger.info("Imputing SS (Sunshine)...")
            df = self._impute_sunshine(df)
        
        # 7. CARDINAL DIRECTION (DDD_CAR)
        if 'DDD_CAR' in df.columns:
            logger.info("Imputing DDD_CAR (Cardinal Direction)...")
            df = self._impute_cardinal_direction(df)
            
        # Track missing values after imputation
        missing_after = {}
        for param in self.params_to_process:
            if param in df.columns:
                missing_after[param] = int(df[param].isna().sum())
                
                if param in missing_before:
                    imputed_count = missing_before[param] - missing_after[param]
                    if imputed_count > 0:
                        imputation_stats[param] = {
                            "before": missing_before[param],
                            "after": missing_after[param],
                            "imputed": imputed_count,
                            "success_rate": round((imputed_count / missing_before[param]) * 100, 2) 
                                        if missing_before[param] > 0 else 0,
                            "method": self._get_imputation_method(param)
                        }
        
        self.preprocessing_report["imputation"] = imputation_stats
        
        total_imputed = sum(v["imputed"] for v in imputation_stats.values())
        logger.info(f"    Successfully imputed {total_imputed} values across {len(imputation_stats)} parameters")
        
        return df
    
    def _impute_rainfall(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-stage rainfall imputation approach
        
        Stage 1: Binary rain occurrence classification
        Stage 2: Conditional amount imputation for rain days
        
        Expected Results:
        - GCV improvement: 24M → ~750k (96.9% reduction)
        - Trend preservation: 71.8% → ~87%
        - Agricultural logic: Rain/no-rain patterns preserved
        """
        logger.info(f"Starting Two-Stage Rainfall Imputation...")
        
        # Track original missing pattern
        df['is_RR_missing'] = df['RR'].isna().astype(int)
        original_missing_count = df['is_RR_missing'].sum()
        
        if original_missing_count == 0:
            logger.info("No missing rainfall values to impute")
            return df
        
        logger.info(f"Processing {original_missing_count} missing rainfall values")
        
        # Keep dry peak logic (existing logic that works)
        dry_peak_mask = (
            (df['Season'] == 'Dry') & 
            df['RR'].isna() & 
            df.index.month.isin(self.season_config['dry_season_peak'])
        )
        dry_peak_count = 0
        if dry_peak_mask.sum() > 0:
            df.loc[dry_peak_mask, 'RR'] = 0
            dry_peak_count = dry_peak_mask.sum()
            logger.info(f"Set {dry_peak_count} peak dry season values to 0")
            
        # Stage 1: Binary rain occurrence classification
        remaining_missing = df['RR'].isna().sum()
        if remaining_missing > 0:
            logger.info(f"Stage 1: Classifying rain occurrence for {remaining_missing} values")
            df = self._classify_rain_occurrence(df)
            
            # Stage 2: Conditional amount imputation
            rain_days_to_impute = (df['RR'].isna() == False) & (df['is_RR_missing'] == 1) & (df['RR'] > 0)
            rain_days_count = rain_days_to_impute.sum()
            
            if rain_days_count > 0:
                logger.info(f"Stage 2: Imputing amounts for {rain_days_count} classified rain days")
                df = self._impute_rain_amounts(df)
                
        
        # Final validation and cleanup
        df = self._finalize_rainfall_imputation(df)
        final_missing = df['RR'].isna().sum()
        imputed_count = original_missing_count - final_missing
        
        logger.info(f"  Original missing values: {original_missing_count}")
        logger.info(f"  Dry peak set to 0: {dry_peak_count}")
        logger.info(f"  Two-stage imputed: {imputed_count - dry_peak_count}")
        logger.info(f"  Total resolved: {imputed_count}")
        logger.info(f"  Still missing: {final_missing}")
        logger.info(f"  Success rate: {((imputed_count / original_missing_count) * 100) if original_missing_count > 0 else 0:.1f}%")
        return df
    
    def _classify_rain_occurrence(
        self,
        df: pd.DataFrame
    )-> pd.DataFrame:
        """
        Stage 1: Classify rain/no-rain for missing values
        
        Agricultural Logic:
        1. Seasonal probability (Indonesian wet/dry patterns)
        2. Markov chain (rain event clustering)
        3. Context awareness (meteorological indicators)
        
        Decision Framework:
        - Combine all 3 methods with weighted scoring
        - Threshold: >0.5 = rain day, ≤0.5 = no rain
        """
        
        missing_mask = df['RR'].isna()
        classified_rain = 0
        classified_dry = 0
        confidence_scores = []
        
        for idx in df.index[missing_mask]:
            # Method 1: Seasonal probability
            month = idx.month
            season = 'Wet' if month in self.season_config['wet_months'] else 'Dry'
            
            if month in self.season_config['dry_season_peak']:
                base_prob = self.rain_classification_config['probability_thresholds']['dry_peak_base']
            elif season == 'Wet':
                base_prob = self.rain_classification_config['probability_thresholds']['wet_season_base']
            else:
                base_prob = self.rain_classification_config['probability_thresholds']['dry_season_base']
                
            # Method 2: Markov chain adjustment
            markov_prob = self._calculate_markov_probability(df, idx)
            
            # Method 3: Context awareness (meteorological)
            context_prob = self._calculate_context_probability(df, idx)
            
            # Weighted combination
            # Seasonal: 40%, Markov: 35%, Context: 25%
            final_probability = (
                0.40 * base_prob +
                0.35 * markov_prob +
                0.25 * context_prob
            )
            
            # Confidence = how certain we are about the classification
            # High probability (close to 0 or 1) = high confidence
            # Probability around 0.5 = low confidence
            confidence = 1 - (2 * abs(final_probability - 0.5))  # 0.5 = maximum uncertainty
            confidence_scores.append(confidence)
                
            # DECISION: Rain or No Rain
            if final_probability > 0.5:
                # Classify as rain day - will get amount in Stage 2
                df.loc[idx, 'rain_classified'] = 1
                classified_rain += 1
            else:
                # Classify as no-rain day
                df.loc[idx, 'RR'] = 0
                df.loc[idx, 'rain_classified'] = 0
                classified_dry += 1
                
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            self.imputation_confidence['RR'] = avg_confidence
            
            logger.info(f"RR classification confidence: {avg_confidence:.3f} (1.0=certain, 0.0=uncertain)")
        
        logger.info(f"Binary classification results:")
        logger.info(f"Classified as rain days: {classified_rain}")
        logger.info(f"Classified as dry days: {classified_dry}")
        logger.info(f"Rain day ratio: {(classified_rain / (classified_rain + classified_dry) * 100) if (classified_rain + classified_dry) > 0 else 0:.1f}%")
        
        return df
    
    def _calculate_markov_probability(
        self, 
        df: pd.DataFrame,
        current_idx
        )-> float:
        """
        Calculate rain probability based on yesterday's state (Markov chain)
        
        Indonesian Rainfall Patterns:
        - Rain events cluster (monsoon bursts)
        - Dry spells persist (trade wind dominance)  
        - Transition probabilities vary by season
        """
        
        try:
            # Get yesterday date
            yesterday_idx = current_idx - pd.Timedelta(days=1)
            
            if yesterday_idx not in df.index:
                return 0.5
            
            yesterday_rr = df.loc[yesterday_idx, 'RR']
            
            # If yesterday data is missing, look 2-3 days back
            if pd.isna(yesterday_rr):
                for days_back in [2,3]:
                    prev_idx = current_idx - pd.Timedelta(days=days_back)
                    if prev_idx in df.index:
                        prev_rr = df.loc[prev_idx, 'RR']
                        if not pd.isna(prev_rr):
                            yesterday_rr = prev_rr
                            break
                        
            # Still no data, return netral
            if pd.isna(yesterday_rr):
                return 0.5
            
            # Apply Markov transitions
            if yesterday_rr > 0:
                # Yesterday rained → higher chance today
                return self.rain_classification_config['markov_weights']['rain_to_rain']
            else:
                # Yesterday dry → lower chance today  
                return self.rain_classification_config['markov_weights']['dry_to_rain']
        except Exception as e:
            logger.debug(f"Markov calculation error for {current_idx}: {str(e)}")
            return 0.5
        
    
    def _calculate_context_probability(
        self,
        df: pd.DataFrame, 
        current_idx
    )-> float:
        """
        Calculate rain probability from meteorological context
        
        Indonesian Meteorological Indicators:
        - High humidity (>85%) = rain likely
        - Very high humidity (>95%) = rain very likely  
        - Calm winds (<2.5 m/s) during high humidity = rain
        - Temperature-humidity combinations
        """
        
        try:
            # Get meteorological values for current day
            rh_avg = df.loc[current_idx, 'RH_AVG'] if 'RH_AVG' in df.columns else None
            ff_x = df.loc[current_idx, 'FF_X'] if 'FF_X' in df.columns else None
            tavg = df.loc[current_idx, 'TAVG'] if 'TAVG' in df.columns else None
            
            context_score = 0.0
            factors_used = 0
            
            # Factor 1: Humidity analysis
            if pd.notna(rh_avg):
                factors_used += 1
                if rh_avg >= self.rain_classification_config['context_thresholds']['rh_very_high']:
                    context_score += 0.9  # Very high humidity
                elif rh_avg >= self.rain_classification_config['context_thresholds']['rh_high']:
                    context_score += 0.7  # High humidity
                elif rh_avg >= 70:
                    context_score += 0.4  # Moderate humidity
                else:
                    context_score += 0.1  # Low humidity
            
            # Factor 2: Wind-humidity combination  
            if pd.notna(rh_avg) and pd.notna(ff_x):
                factors_used += 1
                if (rh_avg >= 85 and 
                    ff_x <= self.rain_classification_config['context_thresholds']['wind_calm']):
                    context_score += 0.8  # Calm + humid = rain likely
                elif (rh_avg >= 75 and 
                      ff_x <= self.rain_classification_config['context_thresholds']['wind_moderate']):
                    context_score += 0.5  # Moderate conditions
                else:
                    context_score += 0.2  # Other combinations
            
            # Factor 3: Multi-day humidity trend
            if pd.notna(rh_avg):
                factors_used += 1
                try:
                    # Look at 3-day humidity trend
                    humidity_window = []
                    for days_back in range(1, 4):
                        prev_idx = current_idx - pd.Timedelta(days=days_back)
                        if prev_idx in df.index and 'RH_AVG' in df.columns:
                            prev_rh = df.loc[prev_idx, 'RH_AVG']
                            if pd.notna(prev_rh):
                                humidity_window.append(prev_rh)
                    
                    if len(humidity_window) >= 2:
                        avg_prev_humidity = np.mean(humidity_window)
                        if rh_avg > avg_prev_humidity + 10:  # Rising humidity
                            context_score += 0.6
                        elif rh_avg > avg_prev_humidity:
                            context_score += 0.4
                        else:
                            context_score += 0.2
                    else:
                        context_score += 0.3  # Neutral when no trend available
                        
                except Exception:
                    context_score += 0.3  # Neutral on error
            
            # Calculate final context probability
            if factors_used > 0:
                final_prob = context_score / factors_used
                return max(0.0, min(1.0, final_prob))  # Clip to [0,1]
            else:
                return 0.5  # Neutral when no context available
            
        except Exception as e:
            logger.debug(f"Context calculation error for {current_idx}: {str(e)}")
            return 0.5
        
    def _impute_rain_amounts(
        self,
        df: pd.DataFrame
        )-> pd.DataFrame:
        """
        Stage 2: Impute amounts for classified rain days
        
        Methods:
        1. Neighbor context (light/moderate/heavy based on surrounding days)
        2. Seasonal medians (month-specific realistic amounts)
        3. Physical constraints (no negatives, reasonable maxima)
        """
        
        # Find rain days that need amounts
        rain_days_mask = (
            (df['is_RR_missing'] == 1) & 
            (df.get('rain_classified', 0) == 1) &
            df['RR'].isna()
        )
        
        amount_imputed = 0
        
        for idx in df.index[rain_days_mask]:
            # Method 1: Neighbor context analysis
            neighbor_context = self._analyze_neighbor_context(df, idx)
            
            # Method 2: Seasonal median baseline
            seasonal_baseline = self._get_seasonal_rain_baseline(df, idx)
            
            # Method 3: Meteorological intensity adjustment
            intensity_factor = self._calculate_intensity_factor(df, idx)
            
            # COMBINE MethodS TO GET AMOUNT
            base_amount = seasonal_baseline
            context_adjusted = base_amount * neighbor_context['multiplier']
            final_amount = context_adjusted * intensity_factor
            
            # Apply physical constraints
            final_amount = self._apply_rain_amount_constraints(
                final_amount, df, idx, neighbor_context
            )
            
            # Assign the amount
            df.loc[idx, 'RR'] = final_amount
            amount_imputed += 1
        
        logger.info(f"Amount imputation results:")
        logger.info(f"Rain days processed: {amount_imputed}")
        
        return df
    
    def _analyze_neighbor_context(
        self,
        df: pd.DataFrame,
        current_idx
    )-> Dict[str, Any]:
        """
        Analyze surrounding days to determine rain intensity context
        
        Context Categories:
        - Isolated: Single rain day (light amount)
        - Cluster: Multi-day rain event (moderate amounts)
        - Peak: Center of rain event (heavy amount)
        - Tail: End of rain event (light-moderate)
        """
        try:
            # Look at ±3 days window
            window_rr = []
            window_indices = []
            
            for offset in range(-3, 4):
                if offset == 0:
                    continue
                    
                check_idx = current_idx + pd.Timedelta(days=offset)
                if check_idx in df.index:
                    rr_val = df.loc[check_idx, 'RR']
                    if pd.notna(rr_val):
                        window_rr.append(rr_val)
                        window_indices.append(offset)
            
            if len(window_rr) == 0:
                return {
                    'context_type': 'isolated',
                    'multiplier': 0.8,  # Conservative for isolated
                    'neighbor_count': 0,
                    'max_neighbor': 0
                }
            
            # Analyze neighbor patterns
            rain_neighbors = [rr for rr in window_rr if rr > 0]
            neighbor_count = len(rain_neighbors)
            max_neighbor = max(window_rr) if window_rr else 0
            avg_neighbor = np.mean([rr for rr in window_rr if rr > 0]) if rain_neighbors else 0
            
            # Determine context type and multiplier
            if neighbor_count == 0:
                context_type = 'isolated'
                multiplier = 0.8
            elif neighbor_count <= 2:
                context_type = 'cluster_light'
                multiplier = 1.0
            elif neighbor_count <= 4:
                context_type = 'cluster_moderate' 
                multiplier = 1.2
            else:
                context_type = 'cluster_heavy'
                multiplier = 1.4
                
            # Intensity-based adjustment
            if max_neighbor > 50:  # Heavy rain nearby
                multiplier *= 1.3
            elif max_neighbor > 20:  # Moderate rain nearby
                multiplier *= 1.1
            elif max_neighbor < 5:  # Light rain nearby
                multiplier *= 0.9
            
            return {
                'context_type': context_type,
                'multiplier': multiplier,
                'neighbor_count': neighbor_count,
                'max_neighbor': max_neighbor,
                'avg_neighbor': avg_neighbor
            }
            
        except Exception as e:
            logger.debug(f"Neighbor context error for {current_idx}: {str(e)}")
            return {
                'context_type': 'unknown',
                'multiplier': 1.0,
                'neighbor_count': 0,
                'max_neighbor': 0
            }
    
    def _get_seasonal_rain_baseline(
        self,
        df: pd.DataFrame,
        current_idx
    )-> float:
        """
        Get seasonal median rainfall amount as baseline
        Seasonal Patterns:
        - Wet season: Higher baseline amounts
        - Dry season: Lower baseline amounts  
        - Monthly variation: Peak wet vs transition months
        """
        
        try:
            month = current_idx.month
            season = 'Wet' if month in self.season_config['wet_months'] else 'Dry'
            
            # Get monthly data for same month
            monthly_data = df[
                (df.index.month == month) & 
                (df['RR'] > 0) & 
                (df['RR'].notna())
            ]['RR']
            
            if len(monthly_data) >= 3:
                # Use monthly median (robust to outliers)
                baseline = monthly_data.median()
            else:
                # Fallback to seasonal data
                seasonal_months = (self.season_config['wet_months'] 
                                 if season == 'Wet' 
                                 else self.season_config['dry_months'])
                
                seasonal_data = df[
                    (df.index.month.isin(seasonal_months)) &
                    (df['RR'] > 0) & 
                    (df['RR'].notna())
                ]['RR']
                
                if len(seasonal_data) >= 5:
                    baseline = seasonal_data.median()
                else:
                    # Ultimate fallback: overall median
                    overall_data = df[(df['RR'] > 0) & (df['RR'].notna())]['RR']
                    baseline = overall_data.median() if len(overall_data) > 0 else 5.0
            
            # Apply seasonal multipliers
            if month in [11, 12, 1, 2]:  # Peak wet months
                baseline *= 1.2
            elif month in [5, 6, 7]:     # Peak dry months  
                baseline *= 0.7
            elif month in [3, 4, 9, 10]: # Transition months
                baseline *= 0.9
            
            return max(0.5, baseline)  # Minimum 0.5mm for rain days
        except Exception as e:
            logger.debug(f"Seasonal baseline error for {current_idx}: {str(e)}")
            return 8.0  # Default moderate amount
        
    def _calculate_intensity_factor(
        self,
        df: pd.DataFrame,
        current_idx
    )-> float:
        """
        Calculate intensity multiplier based on meteorological conditions
        
        High intensity indicators:
        - Very high humidity (>95%)
        - Low wind + high humidity
        - Temperature-humidity instability
        """
        try:
            rh_avg = df.loc[current_idx, 'RH_AVG'] if 'RH_AVG' in df.columns else None
            ff_x = df.loc[current_idx, 'FF_X'] if 'FF_X' in df.columns else None
            
            intensity_factor = 1.0
            
            # Humidity intensity
            if pd.notna(rh_avg):
                if rh_avg >= 95:
                    intensity_factor *= 1.4  # Very high humidity
                elif rh_avg >= 90:
                    intensity_factor *= 1.2  # High humidity  
                elif rh_avg >= 80:
                    intensity_factor *= 1.0  # Moderate humidity
                else:
                    intensity_factor *= 0.8  # Lower humidity
            
            # Wind-humidity combination
            if pd.notna(rh_avg) and pd.notna(ff_x):
                if rh_avg >= 90 and ff_x <= 2:
                    intensity_factor *= 1.3  # Calm + very humid = intense
                elif rh_avg >= 85 and ff_x <= 3:
                    intensity_factor *= 1.1  # Moderate intensity
            
            # Keep within reasonable bounds
            return max(0.6, min(2.0, intensity_factor))
        except Exception as e:
            logger.debug(f"Intensity factor error for {current_idx}: {str(e)}")
            return 1.0
        
    def _apply_rain_amount_constraints(
        self,
        amount: float,
        df: pd.DataFrame,
        current_idx,
        neighbor_context: Dict[str, Any]
    ) -> float:
        """
        Apply physical and agricultural constraints to rain amounts
        
        Constraints:
        - Minimum: 0.1mm (trace amounts don't count as rain)
        - Maximum: Based on season and neighbors
        - Agricultural logic: Realistic daily amounts for farming
        """
        
        try:
            month = current_idx.month
            season = 'Wet' if month in self.season_config['wet_months'] else 'Dry'
            
            # Minimum constraint
            if amount < 0.1:
                amount = 0.1
            
            # Maximum constraint based on season
            if season == 'Wet':
                if month in [12, 1, 2]:  # Peak wet season
                    max_daily = 120.0
                else:
                    max_daily = 80.0
            else:  # Dry season
                max_daily = 40.0
            
            # Neighbor-based maximum adjustment
            max_neighbor = neighbor_context.get('max_neighbor', 0)
            if max_neighbor > 0:
                # Don't exceed 2x the maximum neighbor
                neighbor_max = max_neighbor * 2.0
                max_daily = min(max_daily, neighbor_max)
            
            # Apply maximum constraint
            amount = min(amount, max_daily)
            
            # Agricultural categories (ensure realistic amounts)
            categories = self.rain_classification_config['amount_categories']
            
            if amount <= categories['light']['max']:
                # Keep light amounts light
                amount = min(amount, categories['light']['max'])
            elif amount <= categories['moderate']['max']:
                # Moderate amounts are good for agriculture
                pass  # No additional constraint
            else:
                # Heavy amounts: cap based on neighbor context
                if neighbor_context.get('neighbor_count', 0) < 2:
                    # Isolated heavy rain: reduce to moderate
                    amount = min(amount, categories['moderate']['max'])
            
            return round(amount, 1)  # Round to 0.1mm precision
        except Exception as e:
            logger.debug(f"Constraint application error for {current_idx}: {str(e)}")
            return min(50.0, amount)
    
    def _finalize_rainfall_imputation(
        self,
        df: pd.DataFrame
    )-> pd.DataFrame:
        """
        Final cleanup and validation of rainfall imputation
        
        Cleanup:
        - Remove temporary columns
        - Apply final constraints
        - Validate results
        - Handle any remaining missing values
        """
        try:
            # Remove temporary columns
            temp_columns = ['rain_classified']
            for col in temp_columns:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            
            # Final constraint: ensure non-negative
            df['RR'] = df['RR'].clip(lower=0)
            
            # Handle any remaining missing values (shouldn't happen, but safety)
            if df['RR'].isna().any():
                remaining_missing = df['RR'].isna().sum()
                logger.warning(f"{remaining_missing} values still missing after two-stage imputation")
                
                # Emergency fallback: seasonal median
                for month in range(1, 13):
                    month_mask = (df.index.month == month) & df['RR'].isna()
                    if month_mask.sum() > 0:
                        monthly_median = df[
                            (df.index.month == month) & 
                            (df['RR'].notna())
                        ]['RR'].median()
                        
                        if pd.notna(monthly_median) and monthly_median > 0:
                            df.loc[month_mask, 'RR'] = monthly_median
                        else:
                            df.loc[month_mask, 'RR'] = 0  # Fallback to 0
            
            # Validate results
            negative_count = (df['RR'] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Fixed {negative_count} negative rainfall values")
                df['RR'] = df['RR'].clip(lower=0)
            
            extreme_count = (df['RR'] > 200).sum()
            if extreme_count > 0:
                logger.info(f"Found {extreme_count} extreme rainfall values (>200mm) - kept as valid")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in rainfall imputation finalization: {str(e)}")
            # Emergency: ensure no missing values
            if df['RR'].isna().any():
                df['RR'] = df['RR'].fillna(0)
            df['RR'] = df['RR'].clip(lower=0)
            return df
        
    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute temperature using mathematical relationships
        
        Relationships:
        - TAVG = (TX + TN) / 2
        - TX ≥ TAVG ≥ TN (constraints applied later)
        """
        
        # Calculate TAVG from TX and TN where possible
        if 'TAVG' in df.columns and 'TX' in df.columns and 'TN' in df.columns:
            calc_mask = df['TAVG'].isna() & df['TX'].notna() & df['TN'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TAVG'] = (df.loc[calc_mask, 'TX'] + df.loc[calc_mask, 'TN']) / 2
                logger.info(f"Calculated {calc_mask.sum()} TAVG from TX/TN")
        
        # Calculate TX from TAVG and TN
        if 'TX' in df.columns and 'TAVG' in df.columns and 'TN' in df.columns:
            calc_mask = df['TX'].isna() & df['TAVG'].notna() & df['TN'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TX'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TN']
                logger.info(f"Calculated {calc_mask.sum()} TX from TAVG/TN")
        
        # Calculate TN from TAVG and TX
        if 'TN' in df.columns and 'TAVG' in df.columns and 'TX' in df.columns:
            calc_mask = df['TN'].isna() & df['TAVG'].notna() & df['TX'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TN'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TX']
                logger.info(f"Calculated {calc_mask.sum()} TN from TAVG/TX")
        
        # Interpolate remaining values
        for param in ['TX', 'TN', 'TAVG']:
            if param in df.columns and df[param].isna().any():
                # Try cubic spline first
                df[param] = df[param].interpolate(method='cubic', limit_direction='both')
                
                # Fallback to seasonal median
                if df[param].isna().any():
                    for month in range(1, 13):
                        month_mask = (df.index.month == month) & df[param].isna()
                        if month_mask.sum() > 0:
                            monthly_median = df[df.index.month == month][param].median()
                            if not np.isnan(monthly_median):
                                df.loc[month_mask, param] = monthly_median
                
                logger.info(f"Interpolated remaining {param} values")
        
        for temp_param in ['TX', 'TN', 'TAVG']:
            if temp_param in df.columns:
                confidence = self._calculate_interpolation_confidence(df, temp_param)
                self.imputation_confidence[temp_param] = confidence
                logger.debug(f"{temp_param} imputation confidence: {confidence:.3f}")
        
        return df

    def _impute_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute humidity using dewpoint relationship
        
        Dewpoint formula: Td ≈ T - ((100 - RH) / 5)
        """
        # Calculate dewpoint where both TAVG and RH_AVG are available
        mask = df['RH_AVG'].notna() & df['TAVG'].notna()
        
        if mask.sum() > 50:  # Need enough data
            df.loc[mask, 'dewpoint'] = df.loc[mask, 'TAVG'] - ((100 - df.loc[mask, 'RH_AVG']) / 5)
            
            # Interpolate dewpoint
            df['dewpoint'] = df['dewpoint'].interpolate(method='cubic', limit_direction='both')
            
            # Calculate RH_AVG from dewpoint
            rh_missing = df['RH_AVG'].isna() & df['TAVG'].notna() & df['dewpoint'].notna()
            if rh_missing.sum() > 0:
                df.loc[rh_missing, 'RH_AVG'] = 100 - 5 * (df.loc[rh_missing, 'TAVG'] - df.loc[rh_missing, 'dewpoint'])
                logger.info(f"      Calculated {rh_missing.sum()} RH_AVG from dewpoint")
            
            df.drop('dewpoint', axis=1, inplace=True, errors='ignore')
        
        # Interpolate remaining values
        if df['RH_AVG'].isna().any():
            df['RH_AVG'] = df['RH_AVG'].interpolate(method='cubic', limit_direction='both')
            logger.info(f"      Interpolated remaining RH_AVG values")
        
        # Apply physical constraints (0-100%)
        df['RH_AVG'] = df['RH_AVG'].clip(0, 100)
        
        # NEW: Calculate and store confidence for uncertainty penalty
        if 'RH_AVG' in df.columns:
            confidence = self._calculate_interpolation_confidence(df, 'RH_AVG')
            self.imputation_confidence['RH_AVG'] = confidence
            logger.debug(f"RH_AVG imputation confidence: {confidence:.3f}")
        
        return df
    
    def _impute_wind_speed_gap_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gap-aware wind speed imputation - DON'T force completion for long gaps
        """
        
        # TIER 1: FF_X imputation with gap limits
        if 'FF_X' in df.columns:
            param_config = self.options.get("parameter_configs", {}).get("FF_X", {})
            max_gap = param_config.get("max_gap_interpolate", 3)
            skip_long_gaps = param_config.get("skip_long_gaps", False)
            
            ff_x_missing = df['FF_X'].isna().sum()
            
            if ff_x_missing > 0:
                logger.info(f"  • FF_X: Processing {ff_x_missing} missing values (gap-aware)...")
                
                # Only interpolate SHORT gaps
                df['FF_X'] = df['FF_X'].interpolate(
                    method='linear', 
                    limit=max_gap,  # 3 days only
                    limit_direction='both'
                )
                
                final_missing = df['FF_X'].isna().sum()
                imputed_count = ff_x_missing - final_missing
                
                if skip_long_gaps and final_missing > 0:
                    logger.info(
                        f"    ✓ FF_X: Imputed {imputed_count}/{ff_x_missing} values "
                        f"({final_missing} remain due to long gaps - NOT imputed by design)"
                    )
                else:
                    logger.info(f"    ✓ Imputed {imputed_count}/{ff_x_missing} FF_X values")
        
        # TIER 2: FF_AVG imputation with calculation + gap limits
        if 'FF_AVG' in df.columns and 'FF_X' in df.columns:
            param_config = self.options.get("parameter_configs", {}).get("FF_AVG", {})
            max_gap = param_config.get("max_gap_interpolate", 3)
            skip_long_gaps = param_config.get("skip_long_gaps", False)
            
            missing_count = df['FF_AVG'].isna().sum()
            
            if missing_count > 0:
                logger.info(f"  • FF_AVG: Processing {missing_count} missing values (gap-aware)...")
                
                # Strategy 1: Calculate from FF_X where possible
                calculated = 0
                can_calculate = df['FF_AVG'].isna() & df['FF_X'].notna()
                
                if can_calculate.sum() > 0:
                    # Use typical ratio FF_AVG ≈ 0.6 * FF_X
                    df.loc[can_calculate, 'FF_AVG'] = df.loc[can_calculate, 'FF_X'] * 0.6
                    calculated = can_calculate.sum()
                    logger.info(f"    Calculated {calculated} FF_AVG from FF_X ratio")
                
                # Strategy 2: Interpolate SHORT gaps only
                df['FF_AVG'] = df['FF_AVG'].interpolate(
                    method='linear',
                    limit=max_gap,
                    limit_direction='both'
                )
                
                final_missing = df['FF_AVG'].isna().sum()
                total_imputed = missing_count - final_missing
                interpolated = total_imputed - calculated
                
                if skip_long_gaps and final_missing > 0:
                    logger.info(
                        f"    ✓ FF_AVG: Calculated {calculated}, Interpolated {interpolated} "
                        f"({final_missing} remain due to long gaps - NOT imputed by design)"
                    )
                else:
                    logger.info(f"    ✓ Imputed {total_imputed}/{missing_count} FF_AVG values")
                
                # Store detailed results
                if "imputation" not in self.preprocessing_report:
                    self.preprocessing_report["imputation"] = {}
                
                self.preprocessing_report["imputation"]["FF_AVG"] = {
                    "missing_count": int(missing_count),
                    "calculated_from_ff_x": int(calculated),
                    "interpolated": int(interpolated),
                    "remaining_missing": int(final_missing),
                    "method": "ratio_calculation + linear_gap_aware",
                    "max_gap_days": max_gap,
                    "note": "Long gaps not imputed by design" if final_missing > 0 else None
                }
        
        return df
        

       
    
    def _impute_wind_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute wind direction using circular mean
        """
        
        def circular_mean(angles):
            """Calculate circular mean for angles in degrees"""
            if len(angles) == 0 or angles.isna().all():
                return np.nan
            
            angles_rad = np.radians(angles.dropna())
            x_mean = np.mean(np.cos(angles_rad))
            y_mean = np.mean(np.sin(angles_rad))
            
            mean_angle = np.degrees(np.arctan2(y_mean, x_mean))
            return (mean_angle + 360) % 360
        
        # Calculate monthly circular means
        for month in range(1, 13):
            month_data = df[df.index.month == month]['DDD_X'].dropna()
            if not month_data.empty:
                month_mean = circular_mean(month_data)
                
                month_mask = (df.index.month == month) & df['DDD_X'].isna()
                if month_mask.sum() > 0 and not np.isnan(month_mean):
                    df.loc[month_mask, 'DDD_X'] = month_mean
        
        # Fallback to seasonal mean
        for season in ['Wet', 'Dry']:
            season_data = df[df['Season'] == season]['DDD_X'].dropna()
            if not season_data.empty:
                season_mean = circular_mean(season_data)
                
                season_mask = (df['Season'] == season) & df['DDD_X'].isna()
                if season_mask.sum() > 0 and not np.isnan(season_mean):
                    df.loc[season_mask, 'DDD_X'] = season_mean
        
        logger.info(f"Applied circular mean imputation for DDD_X")
        
        return df
    
    def _impute_sunshine(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute sunshine duration with interpolation
        """
        
        if df['SS'].isna().any():
            df['SS'] = df['SS'].interpolate(method='cubic', limit_direction='both')
            
            # Apply physical constraint (0-14 hours)
            df['SS'] = df['SS'].clip(0, 14)
            logger.info(f"Interpolated SS (Sunshine) values")
        
        return df
    
    def _impute_cardinal_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute cardinal wind direction using mode
        """
        
        # Convert to string type
        df['DDD_CAR'] = df['DDD_CAR'].astype(str)
        df.loc[df['DDD_CAR'] == 'nan', 'DDD_CAR'] = np.nan
        
        # Use monthly mode
        for month in range(1, 13):
            month_data = df[df.index.month == month]['DDD_CAR'].dropna()
            if not month_data.empty:
                month_mode = month_data.mode()[0] if len(month_data.mode()) > 0 else None
                
                if month_mode:
                    month_mask = (df.index.month == month) & df['DDD_CAR'].isna()
                    df.loc[month_mask, 'DDD_CAR'] = month_mode
        
        # Fallback to overall mode
        if df['DDD_CAR'].isna().any():
            overall_mode = df['DDD_CAR'].mode()[0] if len(df['DDD_CAR'].mode()) > 0 else 'N'
            df['DDD_CAR'] = df['DDD_CAR'].fillna(overall_mode)
        
        logger.info(f"Applied mode-based imputation for DDD_CAR")
        
        return df
    def _apply_physical_constraints(
        self,
        df: pd.DataFrame
    )-> pd.DataFrame:
        """
        Apply physical constraints to ensure data consistency
        
        Constraints:
        - TX ≥ TAVG ≥ TN
        - FF_X ≥ FF_AVG
        - 0 ≤ RH_AVG ≤ 100
        - RR ≥ 0, SS ≥ 0
        - 0 ≤ DDD_X < 360
        """
        constraint_stats = {}
        
        # Temperature constraints 
        if all(p in df.columns for p in ['TX', 'TAVG', 'TN']):
            # TAVG should not exceed TX
            tavg_tx_violations = (df['TAVG'] > df['TX']).sum()
            if tavg_tx_violations > 0:
                df.loc[df['TAVG'] > df['TX'], 'TAVG'] = df.loc[df['TAVG'] > df['TX'], 'TX']
                constraint_stats['TAVG_TX'] = int(tavg_tx_violations)
            
            # TAVG should not be less than TN
            tavg_tn_violations = (df['TAVG'] < df['TN']).sum()
            if tavg_tn_violations > 0:
                df.loc[df['TAVG'] < df['TN'], 'TAVG'] = df.loc[df['TAVG'] < df['TN'], 'TN']
                constraint_stats['TAVG_TN'] = int(tavg_tn_violations)
        
        # Wind speed constraint: FF_X ≥ FF_AVG
        if 'FF_X' in df.columns and 'FF_AVG' in df.columns:
            ff_violations = (df['FF_AVG'] > df['FF_X']).sum()
            if ff_violations > 0:
                df.loc[df['FF_AVG'] > df['FF_X'], 'FF_AVG'] = df.loc[df['FF_AVG'] > df['FF_X'], 'FF_X']
                constraint_stats['FF_AVG'] = int(ff_violations)
        
        # Humidity bounds: 0-100%
        if 'RH_AVG' in df.columns:
            rh_violations = ((df['RH_AVG'] < 0) | (df['RH_AVG'] > 100)).sum()
            if rh_violations > 0:
                df['RH_AVG'] = df['RH_AVG'].clip(0, 100)
                constraint_stats['RH_AVG'] = int(rh_violations)
        
        # Non-negative constraints
        for param in ['RR', 'SS']:
            if param in df.columns:
                negative_count = (df[param] < 0).sum()
                if negative_count > 0:
                    df[param] = df[param].clip(lower=0)
                    constraint_stats[param] = int(negative_count)
        
        # Wind direction: 0-360
        if 'DDD_X' in df.columns:
            df['DDD_X'] = df['DDD_X'] % 360
        
        self.preprocessing_report["physical_constraints"] = constraint_stats
        
        if constraint_stats:
            total_fixes = sum(constraint_stats.values())
            logger.info(f"Applied {total_fixes} physical constraint fixes")
        else:
            logger.info("No physical constraint violations found")
        
        return df
    
    def _validate_preprocessing_quality(
        self,
        df: pd.DataFrame
    )-> None:
        """
        Validate preprocessing quality using GCV and trend preservation
        
        Phase 3 Implementation:
        - Calculate GCV (Generalized Cross-Validation) scores
        - Measure trend preservation percentage
        - Determine quality status per parameter
        - Generate warnings for poor quality
        
        Quality Thresholds:
        - Excellent: GCV < 2.0 AND Trend > 80%
        - Good: GCV < 4.0 AND Trend > 75%
        - Fair: GCV < 10.0 AND Trend > 70%
        - Poor: Otherwise
        """
        
        # Place holder for phase 3 implementation
        logger.info("Validating preprocessing quality (GCV + Trend Preservation)...")
        
        params = self.options.get("columns_to_process", [])
        validation_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Skip non numeric parameters
            if param == 'DDD_CAR':
                validation_results[param] = {
                    "status": "skipped_non_numeric",
                    "gcv_score": None,
                    "trend_preservation_pct": None
                }
                continue
            
            # check if parameters had any imputation
            imputation_stats = self.preprocessing_report.get("imputation", {})
            param_was_processed = param in imputation_stats
            
            if not param_was_processed:
                validation_results[param] = {
                    "status": "not_preprocessing",
                    "gcv_score": None,
                    "trend_preservation_pct": None,
                    "message": "No imputation performed"
                }
                continue
            
            # Validate data availability
            if self.original_data is None:
                logger.warning(f"Original data not available for {param} validation")
                validation_results[param] = {
                    "status": "missing_original_data",
                    "gcv_score": None,
                    "trend_preservation_pct": None
                }
                continue
            
            if param not in self.original_data.columns:
                logger.warning(f"Parameter {param} not in original data")
                validation_results[param] = {
                    "status": "missing_in_original",
                    "gcv_score": None,
                    "trend_preservation_pct": None
                }
                continue
                
            try:
                # Extract alligned time series data
                original_series, processed_series = self._align_time_series(
                    self.original_data, df, param
                )
                if original_series is None or processed_series is None:
                    validation_results[param] = {
                        "status": "alignment_failed",
                        "gcv_score": None,
                        "trend_preservation_pct": None
                    }
                    continue
                
                # Check minimum data requirement
                if len(original_series) < 30:
                    validation_results[param] = {
                        "status": "insufficient_data",
                        "data_points": len(original_series),
                        "gcv_score": None,
                        "trend_preservation_pct": None
                    }
                    continue
                # Metric 1: Calculate GCV Score
                gcv_score = self._calculate_gcv_score(
                    original_series, processed_series, param
                )
                
                # Metric 2: Calculate Trend Preservation
                trend_preservation = self._calculate_trend_preservation_pct(
                    original_series, processed_series
                )
                
                # Metric 3: Determine Quality Status
                quality_status = self._determine_quality_status(
                    gcv_score, trend_preservation, param
                )
                
                # Store results
                validation_results[param] = {
                    "gcv_score": round(float(gcv_score), 4),
                    "trend_preservation_pct": round(float(trend_preservation), 2),
                    "quality_status": quality_status,
                    "data_points": len(original_series),
                    "imputation_method": imputation_stats.get(param, {}).get("method", "unknown")
                }
                
                # Log results
                logger.info(
                    f"  {param}: GCV={gcv_score:.4f}, "
                    f"Trend={trend_preservation:.2f}%, "
                    f"Quality={quality_status.upper()}"
                )
                
                # Generate warnings for poor quality
                if quality_status == "poor":
                    warning_msg = (
                        f"Parameter {param}: Poor preprocessing quality detected "
                        f"(GCV={gcv_score:.3f}, Trend={trend_preservation:.1f}%)"
                    )
                    self.preprocessing_report["warnings"].append(warning_msg)
                    logger.warning(f"{warning_msg}")
                    
                elif quality_status == "fair":
                    logger.info(f"{param}: Fair quality - acceptable but not optimal")
                
                if param == 'FF_AVG' and 'FF_AVG' in df.columns and self.original_data is not None:
                    logger.info("FF_AVG SAMPLE COMPARISON")
                    
                    # Debug: Check index types
                    logger.info(f"original_data index type: {type(self.original_data.index)}")
                    logger.info(f"processed df index type: {type(df.index)}")
                    
                    # Get samples where original was 0.0
                    zero_samples = self.original_data[self.original_data['FF_AVG'] == 0.0].head(20)
                    logger.info(f"Found {len(zero_samples)} zero samples in original data")
                    
                    if len(zero_samples) > 0:
                        # Print header
                        logger.info(f"{'Date':<12} {'FF_X':>6} {'Original':>10} {'Processed':>10} {'Diff':>8} {'Status'}")
                        
                        # Iterate through samples
                        matched_count = 0
                        for idx in zero_samples.index:
                            # FIX: Check if index exists in processed df
                            if idx in df.index:
                                matched_count += 1
                                
                                # Get values
                                ff_x = df.loc[idx, 'FF_X'] if 'FF_X' in df.columns else np.nan
                                orig = self.original_data.loc[idx, 'FF_AVG']
                                proc = df.loc[idx, 'FF_AVG']
                                diff = proc - orig
                                
                                # Determine status
                                if pd.notna(ff_x) and ff_x > 0 and orig == 0:
                                    status = "SUSPICIOUS"
                                else:
                                    status = "OK"
                                
                                # Format and log
                                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                                logger.info(
                                    f"{date_str:<12} {ff_x:>6.1f} {orig:>10.1f} "
                                    f"{proc:>10.1f} {diff:>8.1f} {status}"
                                )
                            else:
                                # Debug: Index not found
                                logger.debug(f"Index {idx} not found in processed df")
                        
                        logger.info(f"Displayed {matched_count}/{len(zero_samples)} samples")
                        
                        if matched_count == 0:
                            logger.warning(
                                "No matching indices found! "
                                "This indicates index type mismatch between original_data and processed df."
                            )
                            logger.warning(f"original_data index sample: {self.original_data.index[:5].tolist()}")
                            logger.warning(f"processed df index sample: {df.index[:5].tolist()}")
                    else:
                        logger.info("No FF_AVG = 0.0 values found in original data")
                    
            except Exception as e:
                logger.error(f"Error validating {param}: {str(e)}")
                validation_results[param] = {
                    "status": "error",
                    "error": str(e),
                    "gcv_score": None,
                    "trend_preservation_pct": None
                    
                }
        # store in preprocessing report
        self.preprocessing_report["smoothing_validation"] = validation_results
        # generate summary statistics
        self._generate_quality_summary(validation_results)
        logger.info(f"Quality validation completed for {len(validation_results)}")
    
    def _align_time_series(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        param: str
    )-> tuple:
        """
        Align time series on common dates for gcv evaluation
        
        Conditional reference based on imputed classification
        - Edge case handling for sparse rain months
        - Outlier-resistant median calculation
        
        Strategy:
        - For originally valid values: Compare original vs processed
        - For originally missing values (imputed):
            * IF processed = 0 (dry) → reference = 0
            * IF processed > 0 (rain) → reference = conditional median (rain days only)
        
        This evaluates BOTH preservation and imputation quality.
        """
        try:
            # Ensure both DataFrames have Date as index
            if 'Date' in original_df.columns:
                original_df = original_df.set_index('Date')

            if 'Date' in processed_df.columns:
                processed_df = processed_df.set_index('Date')
            
            # Extract parameter series
            original = original_df[param]
            processed = processed_df[param]
            
            # Find common dates
            common_idx = original.index.intersection(processed.index)
            
            if len(common_idx) == 0:
                logger.warning(f"No common dates for {param}")
                return None, None

            # Extract aligned series
            original_aligned = original.loc[common_idx]
            processed_aligned = processed.loc[common_idx]
            
            comparison_original = []
            comparison_processed = []
            
            for idx in common_idx:
                orig_val = original_aligned.loc[idx]
                proc_val = processed_aligned.loc[idx]
                
                # Skip if processed value is missing (shouldn't happen)
                if pd.isna(proc_val):
                    continue
                
                if pd.notna(orig_val):
                    # CASE 1: Original value exists
                    # Direct comparison - preserve original vs processed
                    comparison_original.append(orig_val)
                    comparison_processed.append(proc_val)
                else:
                    # CASE 2: Original missing (imputed value)
                    if param == 'RR':
                        # RAINFALL: Conditional reference based on classification
                        if proc_val == 0:
                            # Classified as dry day → reference = 0
                            reference = 0.0
                        else:
                            # Classified as rain day → conditional median
                            reference = self._get_rainfall_conditional_median(
                                original_df, idx
                            )
                    else:
                        # OTHER PARAMETERS: Standard seasonal median
                        reference = self._get_seasonal_median_reference(
                            original_df, idx, param
                        )
                    
                    # Only add to comparison if reference is valid
                    if pd.notna(reference):
                        comparison_original.append(reference)
                        comparison_processed.append(proc_val)
            
            if len(comparison_original) == 0:
                logger.warning(f"No valid comparison points for {param}")
                return None, None
            
            original_final = np.array(comparison_original)
            processed_final = np.array(comparison_processed)
            
            # Count preserved vs imputed
            valid_count = (original_aligned.notna() & processed_aligned.notna()).sum()
            imputed_count = len(comparison_original) - valid_count
            
            logger.debug(
                f"{param}: {len(original_final)} comparison points "
                f"({valid_count} preserved, {imputed_count} imputed vs reference)"
            )
            
            return original_final, processed_final
        except Exception as e:
            logger.error(f"Error aligning time series for {param}: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
    
    def _get_rainfall_conditional_median(
        self,
        original_df: pd.DataFrame,
        idx
    )-> float:
        """
        Get conditional median rainfall reference (rain days only):
        - IQR-filtered median (outlier-resistant)
        - Priority cascade: Monthly → Seasonal → Overall
        - Edge case handling for sparse data
        
        Args:
            original_df: Original dataset
            idx: Date index for reference calculation
        
        Returns:
            Conditional median rainfall amount (mm)
        """
        
        try:
            month = idx.month
            season = 'Wet' if month in self.season_config['wet_months'] else 'Dry'
            # PRIORITY 1: Monthly conditional median
            monthly_rain = original_df[
                (original_df.index.month == month) & 
                (original_df['RR'] > 0) &
                (original_df['RR'].notna())
            ]['RR']
            
            if len(monthly_rain) >= 5:
                # Enough data - use IQR-filtered median
                reference = self._calculate_robust_median(monthly_rain)
                
                if pd.notna(reference):
                    logger.debug(f"RR reference for {idx}: Monthly median = {reference:.2f}mm")
                    return reference
            
            # PRIORITY 2: Seasonal conditional median
            seasonal_months = (
                self.season_config['wet_months'] if season == 'Wet' 
                else self.season_config['dry_months']
            )
            
            seasonal_rain = original_df[
                (original_df.index.month.isin(seasonal_months)) &
                (original_df['RR'] > 0) &
                (original_df['RR'].notna())
            ]['RR']
            
            if len(seasonal_rain) >= 10:
                reference = self._calculate_robust_median(seasonal_rain)
                
                if pd.notna(reference):
                    logger.debug(f"RR reference for {idx}: Seasonal median = {reference:.2f}mm")
                    return reference
            
            # PRIORITY 3: Overall conditional median
            overall_rain = original_df[
                (original_df['RR'] > 0) &
                (original_df['RR'].notna())
            ]['RR']
            
            if len(overall_rain) >= 20:
                reference = self._calculate_robust_median(overall_rain)
                
                if pd.notna(reference):
                    logger.debug(f"RR reference for {idx}: Overall median = {reference:.2f}mm")
                    return reference
            
            # PRIORITY 4: Hardcoded fallback
            if season == 'Wet':
                fallback = 8.0  # Wet season typical rain day
            else:
                fallback = 2.0  # Dry season typical rain day
            
            logger.debug(f"RR reference for {idx}: Fallback = {fallback}mm (insufficient data)")
            return fallback

        except Exception as e:
            logger.debug(f"Error calculating rainfall reference for {idx}: {str(e)}")
            # Emergency fallback
            return 5.0
    
    def _calculate_robust_median(
        self,
        data: pd.Series
    )-> float:
        """
        Calculate outlier-resistant median using IQR filtering
        - Removes extreme outliers (>Q3 + 1.5*IQR)
        - Falls back to standard median if insufficient data
        
        Args:
            data: Pandas Series of values
        
        Returns:
            Robust median value
        """
        try:
            if len(data) < 3:
                return data.median()
            
            # Calculate IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Filter outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            filtered_data = data[
                (data >= lower_bound) & 
                (data <= upper_bound)
            ]
            
            if len(filtered_data) >= 3:
                return filtered_data.median()
            else:
                # Fallback to standard median
                return data.median()
            
        except Exception as e:
            logger.debug(f"Error calculating robust median: {str(e)}")
            return data.median()
    
    def _get_seasonal_median_reference(
        self,
        original_df: pd.DataFrame,
        idx,
        param: str
    ) -> float:
        """
        Get seasonal median reference for non-rainfall parameters
        
        Args:
            original_df: Original dataset
            idx: Date index
            param: Parameter name
        
        Returns:
            Seasonal median value
        """
        try:
            month = idx.month
            
            # Monthly median
            monthly_data = original_df[
                (original_df.index.month == month) &
                original_df[param].notna()
            ][param]
            
            if len(monthly_data) >= 5:
                return monthly_data.median()
            
            # Seasonal fallback
            season = 'Wet' if month in self.season_config['wet_months'] else 'Dry'
            seasonal_months = (
                self.season_config['wet_months'] if season == 'Wet'
                else self.season_config['dry_months']
            )
            
            seasonal_data = original_df[
                (original_df.index.month.isin(seasonal_months)) &
                original_df[param].notna()
            ][param]
            
            if len(seasonal_data) >= 10:
                return seasonal_data.median()
            
            # Overall median
            return original_df[param].median()
            
        except Exception as e:
            logger.debug(f"Error getting seasonal reference for {param}: {str(e)}")
            return np.nan
    
    def _calculate_gcv_score(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        param: str
    ) -> float:
        """
        Calculate GCV score with adaptive CV-based normalization
        
        PHASE 1 FIX FOR FF_AVG:
        - High CV params (>0.5): Use mean² normalization
        - Moderate CV (0.2-0.5): Use variance normalization
        - Low CV (<0.2): Use variance normalization
        - Apply CV tolerance factor for chaotic parameters
        
        Expected Impact:
        - FF_AVG GCV: 3045 → 30-50 (after Function 1 MSE reduction)
        - Scale-independent comparison
        - Physically meaningful thresholds
        """
        try:
            n = len(original)
            if n == 0:
                return np.nan
            
            # STEP 1: Calculate MSE
            mse = np.mean((processed - original) ** 2)
            
            # STEP 2: Calculate CV (Coefficient of Variation)
            param_mean = np.mean(original)
            param_std = np.std(original)
            param_variance = np.var(original)
            
            if param_mean > 0:
                cv = param_std / param_mean
            else:
                cv = 0
            
            # STEP 3: ADAPTIVE NORMALIZATION based on CV
            # Define normalize_by in all branches
            if cv > 0.5:
                # HIGH CV (like FF_AVG): Use mean² normalization
                if param_mean > 0:
                    normalize_by = param_mean ** 2
                    normalized_mse = mse / normalize_by
                    normalization_method = "mean_squared"
                    
                    logger.debug(
                        f"{param}: CV={cv:.2f} (high) → Mean² normalization "
                        f"(MSE={mse:.4f}, Mean²={normalize_by:.4f}, "
                        f"Normalized={normalized_mse:.4f})"
                    )
                else:
                    # Fallback
                    normalize_by = 1.0
                    normalized_mse = mse
                    normalization_method = "none"
            
            elif cv > 0.2:
                # MODERATE CV: Use variance normalization
                if param_variance > 0.01:
                    normalize_by = param_variance
                    normalized_mse = mse / param_variance
                    normalization_method = "variance"
                    
                    logger.debug(
                        f"{param}: CV={cv:.2f} (moderate) → Variance normalization "
                        f"(MSE={mse:.4f}, Var={param_variance:.4f})"
                    )
                else:
                    normalize_by = 1.0
                    normalized_mse = mse
                    normalization_method = "none"
            
            else:
                # LOW CV: Use variance normalization (standard)
                if param_variance > 0.01:
                    normalize_by = param_variance
                    normalized_mse = mse / param_variance
                    normalization_method = "variance"
                else:
                    normalize_by = 1.0
                    normalized_mse = mse
                    normalization_method = "none"
            
            # STEP 4: Apply CV Tolerance Factor (for high-CV params)
            if cv > 0.5:
                tolerance_factor = 1.0 + (cv - 0.5)
                adjusted_mse = normalized_mse / tolerance_factor
                
                logger.debug(
                    f"{param}: CV tolerance applied "
                    f"(factor={tolerance_factor:.2f}, "
                    f"adjusted={adjusted_mse:.4f})"
                )
            else:
                tolerance_factor = 1.0
                adjusted_mse = normalized_mse
            
            # STEP 5: Estimate EDF (Effective Degrees of Freedom)
            imputation_stats = self.preprocessing_report.get("imputation", {})
            param_method = imputation_stats.get(param, {}).get("method", "unknown")
            
            # Method-specific EDF
            if "two_stage" in param_method.lower() or "two_tier" in param_method.lower():
                edf = 2  # Two-tier seasonal imputation
            elif "seasonal" in param_method.lower():
                edf = 2
            elif "spline" in param_method.lower() or "cubic" in param_method.lower():
                edf = 4
            elif "linear" in param_method.lower():
                edf = 2
            else:
                edf = 3
            
            # Ensure EDF is reasonable
            edf = max(1.0, min(edf, n / 3.0))
            
            # STEP 6: Calculate GCV
            denominator = (1.0 - edf / n) ** 2
            denominator = max(denominator, 0.01)
            
            gcv = adjusted_mse / denominator
            
            # STEP 7: Enhanced logging for SS
            
            if param in ['SS', 'FF_AVG']:
                logger.info(f"{param} GCV DEBUG")
                logger.info(f"original array sample: {original[:20]}")
                logger.info(f"processed array sample: {processed[:20]}")
                logger.info(f"Mean: {param_mean:.4f}")
                logger.info(f"Std: {param_std:.4f}")
                logger.info(f"Variance: {param_variance:.4f}")
                logger.info(f"CV: {cv:.4f}")
                logger.info(f"MSE (raw): {mse:.4f}")
                logger.info(f"Normalization method: {normalization_method}")
                logger.info(f"normalize_by: {normalize_by:.4f}")
                logger.info(f"normalized_mse: {normalized_mse:.4f}")
                if cv > 0.5:
                    logger.info(f"tolerance_factor: {tolerance_factor:.4f}")
                    logger.info(f"adjusted_mse: {adjusted_mse:.4f}")
                logger.info(f"EDF: {edf:.1f}")
                logger.info(f"denominator: {denominator:.6f}")
                logger.info(f"Final GCV: {gcv:.4f}")
            
            return gcv            
        except Exception as e:
            logger.error(f"GCV calculation error for {param}: {str(e)}")
            logger.error(traceback.format_exc())
            return np.nan
    
    def _calculate_trend_preservation_pct(
        self,
        original: np.ndarray,
        processed: np.ndarray
    )-> float:
        """
        Calculate trend preservation percentage
        
        Measures how well preprocessing maintains directional changes
        
        Returns:
            Percentage (0-100) of matching trend directions
        """
        
        if len(original) != len(processed):
            logger.warning("Trend preservation: length mismatch")
            return 0.0
        
        if len(original) < 2:
            return 0.0 
        
        # Calculate first differences (trend direction)
        original_diff = np.diff(original)
        processed_diff = np.diff(processed)
        
        # Get direction signs
        original_direction = np.sign(original_diff)
        processed_direction = np.sign(processed_diff)
        
        # Exclude flat regions (0 values)
        non_zero_mask = (original_direction != 0) & (processed_direction != 0)
        
        if non_zero_mask.sum() == 0:
            # All flat data = perfect preservation
            return 100.0
        
        # Calculate agreement
        agreement = (original_direction[non_zero_mask] == processed_direction[non_zero_mask])
        trend_preservation = agreement.mean() * 100
        
        return trend_preservation
        
    def _determine_quality_status(
        self,
        gcv_score: float,
        trend_preservation: float,
        param: str = None
    )-> str:
        """
        Determine overall preprocessing quality status
        
        Quality Tiers:
        - Excellent: GCV < 2.0 AND Trend > 80%
        - Good: GCV < 4.0 AND Trend > 75%
        - Fair: GCV < 10.0 AND Trend > 70%
        - Poor: Otherwise
        
        Args:
            gcv_score: GCV score (lower is better)
            trend_preservation: Trend preservation % (higher is better)
        
        Returns:
            Quality status string
        """
        if gcv_score < 2.0 and trend_preservation > 80.0:
            return "excellent"
        elif gcv_score < 4.0 and trend_preservation > 75.0:
            return "good"
        elif gcv_score < 10.0 and trend_preservation > 70.0:
            return "fair"
        else:
            return "poor"
        
    def _generate_quality_summary(
        self,
        validation_results: Dict[str, Any]
    )-> None:
        """
        Generate summary statistics for quality validation
        
        Aggregates:
        - Count of parameters by quality status
        - Average GCV and trend preservation
        - Overall quality assessment
        """
        
        # filter valid results
        valid_results = {
            param: result for param, result in validation_results.items()
            if isinstance(result, dict) and "quality_status" in result
        }
        
        if not valid_results:
            logger.warning("No valid quality validation results")
            return
        
        # Count by quality status
        quality_counts = {}
        gcv_scores = []
        trend_preservations = []
        
        for result in valid_results.values():
            status = result.get("quality_status")
            if status:
                quality_counts[status] = quality_counts.get(status, 0) + 1
            
            gcv = result.get("gcv_score")
            if gcv is not None:
                gcv_scores.append(gcv)
            
            trend = result.get("trend_preservation_pct")
            if trend is not None:
                trend_preservations.append(trend)
                
        # Calculate averages
        avg_gcv = np.mean(gcv_scores) if gcv_scores else None
        avg_trend = np.mean(trend_preservations) if trend_preservations else None
        
        # determine overall quality
        excellent_count = quality_counts.get("excellent", 0)
        good_count = quality_counts.get("good", 0)
        total_valid = len(valid_results)
        
        if excellent_count + good_count >= total_valid * 0.8:
            overall_quality = "excellent"
        elif excellent_count + good_count >= total_valid * 0.6:
            overall_quality = "good"
        elif quality_counts.get("fair", 0) + excellent_count + good_count >= total_valid * 0.7:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
            
        # store summary
        quality_summary = {
            "overall_quality": overall_quality,
            "parameters_validated": total_valid,
            "quality_distribution": quality_counts,
            "average_gcv_score": round(avg_gcv, 4) if avg_gcv is not None else None,
            "average_trend_preservation": round(avg_trend, 2) if avg_trend is not None else None
        }
        self.preprocessing_report["quality_summary"] = quality_summary
        logger.info("QUALITY VALIDATION SUMMARY")
        logger.info(f"Overall Quality: {overall_quality.upper()}")
        logger.info(f"Parameters Validated: {total_valid}")
        logger.info(f"Quality Distribution:")
        for status, count in sorted(quality_counts.items()):
            logger.info(f"  - {status.capitalize()}: {count}")
        if avg_gcv is not None:
            logger.info(f"Average GCV Score: {avg_gcv:.4f}")
        if avg_trend is not None:
            logger.info(f"Average Trend Preservation: {avg_trend:.2f}%")
            
            
    def _calculate_model_coverage(self, df: pd.DataFrame) -> None:
        """
        Calculate coverage for Holt-Winters and LSTM models
        
        Coverage criteria:
        - Holt-Winters: Seasonality, stationarity, gap impact, smoothing quality
        - LSTM: Sequence continuity, extreme values, trend preservation
        """
        
        logger.info("Calculate model coverage for Holt-Winters and LSTM models")
        params = self.options.get("columns_to_process", [])
        per_parameter_coverage = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # skip non-numeric parameters
            if param == 'DDD_CAR':
                per_parameter_coverage[param] = {
                    "status": "skipped_non_numeric",
                    "holt_winters_coverage": None,
                    "lstm_coverage": None
                }
                continue
            logger.info(f"Analyzing{param}..")
            
            try:
                # STEP 1: Analyze data characteristics
                seasonality = self._analyze_seasonality(df, param)
                stationarity = self._test_stationarity(df, param)
                gaps = self._analyze_gaps(df, param)
                extreme_values = self._analyze_extreme_values(df, param)
                
                # Get smoothing quality from Phase 3
                smoothing_quality = self._get_smoothing_quality(param)
                
                # Calculate missing data ratio
                missing_ratio = df[param].isna().sum() / len(df)
                
                # Special handling for precipitation
                precipitation = None
                if param == 'RR':
                    precipitation = self._analyze_precipitation_extremes(df, param)
                
                # STEP 2: Calculate Holt-Winters coverage
                hw_coverage = self._calculate_hw_coverage(
                    seasonality=seasonality,
                    stationarity=stationarity,
                    gaps=gaps,
                    extreme_values=extreme_values,
                    smoothing_quality=smoothing_quality,
                    missing_ratio=missing_ratio,
                    param=param
                )
                
                # STEP 3: Calculate LSTM coverage (Phase 5 Batch 12)
                lstm_coverage = self._calculate_lstm_coverage(
                    gaps=gaps,
                    extreme_values=extreme_values,
                    precipitation=precipitation,
                    smoothing_quality=smoothing_quality,
                    missing_ratio=missing_ratio,
                    param=param
                )
                
                # STEP 4: Determine recommended model
                recommended_model = self._determine_recommended_model(
                    hw_coverage["coverage_percentage"],
                    lstm_coverage["coverage_percentage"]
                )
                
                # Store results
                per_parameter_coverage[param] = {
                    "holt_winters_coverage": hw_coverage["coverage_percentage"],
                    "holt_winters_uncovered": hw_coverage["uncovered_reasons"],
                    "lstm_coverage": lstm_coverage["coverage_percentage"],
                    "lstm_uncovered": lstm_coverage["uncovered_reasons"],
                    "recommended_model": recommended_model,
                    "seasonality_strength": seasonality.get("seasonal_strength", 0),
                    "is_stationary": stationarity.get("is_stationary", False)
                }
                
                logger.info(
                    f"    {param}: HW={hw_coverage['coverage_percentage']:.1f}%, "
                    f"LSTM={lstm_coverage['coverage_percentage']:.1f}%, "
                    f"Recommended={recommended_model}"
                )
            except Exception as e:
                logger.error(f"Error analyzing coverage for {param}: {str(e)}")
                per_parameter_coverage[param] = {
                    "status": "error",
                    "error": str(e)
                }

         # STEP 5: Aggregate uncovered reasons
        hw_aggregate = self._aggregate_uncovered_breakdown(
            per_parameter_coverage, "holt_winters"
        )
        lstm_aggregate = self._aggregate_uncovered_breakdown(
            per_parameter_coverage, "lstm"
        )
        
        # Calculate overall coverage
        hw_coverages = [
            v["holt_winters_coverage"] 
            for v in per_parameter_coverage.values() 
            if isinstance(v, dict) and "holt_winters_coverage" in v and v["holt_winters_coverage"] is not None
        ]
        lstm_coverages = [
            v["lstm_coverage"] 
            for v in per_parameter_coverage.values() 
            if isinstance(v, dict) and "lstm_coverage" in v and v["lstm_coverage"] is not None
        ]
        
        overall_hw = np.mean(hw_coverages) if hw_coverages else 0
        overall_lstm = np.mean(lstm_coverages) if lstm_coverages else 0
        
        # Store in report
        self.preprocessing_report["model_coverage"] = {
            "per_parameter": per_parameter_coverage,
            "aggregate_hw_uncovered": hw_aggregate,
            "aggregate_lstm_uncovered": lstm_aggregate,
            "overall_hw_coverage": round(overall_hw, 2),
            "overall_lstm_coverage": round(overall_lstm, 2)
        }
        
        logger.info(f"\n  Overall HW Coverage: {overall_hw:.2f}%")
        logger.info(f"  Overall LSTM Coverage: {overall_lstm:.2f}%")
        
    def _analyze_seasonality(
        self,
        df: pd.DataFrame,
        param: str
    )-> Dict[str, Any]:
        """
        Analyze seasonality strength using STL decomposition
        
        Returns:
            {
                "seasonal_strength": float (0-1),
                "has_seasonality": bool,
                "penalty": float (0-20)
            }
        """
        
        try:
            # Need at least 2 years of data for reliable STL
            if len(df) < 730:
                return {
                    "seasonal_strength": 0.0,
                    "has_seasonality": False,
                    "penalty": 20.0,  # Maximum penalty
                    "message": "Insufficient data for seasonality analysis (<2 years)"
                }
            
            # Get non-null values
            series = df[param].dropna()
            
            if len(series) < 730:
                return {
                    "seasonal_strength": 0.0,
                    "has_seasonality": False,
                    "penalty": 20.0,
                    "message": "Too many missing values for STL"
                }
            
            # Apply STL decomposition
            stl = STL(series, period=365, robust=True)
            result = stl.fit()
            
            # Calculate seasonality strength
            # Formula: 1 - Var(Residual) / Var(Detrended)
            detrended = series - result.trend
            var_residual = np.var(result.resid)
            var_detrended = np.var(detrended)
            
            if var_detrended > 0:
                seasonal_strength = max(0, 1 - (var_residual / var_detrended))
            else:
                seasonal_strength = 0.0
            
            # Determine if seasonality exists
            has_seasonality = seasonal_strength > 0.3
            
            # Calculate penalty (inverse relationship)
            # Strong seasonality (>0.6) = no penalty
            # Moderate (0.3-0.6) = 5-10% penalty
            # Weak (<0.3) = 10-20% penalty
            if seasonal_strength >= 0.6:
                penalty = 0.0
            elif seasonal_strength >= 0.3:
                penalty = 5.0 + (0.6 - seasonal_strength) * (5.0 / 0.3)
            else:
                penalty = 10.0 + (0.3 - seasonal_strength) * (10.0 / 0.3)
            
            return {
                "seasonal_strength": round(seasonal_strength, 3),
                "has_seasonality": has_seasonality,
                "penalty": round(penalty, 2),
                "message": f"Seasonality strength: {seasonal_strength:.3f}"
            }
            
        except Exception as e:
            logger.warning(f"Seasonality analysis failed for {param}: {str(e)}")
            return {
                "seasonal_strength": 0.0,
                "has_seasonality": False,
                "penalty": 15.0,  # Moderate penalty for analysis failure
                "error": str(e)
            }
        
    def _test_stationarity(
        self,
        df: pd.DataFrame,
        param: str
    )-> Dict[str, Any]:
        """
        Test stationarity using Augmented Dickey-Fuller test
        
        Returns:
            {
                "is_stationary": bool,
                "adf_statistic": float,
                "p_value": float,
                "penalty": float (0-15)
            }
        """
        try:
            # Get non-null values
            series = df[param].dropna()
            
            if len(series) < 50:
                return {
                    "is_stationary": False,
                    "adf_statistic": None,
                    "p_value": None,
                    "penalty": 15.0,
                    "message": "Insufficient data for ADF test"
                }
            
            # Perform ADF test
            result = adfuller(series, autolag='AIC')
            
            adf_statistic = result[0]
            p_value = result[1]
            
            # Determine stationarity (p-value < 0.05 = stationary)
            is_stationary = p_value < 0.05
            
            # Calculate penalty
            # Stationary (p < 0.05) = no penalty
            # Borderline (0.05-0.10) = 5% penalty
            # Non-stationary (p > 0.10) = 10-15% penalty
            if p_value < 0.05:
                penalty = 0.0
            elif p_value < 0.10:
                penalty = 5.0
            elif p_value < 0.20:
                penalty = 10.0
            else:
                penalty = 15.0
            
            return {
                "is_stationary": is_stationary,
                "adf_statistic": round(adf_statistic, 4),
                "p_value": round(p_value, 4),
                "penalty": penalty,
                "message": f"ADF p-value: {p_value:.4f}"
            }
            
        except Exception as e:
            logger.warning(f"Stationarity test failed for {param}: {str(e)}")
            return {
                "is_stationary": False,
                "adf_statistic": None,
                "p_value": None,
                "penalty": 10.0,  # Moderate penalty for test failure
                "error": str(e)
            }
            
    def _analyze_gaps(
        self,
        df: pd.DataFrame,
        param: str
    )-> Dict[str, Any]:
        """
        Analyze gap impact from preprocessing report
        
        Returns:
            {
                "large_gaps_count": int,
                "impact_percentage": float (0-15)
            }
        """
        gaps_report = self.preprocessing_report.get("gaps", {})
        # Count large gaps (>30 days)
        large_gaps = gaps_report.get("large_gaps", 0)
        total_gaps = gaps_report.get("total_gaps", 0)
        
        # Calculate impact percentage
        # Each large gap adds 2-3% penalty, capped at 15%
        if large_gaps == 0:
            impact_percentage = 0.0
        elif large_gaps <= 2:
            impact_percentage = large_gaps * 2.5
        elif large_gaps <= 5:
            impact_percentage = 5.0 + (large_gaps - 2) * 2.0
        else:
            impact_percentage = min(15.0, 11.0 + (large_gaps - 5) * 1.0)
        
        return {
            "large_gaps_count": large_gaps,
            "total_gaps": total_gaps,
            "impact_percentage": round(impact_percentage, 2),
            "message": f"{large_gaps} large gaps detected"
        }
    
    def _analyze_extreme_values(
        self, 
        df: pd.DataFrame,
        param: str
    )-> Dict[str, Any]:
        """
        Analyze extreme values/outliers from preprocessing report
        
        Returns:
            {
                "extreme_count": int,
                "impact_percentage": float (0-varies)
            }
        """
        
        # Get outlier information from preprocessing report
        outliers_report = self.preprocessing_report.get("outliers", {})
        param_outliers = outliers_report.get(param, {})
        
        outlier_count = param_outliers.get("count", 0)
        outlier_percentage = param_outliers.get("percentage", 0.0)
        
        # Calculate impact
        # 0-2%: minimal impact (0-2% penalty)
        # 2-5%: moderate impact (2-5% penalty)
        # >5%: high impact (5-10% penalty)
        if outlier_percentage <= 2.0:
            impact_percentage = outlier_percentage
        elif outlier_percentage <= 5.0:
            impact_percentage = 2.0 + (outlier_percentage - 2.0)
        else:
            impact_percentage = min(10.0, 5.0 + (outlier_percentage - 5.0) * 0.5)
        
        return {
            "extreme_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "impact_percentage": round(impact_percentage, 2),
            "message": f"{outlier_count} outliers ({outlier_percentage:.1f}%)"
        }
    
    def _analyze_precipitation_extremes(self, df: pd.DataFrame, param: str) -> Dict[str, Any]:
        """
        Analyze precipitation extremes (0-500mm range)
        
        Returns:
            {
                "zero_percentage": float,
                "extreme_percentage": float,
                "range_impact": float (0-1)
            }
        """
        
        series = df[param].dropna()
        
        if len(series) == 0:
            return {
                "zero_percentage": 0.0,
                "extreme_percentage": 0.0,
                "range_impact": 0.0
            }
        
        # Calculate zero percentage
        zero_count = (series == 0).sum()
        zero_percentage = (zero_count / len(series)) * 100
        
        # Calculate extreme percentage (>200mm)
        extreme_count = (series > 200).sum()
        extreme_percentage = (extreme_count / len(series)) * 100
        
        # Calculate range impact (0-1 scale)
        # High zero percentage (>50%) in wet season = problematic
        # High extreme percentage (>5%) = problematic
        zero_impact = min(1.0, zero_percentage / 50.0) if zero_percentage > 30 else 0
        extreme_impact = min(1.0, extreme_percentage / 5.0)
        
        range_impact = max(zero_impact, extreme_impact)
        
        return {
            "zero_percentage": round(zero_percentage, 2),
            "extreme_percentage": round(extreme_percentage, 2),
            "range_impact": round(range_impact, 3),
            "message": f"{zero_percentage:.1f}% zeros, {extreme_percentage:.1f}% extremes"
        }
    
    def _get_smoothing_quality(self, param: str) -> Dict[str, Any]:
        """
        Extract smoothing quality from Phase 3 validation results
        
        Returns:
            {
                "gcv_score": float,
                "trend_preservation_pct": float,
                "penalty": float (0-15),
                "trend_value": float (for trend penalty calc)
            }
        """
        
        # Get validation results from Phase 3
        validation_results = self.preprocessing_report.get("smoothing_validation", {})
        param_result = validation_results.get(param, {})
        
        gcv_score = param_result.get("gcv_score", 0)
        trend_preservation = param_result.get("trend_preservation_pct", 100)
        
        # Calculate GCV penalty
        # GCV < 2.0: excellent (0% penalty)
        # GCV 2.0-4.0: good (0-5% penalty)
        # GCV 4.0-10.0: fair (5-10% penalty)
        # GCV > 10.0: poor (10-15% penalty)
        if gcv_score < 2.0:
            gcv_penalty = 0.0
        elif gcv_score < 4.0:
            gcv_penalty = (gcv_score - 2.0) * 2.5
        elif gcv_score < 10.0:
            gcv_penalty = 5.0 + (gcv_score - 4.0) * 0.833
        else:
            gcv_penalty = min(15.0, 10.0 + (gcv_score - 10.0) * 0.5)
        
        return {
            "gcv_score": gcv_score,
            "trend_preservation_pct": trend_preservation,
            "penalty": round(gcv_penalty, 2),
            "trend_value": trend_preservation,  # For trend penalty calculation
            "message": f"GCV={gcv_score:.4f}, Trend={trend_preservation:.2f}%"
        }
        
    def _calculate_and_embed_decomposition(
        self,
        df: pd.DataFrame
    )-> None:
        """
        Calculate STL seasonal decomposition and embed results in preprocessing report
        
        Key Differences from NASA:
        - Uses STL decomposition (more robust for outliers)
        - Handles Indonesian wet/dry seasonal patterns  
        - Parameters: ['RR', 'TX', 'TN', 'TAVG', 'RH_AVG', 'FF_X', 'FF_AVG', 'DDD_X', 'SS']
        """
        logger.info("Calculating STL decomposition for BMKG...")
        
        params = ['RR', 'TX', 'TN', 'TAVG', 'RH_AVG', 'FF_X', 'FF_AVG', 'DDD_X', 'SS']
        decomposition_data = {
            "parameters_decomposed": [],
            "decomposition_data": {}
        }
        
        for param in params:
            if param not in df.columns:
                continue
        
            # Check minimum data requirement (+2 years)
            if len(df)< 730:
                logger.warning(f"{param}: Insufficient data for STL decomposition ({len(df)} days < 730)")
                continue
        
            try:
                series = df[param].dropna()
                if len(series) < 730:
                    logger.warning(f"{param}: Insufficient valid data after dropping NaN ({len(series)} < 730)")
                    continue
            
                logger.info(f"Decomposing {param} using STL")
                
                # STL decomposition with Indonesian seasonal patterns
                stl = STL(series, period=365, robust=True, seasonal=13)  # 13 for seasonal smoothing
                result = stl.fit()
                
                # Extract components
                trend = result.trend.dropna()
                seasonal = result.seasonal.dropna()
                residual = result.resid.dropna()
                
                # Calculate Indonesian wet/dry season statistics
                wet_months = [9, 10, 11, 12, 1, 2, 3]  # Sep-Mar
                dry_months = [4, 5, 6, 7, 8]            # Apr-Aug
                
                # Seasonal analysis by Indonesian patterns
                wet_seasonal = seasonal[seasonal.index.month.isin(wet_months)]
                dry_seasonal = seasonal[seasonal.index.month.isin(dry_months)]
                
                seasonal_stats = {
                    "wet_season_avg": float(wet_seasonal.mean()) if len(wet_seasonal) > 0 else 0.0,
                    "dry_season_avg": float(dry_seasonal.mean()) if len(dry_seasonal) > 0 else 0.0,
                    "seasonal_amplitude": float(seasonal.max() - seasonal.min()),
                    "wet_dry_difference": float(wet_seasonal.mean() - dry_seasonal.mean()) if len(wet_seasonal) > 0 and len(dry_seasonal) > 0 else 0.0
                }
                
                # Calculate decomposition statistics
                decomp_stats = {
                    "trend_strength": self._calculate_trend_strength(series, result.trend, result.resid),  # ← FIX: Pass residual
                    "seasonal_strength": self._calculate_seasonal_strength(series, seasonal, result.resid),  # Also use actual residual
                    "data_points": len(series),
                    "trend_direction": "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing",
                    "seasonal_stats": seasonal_stats
                }
                
                # Store components (sample for embedding)
                components_sample = {
                    "trend": trend.tail(90).to_dict(),      # Last 3 months
                    "seasonal": seasonal.tail(90).to_dict(), # Last 3 months  
                    "residual": residual.tail(90).to_dict()  # Last 3 months
                }
                
                decomposition_data["decomposition_data"][param] = {
                    "method": "STL_robust",
                    "period": 365,
                    "statistics": decomp_stats,
                    "components_sample": components_sample,
                    "seasonal_analysis": seasonal_stats
                }
                
                decomposition_data["parameters_decomposed"].append(param)
                
                logger.info(f"{param}: STL decomposition completed (trend_strength={decomp_stats['trend_strength']:.3f}, seasonal_strength={decomp_stats['seasonal_strength']:.3f})")

            except Exception as e:
                logger.error(f"Error decomposing {param}: {str(e)}")
                decomposition_data["decomposition_data"][param] = {
                    "error": str(e),
                    "status": "failed"
                }
        # Store in preprocessing report
        self.preprocessing_report["decomposition"] = decomposition_data
    
        success_count = len(decomposition_data["parameters_decomposed"])
        logger.info(f"STL decomposition completed: {success_count}/{len(params)} parameters successfully decomposed")
        
    def _calculate_trend_strength(self, series: pd.Series, trend: pd.Series, residual: pd.Series) -> float:
        """
        Calculate trend strength for STL decomposition
        
        FIXED: Use actual residual from STL result instead of (series - trend)
        
        Formula: trend_strength = 1 - var(residual) / var(detrended)
        where detrended = series - trend (contains seasonal + residual)
        """
        try:
            # Calculate detrended series (contains seasonal + residual components)
            detrended = (series - trend).dropna()
            
            # Use actual residual from STL decomposition (NOT series - trend)
            actual_residual = residual.dropna()
            
            # Ensure same length by aligning indices
            common_idx = detrended.index.intersection(actual_residual.index)
            if len(common_idx) < 10:  # Need minimum data points
                return 0.0
            
            detrended_aligned = detrended.loc[common_idx]
            residual_aligned = actual_residual.loc[common_idx]
            
            # Calculate variances
            var_detrended = np.var(detrended_aligned)
            var_residual = np.var(residual_aligned)
            
            if var_detrended > 0:
                trend_strength = max(0, 1 - (var_residual / var_detrended))
                
                # Debug logging for verification
                logger.debug(f"Trend strength calc: var_detrended={var_detrended:.4f}, var_residual={var_residual:.4f}, strength={trend_strength:.4f}")
                
                return trend_strength
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _calculate_seasonal_strength(
        self,
        series: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series
    )-> float:
        """Calculate seasonal strength for STL decomposition"""
        try:
            var_seasonal = np.var(seasonal.dropna())
            var_residual = np.var(residual.dropna())
            
            if (var_seasonal + var_residual) > 0:
                return var_seasonal / (var_seasonal + var_residual)
            return 0.0
        except:
            return 0.0
    
    def _calculate_trend_penalty(self, trend_preservation_pct: float) -> float:
        """
        Calculate penalty based on trend preservation loss
        
        Thresholds:
        - > 80%: No penalty
        - 70-80%: 5% penalty
        - 60-70%: 10% penalty
        - < 60%: 15% penalty
        """
        
        if trend_preservation_pct >= 80:
            return 0.0
        elif trend_preservation_pct >= 70:
            return 5.0
        elif trend_preservation_pct >= 60:
            return 10.0
        else:
            return 15.0
    
    
    def _calculate_hw_coverage(
        self,
        seasonality: Dict[str, Any],
        stationarity: Dict[str, Any],
        gaps: Dict[str, Any],
        extreme_values: Dict[str, Any],
        smoothing_quality: Dict[str, Any],
        missing_ratio: float,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate Holt-Winters model coverage with enhanced penalties
        
        FIXED: 
        1. Missing data penalty uses ORIGINAL ratio
        2. Added debug logging for penalty breakdown
        
        Penalties:
        1. GCV Smoothing Quality (0-15%)
        2. Trend Preservation (0-15%)
        3. Seasonality Loss (0-20%)
        4. Non-stationarity (0-15%)
        5. Large Gaps (0-12%, enhanced)
        6. Missing Data - ORIGINAL ratio (0-5%)
        7. Compound Issues (0-15%)
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality
        gcv_penalty = smoothing_quality.get("penalty", 0)
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation
        trend_value = smoothing_quality.get("trend_value", 100)
        trend_penalty = self._calculate_trend_penalty(trend_value)
        
        if trend_penalty > 0:
            base_coverage -= trend_penalty
            uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
        
        # PENALTY 3: Seasonality Loss (Critical for HW)
        seasonality_penalty = seasonality.get("penalty", 0)
        
        if seasonality_penalty > 0:
            base_coverage -= seasonality_penalty
            uncovered_reasons["seasonality_loss"] = round(seasonality_penalty, 2)
        
        # PENALTY 4: Non-stationarity
        stationarity_penalty = stationarity.get("penalty", 0)
        
        if stationarity_penalty > 0:
            base_coverage -= stationarity_penalty
            uncovered_reasons["non_stationarity"] = round(stationarity_penalty, 2)
        
        # PENALTY 5: Large Gaps (enhanced)
        gap_penalty = self._calculate_enhanced_gap_penalty(gaps, param)
        
        if gap_penalty > 0:
            base_coverage -= gap_penalty
            uncovered_reasons["large_gaps"] = round(gap_penalty, 2)
        
        # PENALTY 6: Missing Data - FIXED to use ORIGINAL ratio
        original_missing_penalty = self._calculate_original_missing_penalty(param)
        
        if original_missing_penalty > 0:
            base_coverage -= original_missing_penalty
            uncovered_reasons["original_missing_data"] = round(original_missing_penalty, 2)
        
        # PENALTY 7: Compound Penalty
        issue_count = 0
        
        if gap_penalty > 5:
            issue_count += 1
        if extreme_values.get("impact_percentage", 0) > 3:
            issue_count += 1
        if gcv_penalty > 10:
            issue_count += 1
        if trend_penalty >= 10:
            issue_count += 1
        if seasonality_penalty >= 15:
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(f"{param}: {issue_count} serious issues for HW")
        
        # Final coverage
        final_coverage = max(0, base_coverage)
        
        # DEBUG LOGGING
        logger.debug(f"HW {param} penalty breakdown:")
        logger.debug(f"  GCV: {gcv_penalty:.2f}%")
        logger.debug(f"  Trend: {trend_penalty:.2f}%")
        logger.debug(f"  Seasonality: {seasonality_penalty:.2f}%")
        logger.debug(f"  Stationarity: {stationarity_penalty:.2f}%")
        logger.debug(f"  Gap (enhanced): {gap_penalty:.2f}%")
        logger.debug(f"  Original Missing: {original_missing_penalty:.2f}%")
        logger.debug(f"  Compound: {uncovered_reasons.get('compound_issues', 0):.2f}%")
        logger.debug(f"  TOTAL PENALTY: {100 - final_coverage:.2f}%")
        logger.debug(f"  FINAL COVERAGE: {final_coverage:.2f}%")
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
        
    def _calculate_lstm_coverage(
        self,
        gaps: Dict[str, Any],
        extreme_values: Dict[str, Any],
        precipitation: Optional[Dict[str, Any]],
        smoothing_quality: Dict[str, Any],
        missing_ratio: float,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate LSTM model coverage with refined penalties
        
        FIXED BUGS:
        1. Missing data penalty now uses ORIGINAL missing ratio
        2. Added imputation quality penalty (method-based weighting)
        3. Added debug logging for penalty breakdown
        
        Penalties:
        1. GCV Smoothing Quality (0-15%)
        2. Trend Preservation (0-15%)
        3. Large Gaps (0-12%)
        4. Extreme Outliers (0-10%)
        5. Precipitation Extremes (0-10%, RR only)
        6. Missing Data - ORIGINAL ratio (0-5%)
        7. Imputation Quality (0-15%, NEW)
        8. Compound Issues (0-15%)
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality
        gcv_penalty = smoothing_quality.get("penalty", 0)
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation
        trend_value = smoothing_quality.get("trend_value", 100)
        trend_penalty = self._calculate_trend_penalty(trend_value)
        
        if trend_penalty > 0:
            base_coverage -= trend_penalty
            uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
        
        # PENALTY 3: Large Gaps (enhanced with unfilled gap detection)
        gap_penalty = self._calculate_enhanced_gap_penalty(gaps, param)
        
        if gap_penalty > 0:
            base_coverage -= gap_penalty
            uncovered_reasons["large_gaps"] = round(gap_penalty, 2)
        
        # PENALTY 4: Extreme Outliers
        outlier_penalty = extreme_values.get("impact_percentage", 0)
        
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 5: Precipitation Extremes (if applicable)
        precip_penalty = 0
        
        if precipitation:
            range_impact = precipitation.get("range_impact", 0)
            precip_penalty = range_impact * 10  # 0-10% penalty
            
            if precip_penalty > 0:
                base_coverage -= precip_penalty
                uncovered_reasons["precipitation_extremes"] = round(precip_penalty, 2)
        
        # PENALTY 6: Missing Data - FIXED to use ORIGINAL ratio
        original_missing_penalty = self._calculate_original_missing_penalty(param)
        
        if original_missing_penalty > 0:
            base_coverage -= original_missing_penalty
            uncovered_reasons["original_missing_data"] = round(original_missing_penalty, 2)
        
        # PENALTY 7: Imputation Quality - NEW
        imputation_quality_penalty = self._calculate_imputation_quality_penalty(param)
        
        if imputation_quality_penalty > 0:
            base_coverage -= imputation_quality_penalty
            uncovered_reasons["imputation_quality"] = round(imputation_quality_penalty, 2)
        
        # PENALTY 8: Compound Penalty
        issue_count = 0
        
        if gap_penalty > 5:
            issue_count += 1
        if outlier_penalty > 3:
            issue_count += 1
        if precip_penalty > 5:
            issue_count += 1
        if gcv_penalty > 10:
            issue_count += 1
        if trend_penalty >= 12:
            issue_count += 1
        if imputation_quality_penalty > 5:
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(f"{param}: {issue_count} serious issues for LSTM")
        
        # Final coverage
        final_coverage = max(0, base_coverage)
        
        # DEBUG LOGGING
        logger.debug(f"LSTM {param} penalty breakdown:")
        logger.debug(f"  GCV: {gcv_penalty:.2f}%")
        logger.debug(f"  Trend: {trend_penalty:.2f}%")
        logger.debug(f"  Gap (enhanced): {gap_penalty:.2f}%")
        logger.debug(f"  Outliers: {outlier_penalty:.2f}%")
        logger.debug(f"  Precipitation: {precip_penalty:.2f}%")
        logger.debug(f"  Original Missing: {original_missing_penalty:.2f}%")
        logger.debug(f"  Imputation Quality: {imputation_quality_penalty:.2f}%")
        logger.debug(f"  Compound: {uncovered_reasons.get('compound_issues', 0):.2f}%")
        logger.debug(f"  TOTAL PENALTY: {100 - final_coverage:.2f}%")
        logger.debug(f"  FINAL COVERAGE: {final_coverage:.2f}%")
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
    
    def _calculate_original_missing_penalty(self, param: str) -> float:
        """
        Calculate penalty based on ORIGINAL missing data ratio (before imputation)
        
        CRITICAL FIX: Uses original missing count from imputation report,
        NOT post-imputation missing ratio which is always 0.
        
        Args:
            param: Parameter name
        
        Returns:
            Penalty percentage (0-5%)
        """
        try:
            # Get original missing count from imputation report
            imputation_stats = self.preprocessing_report.get("imputation", {})
            
            if param not in imputation_stats:
                # No imputation performed = no missing data
                return 0.0
            
            param_stats = imputation_stats[param]
            original_missing = param_stats.get("before", 0)
            
            # Get total records
            if self.original_data is not None:
                total_records = len(self.original_data)
            else:
                # Fallback to current dataset length
                total_records = 7565  # Default BMKG dataset size
            
            # Calculate original missing ratio
            if total_records > 0:
                original_missing_ratio = original_missing / total_records
            else:
                original_missing_ratio = 0.0
            
            # Apply penalty (max 5% for missing data)
            # 0-5% missing → 0-1% penalty
            # 5-20% missing → 1-3% penalty
            # >20% missing → 3-5% penalty
            if original_missing_ratio <= 0.05:
                penalty = original_missing_ratio * 20  # 0-1%
            elif original_missing_ratio <= 0.20:
                penalty = 1.0 + (original_missing_ratio - 0.05) * 13.33  # 1-3%
            else:
                penalty = 3.0 + min((original_missing_ratio - 0.20) * 10, 2.0)  # 3-5%
            
            logger.debug(
                f"{param}: Original missing = {original_missing}/{total_records} "
                f"({original_missing_ratio*100:.1f}%) → penalty = {penalty:.2f}%"
            )
            
            return round(penalty, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating original missing penalty for {param}: {str(e)}")
            return 0.0
    
    def _calculate_imputation_quality_penalty(self, param: str) -> float:
        """
        Calculate penalty based on imputation method quality and confidence
        
        NEW FEATURE: Penalizes low-confidence imputation methods
        
        Quality tiers:
        - Calculation/Interpolate (physics-based, short gaps): Low penalty
        - Two-stage/Ratio (algorithmic): Medium penalty
        - Statistical fallback (circular mean, mode): High penalty
        
        Args:
            param: Parameter name
        
        Returns:
            Penalty percentage (0-15%)
        """
        try:
            # Get imputation stats
            imputation_stats = self.preprocessing_report.get("imputation", {})
            
            if param not in imputation_stats:
                # No imputation = no penalty
                return 0.0
            
            param_stats = imputation_stats[param]
            imputed_count = param_stats.get("imputed", 0)
            method = param_stats.get("method", "")
            
            # Get total records
            if self.original_data is not None:
                total_records = len(self.original_data)
            else:
                total_records = 7565
            
            if total_records == 0 or imputed_count == 0:
                return 0.0
            
            # Calculate imputation ratio
            imputation_ratio = imputed_count / total_records
            
            # Determine quality multiplier based on method
            # Lower multiplier = higher confidence = lower penalty
            method_lower = method.lower()
            
            if "calculation" in method_lower or "mathematical" in method_lower:
                # Physics-based calculations (TX from TAVG/TN)
                quality_multiplier = 0.2
                quality_tier = "high_confidence"
                
            elif "interpolate" in method_lower or "spline" in method_lower or "cubic" in method_lower:
                # Interpolation for short gaps
                quality_multiplier = 0.3
                quality_tier = "high_confidence"
                
            elif "ratio" in method_lower and "gap_aware" in method_lower:
                # FF_AVG from FF_X ratio (gap-aware)
                quality_multiplier = 0.4
                quality_tier = "medium_confidence"
                
            elif "two_stage" in method_lower or "two_tier" in method_lower or "binary" in method_lower:
                # RR two-stage imputation
                quality_multiplier = 0.5
                quality_tier = "medium_confidence"
                
            elif "dewpoint" in method_lower:
                # RH_AVG from dewpoint
                quality_multiplier = 0.4
                quality_tier = "medium_confidence"
                
            elif "circular" in method_lower or "circular_mean" in method_lower:
                # DDD_X circular mean (statistical fallback)
                quality_multiplier = 0.7
                quality_tier = "low_confidence"
                
            elif "mode" in method_lower:
                # DDD_CAR mode (categorical fallback)
                quality_multiplier = 0.8
                quality_tier = "low_confidence"
                
            else:
                # Unknown method - moderate penalty
                quality_multiplier = 0.5
                quality_tier = "medium_confidence"
            
            # Calculate penalty
            # penalty = imputation_ratio * quality_multiplier * max_penalty
            max_penalty = 15.0
            penalty = imputation_ratio * quality_multiplier * max_penalty
            
            # Cap at max_penalty
            penalty = min(penalty, max_penalty)
            
            logger.debug(
                f"{param}: Imputation quality penalty calculation:"
            )
            logger.debug(f"  Method: {method} → tier: {quality_tier} (multiplier: {quality_multiplier})")
            logger.debug(f"  Imputed: {imputed_count}/{total_records} ({imputation_ratio*100:.1f}%)")
            logger.debug(f"  Penalty: {penalty:.2f}%")
            
            return round(penalty, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating imputation quality penalty for {param}: {str(e)}")
            return 0.0
        
    def _calculate_enhanced_gap_penalty(self, gaps: Dict[str, Any], param: str) -> float:
        """
        Calculate enhanced gap penalty considering gap size and unfilled gaps
        
        ENHANCEMENT: Also checks for unfilled gaps from gap-aware imputation
        
        Args:
            gaps: Gap analysis results
            param: Parameter name
        
        Returns:
            Penalty percentage (0-12%)
        """
        try:
            penalty = 0.0
            
            # COMPONENT 1: Standard gap penalty from detection
            gap_impact = gaps.get("impact_percentage", 0)
            penalty += gap_impact
            
            # COMPONENT 2: Unfilled gaps penalty (gap-aware parameters only)
            gap_aware_params = ["FF_X", "FF_AVG", "DDD_X", "DDD_CAR"]
            
            if param in gap_aware_params:
                imputation_stats = self.preprocessing_report.get("imputation", {})
                
                if param in imputation_stats:
                    param_stats = imputation_stats[param]
                    
                    # Check for remaining missing values
                    remaining_missing = param_stats.get("remaining_missing", 0)
                    
                    if remaining_missing > 0:
                        # Get total records
                        if self.original_data is not None:
                            total_records = len(self.original_data)
                        else:
                            total_records = 7565
                        
                        # Calculate unfilled ratio
                        unfilled_ratio = remaining_missing / total_records
                        
                        # Unfilled gaps penalty: 0.5 weight (moderate impact)
                        # These are INTENTIONALLY unfilled (long gaps), not failures
                        unfilled_penalty = unfilled_ratio * 50  # Max ~5% for 10% unfilled
                        
                        penalty += unfilled_penalty
                        
                        logger.debug(
                            f"{param}: Unfilled gaps detected: {remaining_missing} "
                            f"({unfilled_ratio*100:.1f}%) → penalty: {unfilled_penalty:.2f}%"
                        )
            
            # Cap at maximum
            penalty = min(penalty, 12.0)
            
            return round(penalty, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced gap penalty for {param}: {str(e)}")
            return gaps.get("impact_percentage", 0)
    
    def _calculate_interpolation_confidence(self, df: pd.DataFrame, param: str) -> float:
        """
        Calculate confidence score for interpolated parameters
        
        Confidence based on:
        - Seasonal stability (lower variance = higher confidence)
        - Gap length (shorter gaps = higher confidence)
        
        Returns confidence score (0.0-1.0)
        """
        try:
            # Get parameter statistics
            param_series = df[param].dropna()
            
            if len(param_series) < 30:
                return 0.5  # Neutral confidence for insufficient data
            
            # Factor 1: Seasonal stability
            monthly_vars = []
            for month in range(1, 13):
                month_data = param_series[param_series.index.month == month]
                if len(month_data) >= 3:
                    monthly_vars.append(np.var(month_data))
            
            if monthly_vars:
                avg_monthly_var = np.mean(monthly_vars)
                overall_var = np.var(param_series)
                
                # If monthly variance is much lower than overall variance,
                # parameter is more predictable (higher confidence)
                if overall_var > 0:
                    stability_factor = 1 - (avg_monthly_var / overall_var)
                    stability_factor = max(0, min(1, stability_factor))
                else:
                    stability_factor = 0.8
            else:
                stability_factor = 0.6
            
            # Factor 2: Gap length penalty
            # Shorter interpolated gaps have higher confidence
            gaps_report = self.preprocessing_report.get("gaps", {})
            large_gaps = gaps_report.get("large_gaps", 0)
            total_gaps = gaps_report.get("total_gaps", 1)
            
            if total_gaps > 0:
                gap_factor = 1 - (large_gaps / total_gaps)
            else:
                gap_factor = 1.0
            
            # Combine factors
            confidence = (stability_factor * 0.7) + (gap_factor * 0.3)
            return max(0.3, min(1.0, confidence))  # Clip to reasonable range
            
        except Exception as e:
            logger.debug(f"Error calculating confidence for {param}: {str(e)}")
            return 0.6  # Default moderate confidence
        
    def _aggregate_uncovered_breakdown(
        self,
        per_parameter_results: Dict[str, Any],
        model_type: str
    )-> Dict[str, float]:
        """
        Aggregate uncovered breakdown across all parameters
        
        Averages penalty percentages across parameters to show
        overall impact of each issue type.
        
        Args:
            per_parameter_results: Coverage results per parameter
            model_type: "holt_winters" or "lstm"
        
        Returns:
            Dictionary of averaged penalty percentages by reason
        """
        
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
    def _determine_recommended_model(
        self,
        hw_coverage: float,
        lstm_coverage: float
    )-> str:
        """
        Determine recommended model based on coverage percentages
        
        Recommendation Logic:
        - Both > 80%: "both" (prefer both models)
        - HW > 80%, LSTM ≤ 80%: "holt_winters"
        - LSTM > 80%, HW ≤ 80%: "lstm"
        - HW > LSTM (both ≤ 80%): "holt_winters_with_caution"
        - LSTM > HW (both ≤ 80%): "lstm_with_caution"
        - Both < 60%: "none" (data quality issues)
        
        Args:
            hw_coverage: Holt-Winters coverage percentage
            lstm_coverage: LSTM coverage percentage
        
        Returns:
            Recommended model string
        """
        
        # Both models have good coverage
        if hw_coverage > 80 and lstm_coverage > 80:
            return "both"
        
        # Only one model has good coverage
        if hw_coverage > 80:
            return "holt_winters"
        
        if lstm_coverage > 80:
            return "lstm"
        
        # Both models have moderate coverage (60-80%)
        if hw_coverage >= 60 and lstm_coverage >= 60:
            if hw_coverage > lstm_coverage:
                return "holt_winters_with_caution"
            else:
                return "lstm_with_caution"
        
        # One model is better but not great
        if hw_coverage >= 60:
            return "holt_winters_with_caution"
        
        if lstm_coverage >= 60:
            return "lstm_with_caution"
        
        # Both models have poor coverage
        if hw_coverage > lstm_coverage:
            return "holt_winters_with_caution"
        elif lstm_coverage > hw_coverage:
            return "lstm_with_caution"
        else:
            return "none"  # Data quality too poor
        
    def _generate_quality_metrics(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
        """Generate quality metrics with forecasting readiness assessment"""
        logger.info("Generating quality metrics...")
        
        quality_metrics = {}
        
        # FORECASTING READINESS - New section
        forecasting_params = ['TAVG', 'RR', 'RH_AVG']
        forecasting_readiness = {}
        
        for param in forecasting_params:
            if param in processed_df.columns:
                coverage = (1 - processed_df[param].isna().sum() / len(processed_df)) * 100
                forecasting_readiness[param] = {
                    "coverage_pct": round(coverage, 2),
                    "status": "READY" if coverage >= 95 else "NEEDS_REVIEW"
                }
        
        # Supporting parameters status
        supporting_params = ['TX', 'TN', 'SS']
        supporting_status = {}
        
        for param in supporting_params:
            if param in processed_df.columns:
                coverage = (1 - processed_df[param].isna().sum() / len(processed_df)) * 100
                supporting_status[param] = {
                    "coverage_pct": round(coverage, 2),
                    "status": "GOOD" if coverage >= 85 else "ACCEPTABLE" if coverage >= 70 else "POOR"
                }
        
        # Metadata parameters status
        metadata_params = ['FF_X', 'FF_AVG', 'DDD_X', 'DDD_CAR']
        metadata_status = {}
        
        for param in metadata_params:
            if param in processed_df.columns:
                coverage = (1 - processed_df[param].isna().sum() / len(processed_df)) * 100
                metadata_status[param] = {
                    "coverage_pct": round(coverage, 2),
                    "note": "Metadata only - gaps expected" if coverage < 50 else "Better than expected coverage"
                }
        
        # Overall forecasting suitability
        forecasting_avg = sum(r["coverage_pct"] for r in forecasting_readiness.values()) / len(forecasting_readiness) if forecasting_readiness else 0
        supporting_avg = sum(r["coverage_pct"] for r in supporting_status.values()) / len(supporting_status) if supporting_status else 0
        
        overall_forecasting_score = (forecasting_avg * 0.7) + (supporting_avg * 0.3)
        
        # Calculate traditional metrics per parameter
        for param in self.params_to_process:
            if param not in processed_df.columns:
                continue
            
            if param == 'DDD_CAR':
                continue
            
            original_missing = original_df[param].isna().sum()
            processed_missing = processed_df[param].isna().sum()
            
            quality_metrics[param] = {
                "original_missing": int(original_missing),
                "processed_missing": int(processed_missing),
                "completeness": round(((len(processed_df) - processed_missing) / len(processed_df)) * 100, 2)
            }
        
        # Store comprehensive results
        self.preprocessing_report["quality_metrics"] = {
            "parameter_details": quality_metrics,
            
            # NEW: Forecasting readiness sections
            "forecasting_readiness": {
                "critical_parameters": forecasting_readiness,
                "supporting_parameters": supporting_status,
                "metadata_parameters": metadata_status,
                "overall_forecasting_score": round(overall_forecasting_score, 1),
                "forecasting_suitable": overall_forecasting_score >= 90
            }
        }
        
        logger.info(f"Quality metrics: {overall_forecasting_score:.1f}% forecasting readiness")
        critical_params_str = [f'{k}={v["coverage_pct"]:.1f}%' for k, v in forecasting_readiness.items()]
        logger.info(f"Critical parameters: {critical_params_str}")
        metadata_params_str = [f'{k}={v["coverage_pct"]:.1f}%' for k, v in metadata_status.items()]
        logger.info(f"Metadata parameters: {metadata_params_str}")
        
    def _save_preprocessing_report(
        self, 
        cleaned_collection_name: str
    )->  Dict[str, Any]:
        """
        Save unified preprocessing report to MongoDB preprocessing_report collection
        Compatible with NASA structure for unified frontend consumption
        """
        
        try:
            # Generate report metadata
            report_metadata = {
                "dataset_type": "bmkg",
                "collection_name": self.collection_name,
                "cleaned_collection_name": cleaned_collection_name,
                "created_at": datetime.now(),
                "processing_time": None,  # Will be calculated in preprocess()
                "total_parameters": len(self.params_to_process),
                "parameters_processed": list(self.preprocessing_report.get("imputation", {}).keys())
            }
            
            # Combine report with metadata
            unified_report = {
                **report_metadata,
                "preprocessing_details": self.preprocessing_report
            }
            
            # Sanitize for MongoDB
            sanitized_report = self._sanitize_for_mongodb(unified_report)
            
            # Save to preprocessing_report collection
            result = self.db["preprocessing_report"].insert_one(sanitized_report)
            
            logger.info(f"Saved unified preprocessing report with ID: {result.inserted_id}")
            
            return {
                "status": "success",
                "report_id": str(result.inserted_id),
                "collection": "preprocessing_report"
            }
        except Exception as e:
            logger.error(f"Error saving preprocessing report: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _sanitize_for_mongodb(self, obj):
        """
        Recursively convert numpy types to native Python types for MongoDB serialization
        (Same implementation as NASA preprocessing)
        """
        if isinstance(obj, dict):
            sanitized_dict = {}
            for key, value in obj.items():
                # Convert pandas Timestamp keys to strings
                if isinstance(key, pd.Timestamp):
                    str_key = key.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    str_key = str(key) if not isinstance(key, str) else key
                sanitized_dict[str_key] = self._sanitize_for_mongodb(value)
            return sanitized_dict
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
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()  # also convert timestamp keys
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
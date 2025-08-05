"""
Excel data processing pipeline for options flow uploads.
Handles validation, cleaning, and transformation of Excel data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, date
import uuid
import logging
from models.data_models import OptionsFlow

logger = logging.getLogger(__name__)


class ExcelDataProcessor:
    """
    Excel data processor for options flow file uploads.
    Provides validation, cleaning, and transformation capabilities.
    """

    def __init__(self):
        self.required_columns = [
            'symbol', 'buy_sell', 'call_put', 'strike', 'spot',
            'expiration_date', 'premium', 'volume', 'side'
        ]
        self.optional_columns = [
            'open_interest', 'price', 'color', 'set_count',
            'implied_volatility', 'dte', 'er_flag'
        ]
        self.column_mappings = {
            # Common alternative column names
            'ticker': 'symbol',
            'underlying': 'symbol',
            'stock': 'symbol',
            'buy/sell': 'buy_sell',
            'buysell': 'buy_sell',
            'call/put': 'call_put',
            'callput': 'call_put',
            'type': 'call_put',
            'strike_price': 'strike',
            'spot_price': 'spot',
            'underlying_price': 'spot',
            'exp_date': 'expiration_date',
            'expiry': 'expiration_date',
            'expiration': 'expiration_date',
            'vol': 'volume',
            'quantity': 'volume',
            'oi': 'open_interest',
            'iv': 'implied_volatility',
            'impl_vol': 'implied_volatility',
            'days_to_exp': 'dte',
            'days_to_expiry': 'dte',
            'er': 'er_flag',
            'earnings': 'er_flag'
        }

    def process_upload(self, file_path: str, progress_callback=None) -> Tuple[List[OptionsFlow], List[str]]:
        """Process Excel upload and return trades and errors."""
        try:
            # Read Excel file
            df = self._read_excel_file(file_path)
            if df is None:
                return [], ["Failed to read Excel file"]

            if progress_callback:
                progress_callback(10, "File loaded successfully")

            # Validate and map columns
            df, column_errors = self._validate_and_map_columns(df)
            if column_errors:
                return [], column_errors

            if progress_callback:
                progress_callback(20, "Columns validated")

            # Clean and transform data
            df, cleaning_errors = self._clean_and_transform(df)

            if progress_callback:
                progress_callback(50, "Data cleaned and transformed")

            # Convert to OptionsFlow objects
            trades, conversion_errors = self._convert_to_options_flow(df, progress_callback)

            if progress_callback:
                progress_callback(80, "Data converted to trade objects")

            # Detect duplicates
            duplicates = self._detect_duplicates(trades)
            duplicate_errors = []
            if duplicates:
                duplicate_errors = [f"Found {len(duplicates)} potential duplicate trade pairs"]

            if progress_callback:
                progress_callback(90, "Duplicate detection completed")

            # Combine all errors
            all_errors = cleaning_errors + conversion_errors + duplicate_errors

            if progress_callback:
                progress_callback(100, f"Processing complete: {len(trades)} trades processed")

            return trades, all_errors

        except Exception as e:
            logger.error(f"Failed to process Excel upload: {e}")
            return [], [f"Processing failed: {str(e)}"]

    def _read_excel_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read Excel file with error handling."""
        try:
            # Try different sheet names and engines
            for sheet_name in [0, 'Sheet1', 'Data', 'Options']:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                    if not df.empty:
                        logger.info(f"Successfully read Excel file with sheet: {sheet_name}")
                        return df
                except:
                    continue

            # If all attempts fail, try with default settings
            df = pd.read_excel(file_path)
            return df if not df.empty else None

        except Exception as e:
            logger.error(f"Failed to read Excel file: {e}")
            return None

    def _validate_and_map_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate required columns exist and map alternative names."""
        errors = []

        try:
            # Convert column names to lowercase for matching
            df.columns = df.columns.str.lower().str.strip()

            # Apply column mappings
            df = df.rename(columns=self.column_mappings)

            # Check for required columns
            missing_columns = []
            for col in self.required_columns:
                if col not in df.columns:
                    missing_columns.append(col)

            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                return df, errors

            # Add missing optional columns with default values
            for col in self.optional_columns:
                if col not in df.columns:
                    if col == 'er_flag':
                        df[col] = False
                    elif col in ['open_interest', 'volume', 'set_count', 'dte']:
                        df[col] = 0
                    elif col in ['price', 'implied_volatility']:
                        df[col] = 0.0
                    else:
                        df[col] = ''

            return df, errors

        except Exception as e:
            logger.error(f"Column validation failed: {e}")
            errors.append(f"Column validation error: {str(e)}")
            return df, errors

    def _clean_and_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and transform raw Excel data."""
        errors = []
        original_count = len(df)

        try:
            # Remove completely empty rows
            df = df.dropna(how='all')

            # Clean symbol column
            df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
            df = df[df['symbol'] != '']

            # Clean buy_sell column
            df['buy_sell'] = df['buy_sell'].astype(str).str.upper().str.strip()
            df['buy_sell'] = df['buy_sell'].replace({'B': 'BUY', 'S': 'SELL'})
            valid_buy_sell = df['buy_sell'].isin(['BUY', 'SELL'])
            if not valid_buy_sell.all():
                invalid_count = (~valid_buy_sell).sum()
                errors.append(f"Found {invalid_count} rows with invalid buy_sell values")
                df = df[valid_buy_sell]

            # Clean call_put column
            df['call_put'] = df['call_put'].astype(str).str.upper().str.strip()
            df['call_put'] = df['call_put'].replace({'C': 'CALL', 'P': 'PUT'})
            valid_call_put = df['call_put'].isin(['CALL', 'PUT'])
            if not valid_call_put.all():
                invalid_count = (~valid_call_put).sum()
                errors.append(f"Found {invalid_count} rows with invalid call_put values")
                df = df[valid_call_put]

            # Clean numeric columns
            numeric_columns = ['strike', 'spot', 'premium', 'volume', 'open_interest', 'price', 'implied_volatility', 'dte', 'set_count']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with 0 for optional columns
                    if col not in ['strike', 'spot', 'premium']:
                        df[col] = df[col].fillna(0)

            # Remove rows with missing critical numeric data
            critical_numeric = ['strike', 'spot', 'premium']
            for col in critical_numeric:
                before_count = len(df)
                df = df.dropna(subset=[col])
                after_count = len(df)
                if before_count != after_count:
                    errors.append(f"Removed {before_count - after_count} rows with missing {col}")

            # Clean date column
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
            before_count = len(df)
            df = df.dropna(subset=['expiration_date'])
            after_count = len(df)
            if before_count != after_count:
                errors.append(f"Removed {before_count - after_count} rows with invalid expiration dates")

            # Clean side column
            if 'side' in df.columns:
                df['side'] = df['side'].astype(str).str.strip()
                df['side'] = df['side'].replace({'nan': '', 'NaN': '', 'None': ''})

            # Clean color column
            if 'color' in df.columns:
                df['color'] = df['color'].astype(str).str.strip()
                df['color'] = df['color'].replace({'nan': '', 'NaN': '', 'None': ''})

            # Handle ER flag
            if 'er_flag' in df.columns:
                df['er_flag'] = df['er_flag'].astype(str).str.upper().str.strip()
                df['er_flag'] = df['er_flag'].replace({'T': True, 'TRUE': True, 'F': False, 'FALSE': False, '1': True, '0': False})
                df['er_flag'] = df['er_flag'].astype(bool)

            # Calculate DTE if not provided
            if 'dte' not in df.columns or df['dte'].isna().all():
                df['dte'] = (df['expiration_date'] - pd.Timestamp.now()).dt.days
                df['dte'] = df['dte'].fillna(0).astype(int)

            final_count = len(df)
            if final_count != original_count:
                errors.append(f"Data cleaning removed {original_count - final_count} rows. {final_count} rows remaining.")

            return df, errors

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            errors.append(f"Data cleaning error: {str(e)}")
            return df, errors

    def _convert_to_options_flow(self, df: pd.DataFrame, progress_callback=None) -> Tuple[List[OptionsFlow], List[str]]:
        """Convert DataFrame to OptionsFlow objects."""
        trades = []
        errors = []

        try:
            total_rows = len(df)

            for idx, row in df.iterrows():
                try:
                    # Generate unique ID for each trade
                    trade_id = str(uuid.uuid4())

                    # Create OptionsFlow object
                    trade = OptionsFlow(
                        id=trade_id,
                        created_datetime=datetime.now(),
                        symbol=row['symbol'],
                        buy_sell=row['buy_sell'],
                        call_put=row['call_put'],
                        strike=float(row['strike']),
                        spot=float(row['spot']),
                        expiration_date=row['expiration_date'].date(),
                        premium=float(row['premium']),
                        volume=int(row.get('volume', 0)),
                        open_interest=int(row.get('open_interest', 0)),
                        price=float(row.get('price', 0)),
                        side=str(row.get('side', '')),
                        color=str(row.get('color', '')),
                        set_count=int(row.get('set_count', 0)),
                        implied_volatility=float(row.get('implied_volatility', 0)),
                        dte=int(row.get('dte', 0)),
                        er_flag=bool(row.get('er_flag', False)),
                        trade_value=float(row['premium']) * int(row.get('volume', 1))
                    )

                    trades.append(trade)

                    # Update progress
                    if progress_callback and idx % 10 == 0:
                        progress = 50 + int((idx / total_rows) * 30)  # 50-80% range
                        progress_callback(progress, f"Processing row {idx + 1}/{total_rows}")

                except Exception as e:
                    errors.append(f"Row {idx + 1}: {str(e)}")
                    continue

            logger.info(f"Converted {len(trades)} trades from {total_rows} rows")
            return trades, errors

        except Exception as e:
            logger.error(f"Conversion to OptionsFlow failed: {e}")
            errors.append(f"Conversion error: {str(e)}")
            return trades, errors

    def _detect_duplicates(self, trades: List[OptionsFlow]) -> List[Tuple[int, int]]:
        """Detect duplicate trades and return index pairs."""
        duplicates = []

        try:
            for i, trade1 in enumerate(trades):
                for j, trade2 in enumerate(trades[i+1:], i+1):
                    if self._trades_are_duplicates(trade1, trade2):
                        duplicates.append((i, j))

            logger.info(f"Found {len(duplicates)} potential duplicate pairs")
            return duplicates

        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return []

    def _trades_are_duplicates(self, trade1: OptionsFlow, trade2: OptionsFlow) -> bool:
        """Check if two trades are potential duplicates."""
        try:
            # Check key fields for exact match
            key_fields_match = (
                trade1.symbol == trade2.symbol and
                trade1.buy_sell == trade2.buy_sell and
                trade1.call_put == trade2.call_put and
                abs(trade1.strike - trade2.strike) < 0.01 and
                trade1.expiration_date == trade2.expiration_date and
                abs(trade1.premium - trade2.premium) < 0.01 and
                trade1.volume == trade2.volume
            )

            return key_fields_match

        except Exception as e:
            logger.error(f"Duplicate comparison failed: {e}")
            return False

    def validate_file_format(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate file format before processing."""
        errors = []

        try:
            # Check file extension
            if not file_path.lower().endswith(('.xlsx', '.xls')):
                errors.append("File must be an Excel file (.xlsx or .xls)")
                return False, errors

            # Try to read the file
            df = self._read_excel_file(file_path)
            if df is None:
                errors.append("Unable to read Excel file")
                return False, errors

            # Check if file has data
            if df.empty:
                errors.append("Excel file is empty")
                return False, errors

            # Check for minimum required columns (case-insensitive)
            df_columns_lower = [col.lower().strip() for col in df.columns]
            required_found = 0

            for req_col in self.required_columns:
                # Check direct match or mapped alternatives
                if req_col in df_columns_lower:
                    required_found += 1
                else:
                    # Check if any mapping exists
                    for alt_name, mapped_name in self.column_mappings.items():
                        if mapped_name == req_col and alt_name in df_columns_lower:
                            required_found += 1
                            break

            if required_found < len(self.required_columns):
                missing_ratio = (len(self.required_columns) - required_found) / len(self.required_columns)
                if missing_ratio > 0.5:  # More than 50% missing
                    errors.append(f"File appears to be missing too many required columns")
                    return False, errors
                else:
                    errors.append(f"Some required columns may be missing - will attempt processing")

            return True, errors

        except Exception as e:
            logger.error(f"File validation failed: {e}")
            errors.append(f"File validation error: {str(e)}")
            return False, errors

    def get_file_preview(self, file_path: str, num_rows: int = 5) -> Tuple[pd.DataFrame, List[str]]:
        """Get a preview of the Excel file data."""
        try:
            df = self._read_excel_file(file_path)
            if df is None:
                return pd.DataFrame(), ["Unable to read file"]

            # Return first few rows for preview
            preview_df = df.head(num_rows)

            info = [
                f"File contains {len(df)} rows and {len(df.columns)} columns",
                f"Columns: {', '.join(df.columns.tolist())}"
            ]

            return preview_df, info

        except Exception as e:
            logger.error(f"File preview failed: {e}")
            return pd.DataFrame(), [f"Preview error: {str(e)}"]

    def handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing or invalid data with specific error messages."""
        errors = []

        try:
            # Report missing data statistics
            missing_stats = df.isnull().sum()
            for col, missing_count in missing_stats.items():
                if missing_count > 0:
                    percentage = (missing_count / len(df)) * 100
                    errors.append(f"Column '{col}': {missing_count} missing values ({percentage:.1f}%)")

            # Handle missing data based on column importance
            for col in df.columns:
                if col in self.required_columns:
                    # For required columns, remove rows with missing data
                    before_count = len(df)
                    df = df.dropna(subset=[col])
                    after_count = len(df)
                    if before_count != after_count:
                        errors.append(f"Removed {before_count - after_count} rows due to missing {col}")
                else:
                    # For optional columns, fill with appropriate defaults
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna('')

            return df, errors

        except Exception as e:
            logger.error(f"Missing data handling failed: {e}")
            errors.append(f"Missing data handling error: {str(e)}")
            return df, errors

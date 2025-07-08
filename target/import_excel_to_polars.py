#!/usr/bin/env python3
"""
Import Excel file to Polars DataFrame for data analysis.
This script loads the Excel file and provides basic data exploration capabilities.
"""

import polars as pl
import pandas as pd
import sys
import os
from pathlib import Path

def import_excel_to_polars(excel_file: str = "Xylem Innovation Research Interview - raw coding only.xlsx") -> pl.DataFrame:
    """
    Import Excel file to Polars DataFrame.
    
    Args:
        excel_file: Path to the Excel file (default: the file in this directory)
    
    Returns:
        pl.DataFrame: The imported data as a Polars DataFrame
    """
    
    # Get the full path to the Excel file
    script_dir = Path(__file__).parent
    excel_path = script_dir / excel_file
    
    print(f"📊 Importing Excel file: {excel_path}")
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    try:
        # Try reading Excel file directly with Polars, skipping initial rows and setting header
        try:
            df = pl.read_excel(excel_path, read_options={"skip_rows": 2})
            
            # Clean up column names
            new_columns = [
                'informant', 'role_type', 'time', 'passage', 
                'provisional_logic', 'cultural_schema', 'dominant_logic', 'final_coding'
            ]
            df.columns = new_columns
            
            # Remove the extra header rows that are now part of the data
            df = df.slice(2)

            print(f"✅ Successfully imported with Polars and applied custom headers!")
        except Exception as e1:
            print(f"⚠️  Polars direct import failed: {e1}")
            print(f"🔄 Trying pandas fallback...")
            
            # Fallback to pandas then convert to Polars
            pandas_df = pd.read_excel(excel_path)
            df = pl.from_pandas(pandas_df)
            print(f"✅ Successfully imported via pandas -> polars conversion!")
        
        print(f"   📋 Shape: {df.shape} (rows x columns)")
        print(f"   📝 Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error importing Excel file: {e}")
        print(f"💡 Tip: Make sure the Excel file is not open in another program")
        raise

def explore_dataframe(df: pl.DataFrame) -> None:
    """
    Provide basic exploration of the DataFrame.
    
    Args:
        df: The Polars DataFrame to explore
    """
    
    print(f"\n🔍 Data Exploration:")
    print("=" * 50)
    
    # Basic info
    print(f"📊 DataFrame Info:")
    print(f"   • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   • Memory usage: {df.estimated_size('mb'):.2f} MB")
    
    # Column information
    print(f"\n📋 Column Information:")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        null_count = df.select(pl.col(col).null_count()).item()
        print(f"   {i:2d}. {col:<30} | {str(dtype):<15} | {null_count} nulls")
    
    # Show first few rows
    print(f"\n👀 First 5 rows:")
    print(df.head())
    
    # Show data types and basic stats for numeric columns
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
    
    if numeric_cols:
        print(f"\n📈 Numeric Column Statistics:")
        print(df.select(numeric_cols).describe())
    
    # Show unique values for categorical-like columns (if they have few unique values)
    print(f"\n🏷️ Categorical Analysis:")
    for col in df.columns:
        if df[col].dtype == pl.Utf8:  # String columns
            unique_count = df.select(pl.col(col).n_unique()).item()
            if unique_count <= 20:  # Show unique values if <= 20
                unique_values = df.select(pl.col(col).unique()).to_series().to_list()
                print(f"   • {col}: {unique_count} unique values")
                print(f"     Values: {unique_values}")
            else:
                print(f"   • {col}: {unique_count} unique values (too many to display)")

def save_as_parquet(df: pl.DataFrame, filename: str = "imported_data.parquet") -> None:
    """
    Save the DataFrame as a Parquet file for efficient storage.
    
    Args:
        df: The Polars DataFrame to save
        filename: Output filename (default: imported_data.parquet)
    """
    
    script_dir = Path(__file__).parent
    output_path = script_dir / filename
    
    print(f"\n💾 Saving DataFrame as Parquet file: {output_path}")
    
    try:
        df.write_parquet(output_path)
        print(f"✅ Successfully saved as {filename}")
        print(f"   📁 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving Parquet file: {e}")
        raise

def save_as_csv(df: pl.DataFrame, filename: str = "imported_data.csv") -> None:
    """
    Save the DataFrame as a CSV file.
    
    Args:
        df: The Polars DataFrame to save
        filename: Output filename (default: imported_data.csv)
    """
    
    script_dir = Path(__file__).parent
    output_path = script_dir / filename
    
    print(f"\n📄 Saving DataFrame as CSV file: {output_path}")
    
    try:
        df.write_csv(output_path)
        print(f"✅ Successfully saved as {filename}")
        print(f"   📁 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving CSV file: {e}")
        raise

def filter_example(df: pl.DataFrame) -> None:
    """
    Show example filtering operations with Polars.
    
    Args:
        df: The Polars DataFrame to demonstrate filtering on
    """
    
    print(f"\n🔍 Example Filtering Operations:")
    print("=" * 40)
    
    # Example 1: Show rows with non-null values in first text column
    text_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    
    if text_cols:
        first_text_col = text_cols[0]
        non_null_df = df.filter(pl.col(first_text_col).is_not_null())
        print(f"📝 Rows with non-null '{first_text_col}': {non_null_df.shape[0]} / {df.shape[0]}")
        
        # Show sample of non-null values
        if non_null_df.shape[0] > 0:
            print(f"   Sample values:")
            sample_values = non_null_df.select(pl.col(first_text_col)).head(3).to_series().to_list()
            for i, value in enumerate(sample_values, 1):
                print(f"   {i}. {str(value)[:100]}...")
    
    # Example 2: Count non-null values per column
    print(f"\n📊 Non-null counts per column:")
    for col in df.columns:
        non_null_count = df.select(pl.col(col).count()).item()
        total_count = df.shape[0]
        percentage = (non_null_count / total_count) * 100 if total_count > 0 else 0
        print(f"   • {col:<30}: {non_null_count:>4} / {total_count} ({percentage:5.1f}%)")

def main():
    """Main function to import and explore the Excel data."""
    
    print("🚀 Excel to Polars DataFrame Importer")
    print("=" * 50)
    
    try:
        # Import the Excel file
        df = import_excel_to_polars()
        
        # Explore the data
        explore_dataframe(df)
        
        # Show filtering examples
        filter_example(df)
        
        # Save in different formats
        save_as_parquet(df)
        save_as_csv(df)
        
        print(f"\n🎉 Import and exploration complete!")
        print(f"💡 The DataFrame is available as 'df' variable")
        print(f"📚 Use df.head(), df.describe(), df.columns, etc. for further exploration")
        
        # Return the dataframe for interactive use
        return df
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        return None

# Global variable to store the DataFrame for interactive use
df = None

if __name__ == "__main__":
    # Run the import and store the result globally
    df = main()
    
    # Additional interactive suggestions
    if df is not None:
        print(f"\n🔧 Interactive Usage:")
        print(f"   The DataFrame is stored in the global variable 'df'")
        print(f"   Try these commands:")
        print(f"   • df.shape                    # Shape of the data")
        print(f"   • df.columns                  # Column names")
        print(f"   • df.head(10)                 # First 10 rows")
        print(f"   • df.describe()               # Statistical summary")
        print(f"   • df.select('column_name')    # Select specific column")
        print(f"   • df.filter(condition)        # Filter rows")
        print(f"   • df.group_by('col').agg(pl.count())  # Group and aggregate")

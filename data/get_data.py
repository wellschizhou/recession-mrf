# data/get_data.py
import pandas as pd
import fredapi
import os
import json
from dotenv import load_dotenv

def get_fred_api_key():
    """
    Get FRED API key from environment variable.
    Provides helpful error message if not found.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        raise ValueError(
            "\n❌ FRED API key not found!\n\n"
            "Please follow these steps:\n"
            "1. Get your free API key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "2. Copy .env.example to .env: cp .env.example .env\n"
            "3. Edit .env and add your key: FRED_API_KEY=your_key_here\n"
        )
    
    print("✅ FRED API key loaded successfully")
    return api_key

def get_recession_predictors():
    """Get recession predictor variables for the linear equation (X_t)"""
    # Get API key from environment
    fred_api_key = get_fred_api_key()
    fred = fredapi.Fred(api_key=fred_api_key)
    
    series = {
        'USREC': 'recession_indicator',
        'INDPRO': 'industrial_production',
        'PAYEMS': 'nonfarm_payrolls', 
        'UNRATE': 'unemployment_rate',
        'DGS10': 'treasury_10y',
        'DGS3MO': 'treasury_3m',
        'BAAFFM': 'baa_corporate_yield',
        'AAAFFM': 'aaa_corporate_yield'
    }
    
    print("📈 Fetching recession predictor variables...")
    data = {}
    for fred_id, name in series.items():
        try:
            data[name] = fred.get_series(fred_id, start='1963-01-01')
            print(f"  ✅ {fred_id} downloaded")
        except Exception as e:
            print(f"  ❌ Failed to download {fred_id}: {e}")
    
    df = pd.DataFrame(data)
    
    # Create derived features
    df['yield_spread'] = df['treasury_10y'] - df['treasury_3m'] 
    df['credit_spread'] = df['baa_corporate_yield'] - df['aaa_corporate_yield']
    
    return df

def get_fredmd_data():
    """Get FRED-MD dataset for state variables (S_t)"""
    print("📊 Downloading FRED-MD dataset...")
    
    url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
    
    try:
        fredmd = pd.read_csv(url, skiprows = [1])  # Skip transformation codes
        fredmd['sasdate'] = pd.to_datetime(fredmd['sasdate'])
        fredmd.set_index('sasdate', inplace=True)
        
        print(f"  ✅ FRED-MD downloaded: {fredmd.shape[0]} observations, {fredmd.shape[1]} variables")
        return fredmd
        
    except Exception as e:
        print(f"  ❌ Failed to download FRED-MD: {e}")
        return None

def create_mrf_dataset(recession_predictors, fredmd_data):
    """Create unified dataset for MRF package"""
    print("🔧 Creating unified MRF dataset...")
    
    # Ensure datetime indices
    if not isinstance(recession_predictors.index, pd.DatetimeIndex):
        recession_predictors.index = pd.to_datetime(recession_predictors.index)
    
    # Find common date range
    start_date = max(recession_predictors.index.min(), fredmd_data.index.min())
    end_date = min(recession_predictors.index.max(), fredmd_data.index.max())
    
    print(f"  📅 Common date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    # Align on monthly frequency
    predictors_monthly = recession_predictors.resample('M').last()
    fredmd_monthly = fredmd_data.resample('M').last()
    
    # Trim to common dates
    predictors_aligned = predictors_monthly.loc[start_date:end_date]
    fredmd_aligned = fredmd_monthly.loc[start_date:end_date]
    
    # Create unified DataFrame: [target, X variables, S variables]
    target_col = predictors_aligned[['recession_indicator']]
    x_variables = predictors_aligned[['unemployment_rate', 'yield_spread', 'credit_spread', 'industrial_production']]
    
    # Remove overlapping columns from FRED-MD
    fredmd_clean = fredmd_aligned.drop(columns=['USREC'], errors='ignore')
    
    # Combine all data
    mrf_data = pd.concat([target_col, x_variables, fredmd_clean], axis=1)
    mrf_data = mrf_data.dropna()
    
    print(f"  ✅ Unified dataset created: {mrf_data.shape}")
    
    # Create position mappings for MRF
    positions = {
        'y_pos': 0,  # Target variable position
        'x_pos': list(range(1, len(x_variables.columns) + 1)),  # X variables
        'S_pos': list(range(len(x_variables.columns) + 1, mrf_data.shape[1]))  # S variables
    }
    
    return mrf_data, positions

if __name__ == "__main__":
    print("🚀 Starting data collection for Recession MRF\n")
    
    try:
        # Download both datasets
        recession_predictors = get_recession_predictors()
        fredmd_data = get_fredmd_data()
        
        if recession_predictors is not None and fredmd_data is not None:
            # Create unified MRF dataset
            mrf_data, positions = create_mrf_dataset(recession_predictors, fredmd_data)
            
            # Save files
            mrf_data.to_csv('data/mrf_dataset.csv')
            
            with open('data/mrf_positions.json', 'w') as f:
                json.dump(positions, f, indent=2)
            
            print(f"\n🎉 Success! Files saved:")
            print(f"  📁 data/mrf_dataset.csv ({mrf_data.shape})")
            print(f"  📁 data/mrf_positions.json")
            
            print(f"\n📋 Dataset preview:")
            print(mrf_data.head())
            
        else:
            print("❌ Failed to download datasets")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nDouble-check your .env file setup!")
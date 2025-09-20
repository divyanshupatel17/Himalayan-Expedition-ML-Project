"""
Simple data loader for Himalayan Expedition dataset
"""
import pandas as pd
import os

def load_data():
    """
    Load the Himalayan expedition dataset from local files.
    
    Returns:
        tuple: (expeditions, members, peaks) DataFrames
    """
    try:
        # Get the directory where this script is located
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Print current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {current_script_dir}")
        
        # Try to find the data directory relative to the script location
        # The data directory should be at the same level as src directory
        possible_data_dirs = [
            os.path.join(current_script_dir, '..', 'data'),  # src/../data
            os.path.join(current_script_dir, '..', '..', 'data'),  # src/../../data
            os.path.join(os.getcwd(), 'data'),  # Current directory/data
            os.path.join(os.getcwd(), 'myProject', 'himalayan_ml_minimal', 'data'),  # Full path from workspace
            os.path.join(os.getcwd(), 'himalayan_ml_minimal', 'data'),  # Relative path
        ]
        
        # Try to find the data directory
        data_dir = None
        for path in possible_data_dirs:
            normalized_path = os.path.normpath(path)
            print(f"Checking path: {normalized_path}")
            if os.path.exists(normalized_path):
                files = ['expeditions.csv', 'members.csv', 'peaks.csv']
                files_found = all(os.path.exists(os.path.join(normalized_path, f)) for f in files)
                if files_found:
                    data_dir = normalized_path
                    print(f"Found data directory: {data_dir}")
                    break
        
        if data_dir is None:
            print("Data files not found. Please download the Himalayan expedition dataset from Kaggle:")
            print("https://www.kaggle.com/datasets/majunbajun/himalayan-climbing-expeditions")
            print("And place the following files in the data/ directory:")
            print("- expeditions.csv")
            print("- members.csv") 
            print("- peaks.csv")
            print("\nTried looking in these locations:")
            for path in possible_data_dirs:
                print(f"  - {os.path.normpath(path)}")
            return None, None, None
        
        # Define paths to data files
        expeditions_path = os.path.join(data_dir, 'expeditions.csv')
        members_path = os.path.join(data_dir, 'members.csv')
        peaks_path = os.path.join(data_dir, 'peaks.csv')
        
        # Load data
        print("Loading expedition data...")
        expeditions = pd.read_csv(expeditions_path, low_memory=False)
        print(f"Loaded expeditions: {expeditions.shape}")
        
        print("Loading member data...")
        members = pd.read_csv(members_path, low_memory=False)
        print(f"Loaded members: {members.shape}")
        
        print("Loading peak data...")
        peaks = pd.read_csv(peaks_path, low_memory=False)
        print(f"Loaded peaks: {peaks.shape}")
        
        return expeditions, members, peaks
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_master_dataset(expeditions, members, peaks):
    """
    Create a master dataset by joining all three DataFrames.
    
    Args:
        expeditions (pd.DataFrame): Expeditions data
        members (pd.DataFrame): Members data
        peaks (pd.DataFrame): Peaks data
        
    Returns:
        pd.DataFrame: Master dataset for modeling
    """
    if expeditions is None or members is None or peaks is None:
        return None
    
    try:
        print("Creating master dataset...")
        
        # Join expeditions with members
        exp_mem = pd.merge(expeditions, members, on=['expid', 'peakid'], how='inner')
        print(f"Expeditions-Members join: {exp_mem.shape}")
        
        # Join with peaks
        master = pd.merge(exp_mem, peaks, on='peakid', how='inner')
        print(f"Final master dataset: {master.shape}")
        
        return master
    except Exception as e:
        print(f"Error creating master dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    print("Data Loader for Himalayan Expedition Project")
    print("Use load_data() to load the dataset")
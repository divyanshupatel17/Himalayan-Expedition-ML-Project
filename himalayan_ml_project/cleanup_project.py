"""
Project Cleanup Script
Removes temporary and unnecessary files while preserving essential components
"""
import os
import shutil

def cleanup_project():
    """Clean up temporary and unnecessary files"""
    
    print("🧹 HIMALAYAN ML PROJECT CLEANUP")
    print("=" * 40)
    
    # Files and directories to remove
    cleanup_items = [
        # Cache directories
        '__pycache__',
        'src/__pycache__',
        
        # CatBoost training files
        'catboost_info',
        
        # Duplicate dataset (keep data/ folder)
        '../dataset',
        
        # Assignment files (move to archive)
        '../DA1',
        
        # Documentation files (optional)
        '../Additional.txt',
        '../dataset_description.md', 
        '../himalayan_expedition_ml_plan.md',
        'MODEL_IMPROVEMENTS_SUMMARY.txt'
    ]
    
    removed_count = 0
    
    for item in cleanup_items:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"✅ Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"✅ Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Could not remove {item}: {e}")
        else:
            print(f"⚠️  Not found: {item}")
    
    print(f"\n🎉 Cleanup complete! Removed {removed_count} items.")
    
    # Show what's left
    print("\n📁 REMAINING ESSENTIAL FILES:")
    essential_items = [
        'app.py',
        'demo_model_loader.py', 
        'requirements.txt',
        'README.md',
        'test_complete_pipeline.py',
        'src/',
        'data/',
        'notebooks/models/',
        'notebooks/saved_models/',
        'notebooks/import_helper.py'
    ]
    
    for item in essential_items:
        if os.path.exists(item):
            print(f"✅ {item}")
        else:
            print(f"❌ MISSING: {item}")

if __name__ == "__main__":
    cleanup_project()
"""
Heart Disease Prediction Project - Simple Setup Script
Creates empty folder structure only
"""

import os
import shutil

def create_project_structure():
    """Create all necessary folders for the project"""
    
    print("="*70)
    print("CREATING PROJECT STRUCTURE...")
    print("="*70)
    
    # Define folder structure
    folders = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'reports/figures',
        'models'
    ]
    
    # Create folders
    print("\n📁 Creating folders...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"   ✓ {folder}/")
    
    # Move dataset to data/raw
    print("\n📊 Copying dataset...")
    source_file = 'heart+disease/processed.cleveland.data'
    destination = 'data/raw/processed.cleveland.data'
    
    if os.path.exists(source_file):
        shutil.copy(source_file, destination)
        print(f"   ✓ Dataset copied to data/raw/")
    else:
        print(f"   ⚠ Dataset not found at: {source_file}")
        print(f"   Please copy manually to: {destination}")
    
    # Create empty .gitignore
    print("\n📝 Creating files...")
    with open('.gitignore', 'w') as f:
        f.write("# Will be populated later\n")
    print("   ✓ .gitignore")
    
    # Create empty README
    with open('README.md', 'w') as f:
        f.write("# Heart Disease Prediction\n\n")
        f.write("Project structure created. Documentation will be added later.\n")
    print("   ✓ README.md")
    
    # Create empty requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("# Dependencies will be added as needed\n")
    print("   ✓ requirements.txt")
    
    # Create empty notebooks
    print("\n📓 Creating notebooks...")
    notebooks = [
        'notebooks/01_data_loading.ipynb',
        'notebooks/02_eda.ipynb',
        'notebooks/03_data_cleaning.ipynb',
        'notebooks/04_visualizations.ipynb',
        'notebooks/05_ml_models.ipynb'
    ]
    
    for notebook in notebooks:
        with open(notebook, 'w') as f:
            f.write("")  # Empty file
        print(f"   ✓ {notebook}")
    
    # Create empty Python files
    print("\n🐍 Creating Python files...")
    py_files = [
        'src/model.py',
        'src/predict.py',
        'src/app.py'
    ]
    
    for py_file in py_files:
        with open(py_file, 'w') as f:
            f.write("")  # Empty file
        print(f"   ✓ {py_file}")
    
    # Create .gitkeep for empty folders
    gitkeep_folders = ['data/processed', 'reports/figures', 'models']
    for folder in gitkeep_folders:
        with open(f'{folder}/.gitkeep', 'w') as f:
            f.write('')
    
    print("\n" + "="*70)
    print("✅ STRUCTURE CREATED!")
    print("="*70)
    
    print("\nProject structure:")
    print("""
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── reports/
│   └── figures/
├── models/
├── .gitignore
├── README.md
└── requirements.txt
    """)
    
    print("Ready to start! 🚀\n")

if __name__ == "__main__":
    create_project_structure()
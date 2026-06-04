# Installation Guide

## Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd skills-demand-forecasting-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Setup Configuration
```bash
cp config/development.yaml config/local.yaml
# Edit local.yaml with your settings
```

### 5. Initialize Database (if using)
```bash
python scripts/setup_environment.py
```

### 6. Download Sample Data
```bash
python scripts/download_data.py --sample
```

## Verification
```bash
python -c "import src; print('Installation successful!')"
```

## Troubleshooting
- Ensure Python version compatibility
- Check virtual environment activation
- Verify all dependencies are installed

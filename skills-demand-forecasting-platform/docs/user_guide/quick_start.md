# Quick Start Guide

## Get Started in 5 Minutes

### 1. Load Sample Data
```python
from src.data_processing.data_ingestion import DataIngestionPipeline

pipeline = DataIngestionPipeline()
data = pipeline.load_sample_data()
```

### 2. Extract Skills
```python
from src.nlp.skill_extractor import SkillExtractor

extractor = SkillExtractor()
skills = extractor.extract_skills("Python developer with React experience")
print(skills)  # ['Python', 'React']
```

### 3. Predict Demand
```python
from src.ml_models.forecasting.demand_forecaster import SkillDemandForecaster

forecaster = SkillDemandForecaster()
prediction = forecaster.predict("Python", horizon=6)
```

### 4. Launch Dashboard
```bash
python dashboard/app.py
# Visit http://localhost:8501
```

### 5. Access API
```bash
python src/api/main.py
# API available at http://localhost:8000
```

## Next Steps
- Explore notebooks/ for detailed examples
- Read the Architecture documentation
- Check out the API documentation

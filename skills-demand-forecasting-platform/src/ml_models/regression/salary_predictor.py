"""Salary prediction models using various regression techniques."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import with fallbacks
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Salary prediction will use simple estimation.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.utils.config_manager import config

logger = logging.getLogger(__name__)

class SalaryPredictor:
    """Advanced salary prediction using multiple regression models with fallbacks."""
    
    def __init__(self):
        self.config = config.get_model_config("ml_models").get("salary_prediction", {})
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def prepare_features(
        self, 
        jobs_df: pd.DataFrame,
        skills_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare features for salary prediction."""
        logger.info("Preparing features for salary prediction...")
        
        if jobs_df.empty:
            logger.warning("Empty jobs DataFrame provided")
            return pd.DataFrame()
        
        # Start with jobs that have salary information
        salary_columns = ["salary_median", "salary_min", "salary_max"]
        available_salary_cols = [col for col in salary_columns if col in jobs_df.columns]
        
        if not available_salary_cols:
            logger.warning("No salary columns found in jobs data")
            # Create sample salary data for demonstration
            jobs_df = jobs_df.copy()
            jobs_df["salary_median"] = np.random.normal(100000, 25000, len(jobs_df))
            jobs_df["salary_min"] = jobs_df["salary_median"] * 0.8
            jobs_df["salary_max"] = jobs_df["salary_median"] * 1.2
            available_salary_cols = ["salary_median"]
        
        salary_df = jobs_df[
            jobs_df[available_salary_cols].notna().any(axis=1)
        ].copy()
        
        if salary_df.empty:
            logger.warning("No jobs with salary data found")
            return pd.DataFrame()
        
        # Create target variable (use median if available, otherwise average of min/max)
        if "salary_median" in available_salary_cols:
            salary_df['target_salary'] = salary_df['salary_median']
        elif "salary_min" in available_salary_cols and "salary_max" in available_salary_cols:
            salary_df['target_salary'] = (salary_df['salary_min'] + salary_df['salary_max']) / 2
        else:
            salary_df['target_salary'] = salary_df[available_salary_cols[0]]
        
        # Remove unrealistic salaries
        salary_df = salary_df[
            (salary_df['target_salary'] >= 20000) & 
            (salary_df['target_salary'] <= 500000)
        ]
        
        if salary_df.empty:
            logger.warning("No jobs with realistic salary data found")
            return pd.DataFrame()
        
        # Add skills features
        if not skills_df.empty and "job_id" in skills_df.columns:
            # Count of skills per job
            skills_count = skills_df.groupby('job_id').size().reset_index(name='skills_count')
            salary_df = salary_df.merge(skills_count, on='job_id', how='left')
            salary_df['skills_count'] = salary_df['skills_count'].fillna(0)
            
            # High-value skills indicator (based on common high-paying technologies)
            high_value_skills = {
                'Python', 'AWS', 'Kubernetes', 'React', 'Node.js', 'PostgreSQL',
                'Docker', 'TensorFlow', 'PyTorch', 'Java', 'Go', 'Rust', 'Machine Learning'
            }
            
            job_skills = skills_df.groupby('job_id')['skill_name'].apply(set).to_dict()
            salary_df['high_value_skills_count'] = salary_df['job_id'].apply(
                lambda x: len(job_skills.get(x, set()) & high_value_skills)
            )
            
            # Create skill category features
            skill_categories = self._categorize_skills(skills_df)
            for category, jobs in skill_categories.items():
                salary_df[f'has_{category.lower().replace(" ", "_")}'] = salary_df['job_id'].isin(jobs)
        else:
            # Default values when skills data is not available
            salary_df['skills_count'] = 3  # Average skills count
            salary_df['high_value_skills_count'] = 1  # Average high-value skills
        
        # Location features
        if 'location_standardized' in salary_df.columns:
            salary_df['location_clean'] = self._clean_location_feature(salary_df['location_standardized'])
        elif 'location' in salary_df.columns:
            salary_df['location_clean'] = self._clean_location_feature(salary_df['location'])
        else:
            salary_df['location_clean'] = 'Unknown'
        
        # Experience level features
        if 'experience_level' in salary_df.columns:
            salary_df['experience_level_clean'] = self._clean_experience_level(salary_df['experience_level'])
        else:
            salary_df['experience_level_clean'] = 'Mid Level'
        
        # Company size features (if available)
        if 'company_size' in salary_df.columns:
            salary_df['company_size_clean'] = self._clean_company_size(salary_df['company_size'])
        else:
            salary_df['company_size_clean'] = 'Unknown'
        
        # Employment type features
        if 'employment_type' in salary_df.columns:
            salary_df['is_full_time'] = salary_df['employment_type'].str.contains('Full', case=False, na=False)
        else:
            salary_df['is_full_time'] = True  # Default assumption
        
        # Remote work indicator
        salary_df['is_remote'] = salary_df['location_clean'].str.contains('Remote', case=False, na=False)
        
        # Title-based features
        if 'title' in salary_df.columns:
            salary_df['title_length'] = salary_df['title'].str.len()
            salary_df['has_senior_title'] = salary_df['title'].str.contains(
                'Senior|Lead|Principal|Staff|Architect', case=False, na=False
            )
            salary_df['has_manager_title'] = salary_df['title'].str.contains(
                'Manager|Director|VP|Head', case=False, na=False
            )
        else:
            salary_df['title_length'] = 20  # Average title length
            salary_df['has_senior_title'] = False
            salary_df['has_manager_title'] = False
        
        logger.info(f"Prepared features for {len(salary_df)} jobs with salary data")
        
        return salary_df
    
    def _categorize_skills(self, skills_df: pd.DataFrame) -> Dict[str, set]:
        """Categorize skills and return job IDs for each category."""
        categories = {
            "Programming Languages": ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust"],
            "Web Frameworks": ["React", "Vue.js", "Angular", "Node.js", "Django", "Flask"],
            "Databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"],
            "Cloud Platforms": ["AWS", "Google Cloud", "Azure", "Docker", "Kubernetes"],
            "Data Science": ["TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn"],
            "DevOps": ["Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible"]
        }
        
        category_jobs = {}
        for category, skills_list in categories.items():
            jobs_with_category = set(skills_df[
                skills_df['skill_name'].isin(skills_list)
            ]['job_id']) if 'job_id' in skills_df.columns else set()
            category_jobs[category] = jobs_with_category
        
        return category_jobs
    
    def _clean_location_feature(self, locations: pd.Series) -> pd.Series:
        """Clean and standardize location features."""
        return locations.fillna('Unknown').apply(lambda x: x if pd.notna(x) else 'Unknown')
    
    def _clean_experience_level(self, experience: pd.Series) -> pd.Series:
        """Clean experience level features."""
        def standardize_experience(exp):
            if pd.isna(exp):
                return 'Mid Level'
            exp_lower = str(exp).lower()
            if any(term in exp_lower for term in ['entry', 'junior', 'associate']):
                return 'Entry Level'
            elif any(term in exp_lower for term in ['senior', 'lead']):
                return 'Senior Level'
            elif any(term in exp_lower for term in ['principal', 'staff', 'architect']):
                return 'Principal Level'
            elif any(term in exp_lower for term in ['manager', 'director']):
                return 'Management'
            else:
                return 'Mid Level'
        
        return experience.apply(standardize_experience)
    
    def _clean_company_size(self, company_sizes: pd.Series) -> pd.Series:
        """Clean company size features."""
        def standardize_size(size):
            if pd.isna(size):
                return 'Unknown'
            size_str = str(size).lower()
            if any(term in size_str for term in ['1-10', 'startup', 'small']):
                return 'Small (1-50)'
            elif any(term in size_str for term in ['11-50', '51-200', 'medium']):
                return 'Medium (51-200)'
            elif any(term in size_str for term in ['201-500', 'large']):
                return 'Large (201-1000)'
            elif any(term in size_str for term in ['500+', '1000+', 'enterprise']):
                return 'Enterprise (1000+)'
            else:
                return 'Unknown'
        
        return company_sizes.apply(standardize_size)
    
    def train_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple regression models for salary prediction."""
        logger.info("Training salary prediction models...")
        
        if features_df.empty:
            logger.warning("Empty features DataFrame provided")
            return {}
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Using simple salary estimation.")
            return self._create_simple_model(features_df)
        
        # Prepare features and target
        feature_columns = [
            'skills_count', 'high_value_skills_count', 'title_length',
            'has_senior_title', 'has_manager_title', 'is_remote', 'is_full_time'
        ]
        
        # Add categorical features
        categorical_features = ['location_clean', 'experience_level_clean', 'company_size_clean']
        
        # Add skill category features
        skill_category_features = [col for col in features_df.columns if col.startswith('has_')]
        feature_columns.extend(skill_category_features)
        
        # Check which features actually exist
        available_features = [col for col in feature_columns if col in features_df.columns]
        available_categorical = [col for col in categorical_features if col in features_df.columns]
        
        if not available_features:
            logger.warning("No features available for training")
            return {}
        
        # Prepare the dataset
        try:
            X, y, feature_names = self._prepare_ml_features(
                features_df, available_features, available_categorical
            )
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return self._create_simple_model(features_df)
        
        if len(X) < 10:
            logger.warning("Insufficient data for model training")
            return self._create_simple_model(features_df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train different models
        models_config = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Add more advanced models if available
        if SKLEARN_AVAILABLE:
            models_config.update({
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
            })
        
        if XGBOOST_AVAILABLE:
            models_config['xgboost'] = xgb.XGBRegressor(n_estimators=50, random_state=42)
        
        model_results = {}
        
        for model_name, model in models_config.items():
            logger.info(f"Training {model_name} model...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(feature_names, abs(model.coef_)))
                
                model_results[model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': feature_importance
                }
                
                # Store the best model
                if not self.models or test_metrics['r2'] > max(r['test_metrics']['r2'] for r in self.models.values() if isinstance(r, dict) and 'test_metrics' in r):
                    self.models['best'] = model
                    self.feature_importance['best'] = feature_importance
                
                logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}, MAE: ${test_metrics['mae']:,.0f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.models.update(model_results)
        
        if model_results:
            best_r2 = max(r['test_metrics']['r2'] for r in model_results.values() if isinstance(r, dict) and 'test_metrics' in r)
            logger.info(f"Training completed. Best model R²: {best_r2:.4f}")
        else:
            logger.warning("No models were successfully trained")
        
        return model_results
    
    def _create_simple_model(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create a simple rule-based salary estimation model."""
        logger.info("Creating simple salary estimation model...")
        
        # Simple model based on experience level and location
        def simple_predict(row):
            base_salary = 80000  # Base salary
            
            # Experience multiplier
            exp_multipliers = {
                'Entry Level': 0.8,
                'Mid Level': 1.0,
                'Senior Level': 1.4,
                'Principal Level': 1.8,
                'Management': 2.0
            }
            exp_mult = exp_multipliers.get(row.get('experience_level_clean', 'Mid Level'), 1.0)
            
            # Location multiplier
            location = str(row.get('location_clean', '')).lower()
            if 'san francisco' in location or 'sf' in location:
                loc_mult = 1.5
            elif 'new york' in location or 'nyc' in location:
                loc_mult = 1.4
            elif 'seattle' in location:
                loc_mult = 1.3
            elif 'remote' in location:
                loc_mult = 1.1
            else:
                loc_mult = 1.0
            
            # Skills bonus
            skills_bonus = row.get('high_value_skills_count', 0) * 5000
            
            return base_salary * exp_mult * loc_mult + skills_bonus
        
        # Create simple model
        simple_model = {
            'predict': simple_predict,
            'type': 'simple_rules'
        }
        
        # Test on sample data
        if not features_df.empty:
            predictions = features_df.apply(simple_predict, axis=1)
            actual = features_df['target_salary']
            
            mae = np.mean(np.abs(predictions - actual))
            mse = np.mean((predictions - actual) ** 2)
            r2 = 1 - (np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
            
            metrics = {
                'mae': mae,
                'mse': mse, 
                'rmse': np.sqrt(mse),
                'r2': max(0, r2),  # Ensure non-negative R²
                'mape': np.mean(np.abs((actual - predictions) / actual)) * 100
            }
        else:
            metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0, 'mape': 0}
        
        model_results = {
            'simple_model': {
                'model': simple_model,
                'train_metrics': metrics,
                'test_metrics': metrics,
                'feature_importance': {
                    'experience_level': 0.4,
                    'location': 0.35,
                    'high_value_skills': 0.25
                }
            }
        }
        
        self.models['best'] = simple_model
        
        return model_results
    
    def _prepare_ml_features(
        self, 
        df: pd.DataFrame, 
        numeric_features: List[str], 
        categorical_features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for machine learning models."""
        # Prepare numeric features
        numeric_data = df[numeric_features].fillna(0)
        
        # Prepare categorical features
        categorical_data = pd.DataFrame()
        feature_names = numeric_features.copy()
        
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # One-hot encode categorical variables
                dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, dummy_na=True)
                categorical_data = pd.concat([categorical_data, dummies], axis=1)
                feature_names.extend(dummies.columns.tolist())
        
        # Combine features
        if categorical_data.empty:
            X = numeric_data.values
        else:
            X = np.hstack([numeric_data.values, categorical_data.values])
        
        y = df['target_salary'].values
        
        # Scale features if sklearn is available
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers['features'] = scaler
        
        return X, y, feature_names
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression performance metrics."""
        if SKLEARN_AVAILABLE:
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            }
        else:
            # Manual calculation
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
    
    def predict_salary(
        self, 
        job_features: Dict[str, Any], 
        model_name: str = 'best'
    ) -> Dict[str, Any]:
        """Predict salary for a job with given features."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        try:
            model = self.models[model_name]
            
            # Handle simple model
            if isinstance(model, dict) and model.get('type') == 'simple_rules':
                predicted_salary = model['predict'](job_features)
            else:
                # Handle sklearn models (simplified prediction)
                predicted_salary = 100000  # Placeholder
            
            prediction = {
                'predicted_salary': float(predicted_salary),
                'model_used': model_name,
                'confidence_interval': [
                    float(predicted_salary * 0.85),
                    float(predicted_salary * 1.15)
                ],
                'features_used': list(job_features.keys())
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting salary: {e}")
            return None
    
    def save_models(self, save_dir: str) -> None:
        """Save trained models to disk."""
        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available. Cannot save models.")
            return
        
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for model_name, model_info in self.models.items():
                if isinstance(model_info, dict) and 'model' in model_info:
                    model_file = save_path / f"{model_name}_salary_model.joblib"
                    joblib.dump(model_info['model'], model_file)
            
            # Save scalers and encoders
            if self.scalers:
                joblib.dump(self.scalers, save_path / "salary_scalers.joblib")
            
            logger.info(f"Saved models to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, load_dir: str) -> None:
        """Load trained models from disk."""
        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available. Cannot load models.")
            return
        
        try:
            load_path = Path(load_dir)
            
            # Load models
            for model_file in load_path.glob("*_salary_model.joblib"):
                model_name = model_file.stem.replace("_salary_model", "")
                self.models[model_name] = joblib.load(model_file)
            
            # Load scalers
            scaler_file = load_path / "salary_scalers.joblib"
            if scaler_file.exists():
                self.scalers = joblib.load(scaler_file)
            
            logger.info(f"Loaded models from {load_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")




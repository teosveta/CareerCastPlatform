"""FastAPI application for Skills Demand Forecasting Platform with Frontend Integration."""

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

import uvicorn

# Add the project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import your existing endpoints (with error handling)
ENDPOINTS_AVAILABLE = False
try:
    # Try absolute imports first
    from src.api.endpoints import skills, predictions, analytics, career
    ENDPOINTS_AVAILABLE = True
    print("✅ Endpoints imported successfully (absolute import)")
except ImportError:
    try:
        # Try relative imports
        from .endpoints import skills, predictions, analytics, career
        ENDPOINTS_AVAILABLE = True
        print("✅ Endpoints imported successfully (relative import)")
    except ImportError:
        print("⚠️  Endpoints could not be imported - using fallback endpoints only")

# Import request/response models (with error handling)
REQUEST_MODELS_AVAILABLE = False
try:
    from src.api.models.request_models import SkillQuery, SalaryPredictionRequest, ForecastRequest
    from src.api.models.response_models import SkillResponse, ForecastResponse, SalaryPredictionResponse
    REQUEST_MODELS_AVAILABLE = True
except ImportError:
    try:
        from .models.request_models import SkillQuery, SalaryPredictionRequest, ForecastRequest
        from .models.response_models import SkillResponse, ForecastResponse, SalaryPredictionResponse
        REQUEST_MODELS_AVAILABLE = True
    except ImportError:
        print("⚠️  Request/Response models not available - using basic dict types")
        # Define basic types as fallback
        SkillQuery = dict
        SalaryPredictionRequest = dict
        ForecastRequest = dict

# Get the project root directory
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CareerCast Platform - Skills Demand Forecasting API",
    description="AI-powered platform for analyzing technology job market trends and predicting skill demand",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    print(f"✅ Static files mounted from: {STATIC_DIR}")
else:
    print(f"⚠️  Static directory not found at {STATIC_DIR}")
    # Create the directory
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created static directory: {STATIC_DIR}")

# Setup templates
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    print(f"✅ Templates loaded from: {TEMPLATES_DIR}")
else:
    print(f"⚠️  Templates directory not found at {TEMPLATES_DIR}")
    # Create the directory
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created templates directory: {TEMPLATES_DIR}")
    templates = None

# Include API routers with /api prefix (if available)
if ENDPOINTS_AVAILABLE:
    try:
        app.include_router(skills.router, prefix="/api/v1/skills", tags=["skills"])
        app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
        app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
        app.include_router(career.router, prefix="/api/v1/career", tags=["career"])
        print("✅ All API endpoints loaded successfully")
    except Exception as e:
        print(f"⚠️  Some endpoints could not be loaded: {e}")
        ENDPOINTS_AVAILABLE = False

# Frontend Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend application."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Fallback HTML if templates directory is not found
        return HTMLResponse(content=get_fallback_html(), status_code=200)

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the dashboard page."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content=get_fallback_html(), status_code=200)

# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "frontend_enabled": templates is not None,
        "static_files_enabled": STATIC_DIR.exists()
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "1.0.0",
        "status": "operational",
        "features": {
            "skills_analysis": True,
            "demand_forecasting": True,
            "salary_prediction": True,
            "career_guidance": True,
            "frontend_integration": True
        },
        "endpoints_loaded": ENDPOINTS_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }

# Fallback API endpoints (if main endpoints not available)
@app.post("/api/v1/skills/extract")
async def extract_skills_fallback(request: dict):
    """Fallback skill extraction endpoint."""
    text = request.get("text", "")
    method = request.get("method", "hybrid")
    confidence_threshold = request.get("confidence_threshold", 0.7)

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Simple keyword-based extraction
    skills_keywords = [
        "Python", "JavaScript", "React", "AWS", "Docker", "Kubernetes",
        "PostgreSQL", "MongoDB", "Machine Learning", "TensorFlow",
        "Java", "C++", "Node.js", "Angular", "Vue.js", "Git", "SQL"
    ]

    text_lower = text.lower()
    extracted_skills = []
    confidence_scores = {}

    for skill in skills_keywords:
        if skill.lower() in text_lower:
            extracted_skills.append(skill)
            confidence_scores[skill] = 0.85  # Default confidence

    return {
        "extracted_skills": extracted_skills,
        "confidence_scores": confidence_scores,
        "method_used": method,
        "total_skills_found": len(extracted_skills)
    }

@app.post("/api/v1/predictions/salary-prediction")
async def predict_salary_fallback(request: dict):
    """Fallback salary prediction endpoint."""
    skills = request.get("skills", [])
    location = request.get("location", "")
    experience_level = request.get("experience_level", "Mid Level")

    if not skills:
        raise HTTPException(status_code=400, detail="Skills are required")

    # Simple salary calculation
    base_salaries = {
        "Entry Level": 70000,
        "Mid Level": 95000,
        "Senior Level": 130000,
        "Principal Level": 170000,
        "Management": 200000
    }

    location_multipliers = {
        "San Francisco, CA": 1.5,
        "New York, NY": 1.4,
        "Seattle, WA": 1.3,
        "Austin, TX": 1.1,
        "Remote": 1.0,
        "Chicago, IL": 1.05,
        "Boston, MA": 1.2
    }

    base_salary = base_salaries.get(experience_level, 95000)
    location_mult = location_multipliers.get(location, 1.0)

    # High-value skills bonus
    high_value_skills = {"Python", "AWS", "Kubernetes", "React", "Machine Learning"}
    skill_bonus = len([s for s in skills if s in high_value_skills]) * 8000

    predicted_salary = int(base_salary * location_mult + skill_bonus)

    return {
        "predicted_salary": predicted_salary,
        "salary_range": {
            "min": int(predicted_salary * 0.85),
            "max": int(predicted_salary * 1.15)
        },
        "confidence_score": 0.87,
        "factors_analysis": {
            "high_impact_skills": [s for s in skills if s in high_value_skills],
            "location_impact": "positive" if location_mult > 1 else "neutral",
            "experience_impact": "high" if experience_level in ["Senior Level", "Principal Level"] else "moderate"
        },
        "model_used": "fallback_model",
        "similar_jobs_count": 150
    }

@app.post("/api/v1/career/skill-gap-analysis")
async def skill_gap_analysis_fallback(request: dict):
    """Fallback skill gap analysis endpoint."""
    current_skills = request.get("current_skills", [])
    target_role = request.get("target_role", "")
    experience_level = request.get("experience_level", "Mid Level")

    if not current_skills or not target_role:
        raise HTTPException(status_code=400, detail="Current skills and target role are required")

    # Role requirements
    role_requirements = {
        "Data Scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas"],
        "Full Stack Developer": ["JavaScript", "React", "Node.js", "SQL", "Git"],
        "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Linux", "Python"],
        "Product Manager": ["Analytics", "SQL", "Project Management", "User Research"],
        "ML Engineer": ["Python", "TensorFlow", "Docker", "Kubernetes", "AWS"]
    }

    required_skills = role_requirements.get(target_role, [])
    missing_skills = [skill for skill in required_skills if skill not in current_skills]

    readiness_score = ((len(required_skills) - len(missing_skills)) / len(required_skills)) * 100 if required_skills else 100

    return {
        "target_role": target_role,
        "readiness_score": round(readiness_score, 1),
        "missing_required_skills": missing_skills,
        "skills_you_have": [skill for skill in current_skills if skill in required_skills],
        "estimated_learning_time_weeks": len(missing_skills) * 6
    }

@app.post("/api/v1/predictions/demand-forecast")
async def demand_forecast_fallback(request: dict):
    """Fallback demand forecasting endpoint."""
    skills = request.get("skills", [])
    forecast_horizon = request.get("forecast_horizon", 12)

    if not skills:
        raise HTTPException(status_code=400, detail="Skills are required")

    forecasts = []

    for skill in skills:
        # Generate sample forecast data
        dates = []
        values = []

        for i in range(forecast_horizon):
            date = datetime.now() + timedelta(days=30 * i)
            dates.append(date.strftime("%Y-%m-%d"))

            # Simple trend calculation
            base_value = 100
            growth_rate = 0.05  # 5% monthly growth
            noise = (hash(skill + str(i)) % 20 - 10) / 10  # Deterministic "noise"
            value = base_value * (1 + growth_rate) ** i + noise
            values.append(max(0, value))

        forecast = {
            "skill_name": skill,
            "forecast_dates": dates,
            "forecast_values": values,
            "confidence_intervals": {
                "lower": [v * 0.8 for v in values],
                "upper": [v * 1.2 for v in values]
            },
            "trend_direction": "increasing",
            "growth_rate": 15.0 + (hash(skill) % 20)  # Deterministic growth rate
        }

        forecasts.append(forecast)

    return {
        "forecasts": forecasts,
        "forecast_horizon": forecast_horizon,
        "model_type": "fallback_model",
        "generated_at": datetime.now().isoformat()
    }

def get_fallback_html():
    """Fallback HTML when templates are not available."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CareerCast Platform</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                margin: 0;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                text-align: center;
                max-width: 600px;
            }
            h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            p {
                font-size: 1.2rem;
                margin-bottom: 2rem;
                opacity: 0.9;
            }
            .btn {
                display: inline-block;
                padding: 1rem 2rem;
                background: rgba(255,255,255,0.2);
                border: 2px solid white;
                border-radius: 50px;
                color: white;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                margin: 0.5rem;
            }
            .btn:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-2px);
            }
            .status {
                background: rgba(255,255,255,0.1);
                padding: 1rem;
                border-radius: 1rem;
                margin-top: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 CareerCast Platform</h1>
            <p>AI-Powered Skills Demand Forecasting</p>
            <div>
                <a href="/api/docs" class="btn">📚 API Documentation</a>
                <a href="/api/health" class="btn">💚 Health Check</a>
            </div>
            <div class="status">
                <p><strong>Status:</strong> Frontend templates not found</p>
                <p>Please ensure the templates are in: <code>dashboard/templates/</code></p>
                <p>And static files are in: <code>dashboard/static/</code></p>
            </div>
        </div>
    </body>
    </html>
    """

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )

def main():
    """Main function to run the API server."""
    print("🚀 Starting CareerCast Platform...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Templates dir: {TEMPLATES_DIR}")
    print(f"📁 Static dir: {STATIC_DIR}")
    print(f"📊 Frontend available: {templates is not None}")

    # Create directories if they don't exist
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "css").mkdir(exist_ok=True)
    (STATIC_DIR / "js").mkdir(exist_ok=True)

    print("\n🌐 Access points:")
    print("   Frontend: http://localhost:8000/")
    print("   API Docs: http://localhost:8000/api/docs")
    print("   Health: http://localhost:8000/api/health")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

if __name__ == "__main__":
    main()
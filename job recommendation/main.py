from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import asyncio
import httpx
from datetime import datetime

# Import the BERT recommender from your recommendation_engine.py file
from recommendation_engine import BERTJobRecommenderSystem, create_bert_prediction_function


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT-Enhanced Job Recommendation API",
    description="Advanced job recommendation system using BERT embeddings and collaborative filtering",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class UserProfile(BaseModel):
    skills: Union[List[str], str] = Field(..., description="User skills as list or comma-separated string")
    experience: Optional[int] = Field(None, description="Years of experience", ge=0, le=50)
    experience_level: Optional[str] = Field(None, description="Experience level (entry, mid, senior)")
    role_category: Optional[str] = Field(None, description="Desired role category")
    industry: Optional[str] = Field(None, description="Preferred industry")
    functional_area: Optional[str] = Field(None, description="Functional area of expertise")
    job_title: Optional[str] = Field(None, description="Desired job title")
    preferred_location: Optional[str] = Field(None, description="Preferred job location")
    expected_salary: Optional[str] = Field(None, description="Expected salary range")
    career_goals: Optional[List[str]] = Field(None, description="Career goals and objectives")

class JobRecommendation(BaseModel):
    job_id: int
    job_title: str
    industry: str
    functional_area: str
    experience_required: str
    key_skills: str
    salary: Optional[str] = None
    similarity_score: Optional[float] = None
    rank: int

class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    num_recommendations: Optional[int] = Field(5, ge=1, le=20)
    recommendation_type: Optional[str] = Field("hybrid", pattern="^(bert|collaborative|hybrid)$")  # Changed regex to pattern
    user_id: Optional[int] = Field(None, description="User ID for collaborative filtering")
    reference_job_id: Optional[int] = Field(None, description="Reference job ID for similar job recommendations")

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[JobRecommendation]
    total_jobs_in_system: int
    recommendation_type: str
    processing_time_ms: float
    message: Optional[str] = None

class SkillGap(BaseModel):
    skill: str
    importance: str
    frequency_in_jobs: int
    learning_resources: List[str] = []

class SkillAnalysisResponse(BaseModel):
    user_skills: List[str]
    skill_gaps: List[SkillGap]
    skill_match_percentage: float
    recommendations_count: int

# Global variables to hold the model
bert_recommender = None
prediction_function = None
model_loaded = False

class ModelManager:
    def __init__(self):
        self.bert_recommender = None
        self.prediction_function = None
        self.model_loaded = False
        self.jobs_df = None
        
    async def load_model(self, model_path: str = None, jobs_csv_path: str = None):
        """Load the BERT recommender model"""
        try:
            logger.info("Loading BERT job recommender model...")
            
            if model_path and Path(model_path).exists():
                # Load pre-trained BERT model
                self.bert_recommender = BERTJobRecommenderSystem.load_model(model_path)
                logger.info(f"Loaded BERT model from {model_path}")
            else:
                # Load jobs data and create new model
                if jobs_csv_path and Path(jobs_csv_path).exists():
                    jobs_df = pd.read_csv(jobs_csv_path)
                    logger.info(f"Loaded jobs data: {len(jobs_df)} jobs")
                elif Path("jobs.csv").exists():
                    jobs_df = pd.read_csv("jobs.csv")
                    logger.info(f"Loaded jobs data from jobs.csv: {len(jobs_df)} jobs")
                else:
                    # Try to load from your friend's pickle file
                    try:
                        with open("models/jobs_data.pkl", "rb") as f:
                            jobs_df = pickle.load(f)
                        logger.info(f"Loaded jobs data from pickle: {len(jobs_df)} jobs")
                    except:
                        raise FileNotFoundError("No jobs data found. Please provide jobs.csv or models/jobs_data.pkl")
                
                self.jobs_df = jobs_df
                
                # Try to load user interactions
                user_interactions_df = None
                try:
                    if Path("models/user_interactions.pkl").exists():
                        with open("models/user_interactions.pkl", "rb") as f:
                            user_interactions_df = pickle.load(f)
                        logger.info("Loaded user interactions data")
                except:
                    logger.warning("No user interactions data found, using content-based only")
                
                # Create BERT recommender
                self.bert_recommender = BERTJobRecommenderSystem(jobs_df, user_interactions_df)
                
                # Build embeddings (this might take a few minutes)
                logger.info("Building BERT embeddings... This may take a few minutes.")
                self.bert_recommender.build_bert_embeddings()
                
                if user_interactions_df is not None:
                    self.bert_recommender.build_collaborative_model()
                
                logger.info("BERT model initialization complete")
            
            # Create prediction function
            self.prediction_function = create_bert_prediction_function(self.bert_recommender)
            self.model_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_recommendations(self, user_data: dict, num_recommendations: int = 5, rec_type: str = "hybrid"):
        """Get recommendations using the loaded model"""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            if rec_type == "bert":
                # Pure BERT-based recommendations
                recommendations_df = self.bert_recommender.recommend_jobs_for_user_profile(
                    user_data, top_n=num_recommendations
                )
            elif rec_type == "collaborative" and self.bert_recommender.user_interactions_df is not None:
                # Pure collaborative filtering (if user_id provided)
                user_id = user_data.get('user_id')
                if user_id:
                    recommendations_df = self.bert_recommender.get_collaborative_recommendations(
                        user_id, top_n=num_recommendations
                    )
                else:
                    # Fallback to BERT if no user_id
                    recommendations_df = self.bert_recommender.recommend_jobs_for_user_profile(
                        user_data, top_n=num_recommendations
                    )
            else:
                # Hybrid approach (default)
                recommendations_df = self.bert_recommender.get_hybrid_recommendations(
                    user_data, top_n=num_recommendations
                )
            
            return recommendations_df
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load the BERT model on startup"""
    logger.info("Starting BERT Job Recommendation API...")
    
    # Try different data sources in order of preference
    data_sources = [
        ("models/bert_job_recommender.pkl", "models/jobs_data.pkl"),
        (None, "data/jobs.csv"),
        (None, "jobs.csv"),
        (None, "../data/job_postings.csv")
    ]
    
    success = False
    for model_path, jobs_path in data_sources:
        try:
            success = await model_manager.load_model(model_path, jobs_path)
            if success:
                logger.info(f"Successfully loaded model with data from {jobs_path}")
                break
        except Exception as e:
            logger.warning(f"Failed to load from {jobs_path}: {e}")
            continue
    
    if not success:
        logger.error("Failed to load BERT model and data. API will have limited functionality.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BERT-Enhanced Job Recommendation API",
        "version": "2.0.0",
        "model_loaded": model_manager.model_loaded,
        "endpoints": {
            "recommendations": "/recommend",
            "similar_jobs": "/similar-jobs/{job_id}",
            "skill_analysis": "/analyze-skills",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get job recommendations using BERT embeddings"""
    start_time = datetime.now()
    
    try:
        # Convert user profile to dictionary
        user_data = request.user_profile.dict()
        
        # Handle skills input (list or comma-separated string)
        if isinstance(user_data['skills'], str):
            user_data['skills'] = [skill.strip() for skill in user_data['skills'].split(',')]
        
        # Add user_id if provided for collaborative filtering
        if request.user_id:
            user_data['user_id'] = request.user_id
        
        # Get recommendations
        recommendations_df = model_manager.get_recommendations(
            user_data, 
            request.num_recommendations, 
            request.recommendation_type
        )
        
        # Convert to response format
        recommendations = []
        for idx, (_, row) in enumerate(recommendations_df.iterrows()):
            rec = JobRecommendation(
                job_id=int(row.get('job_id', idx)),
                job_title=str(row.get('job_title', 'N/A')),
                industry=str(row.get('industry', 'N/A')),
                functional_area=str(row.get('functional_area', 'N/A')),
                experience_required=str(row.get('experience_required', 'N/A')),
                key_skills=str(row.get('key_skills', 'N/A')),
                salary=str(row.get('salary', None)) if row.get('salary') else None,
                similarity_score=float(row.get('similarity_score', 0.0)),
                rank=idx + 1
            )
            recommendations.append(rec)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            success=True,
            recommendations=recommendations,
            total_jobs_in_system=len(model_manager.jobs_df) if model_manager.jobs_df is not None else 0,
            recommendation_type=request.recommendation_type,
            processing_time_ms=processing_time,
            message=f"Generated {len(recommendations)} recommendations using {request.recommendation_type} approach"
        )
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-jobs/{job_id}")
async def get_similar_jobs(job_id: int, num_similar: int = 5):
    """Get jobs similar to a specific job using BERT embeddings"""
    try:
        if not model_manager.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        similar_jobs = model_manager.bert_recommender.get_similar_jobs(job_id, top_n=num_similar)
        
        return {
            "reference_job_id": job_id,
            "similar_jobs": similar_jobs.to_dict('records'),
            "count": len(similar_jobs)
        }
        
    except Exception as e:
        logger.error(f"Error getting similar jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-skills", response_model=SkillAnalysisResponse)
async def analyze_skills(user_profile: UserProfile):
    """Analyze user skills and identify gaps based on job market"""
    try:
        if not model_manager.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert skills to list if string
        user_skills = user_profile.skills
        if isinstance(user_skills, str):
            user_skills = [skill.strip() for skill in user_skills.split(',')]
        
        # Get skill analysis from BERT recommender
        skill_analysis = model_manager.bert_recommender.analyze_skill_gaps(user_skills)
        
        # Convert to response format
        skill_gaps = []
        for gap in skill_analysis.get('skill_gaps', []):
            skill_gap = SkillGap(
                skill=gap['skill'],
                importance=gap.get('importance', 'medium'),
                frequency_in_jobs=gap.get('frequency', 0),
                learning_resources=gap.get('resources', [])
            )
            skill_gaps.append(skill_gap)
        
        return SkillAnalysisResponse(
            user_skills=user_skills,
            skill_gaps=skill_gaps,
            skill_match_percentage=skill_analysis.get('match_percentage', 0.0),
            recommendations_count=skill_analysis.get('recommendations_count', 0)
        )
        
    except Exception as e:
        logger.error(f"Error in skill analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining with new data"""
    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    def retrain_task():
        try:
            logger.info("Starting model retraining...")
            model_manager.bert_recommender.retrain_model()
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Model retraining started in background"}

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model_manager.model_loaded:
        return {"model_loaded": False, "message": "No model loaded"}
    
    return {
        "model_loaded": True,
        "model_type": "BERT-Enhanced Job Recommender",
        "bert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "total_jobs": len(model_manager.jobs_df) if model_manager.jobs_df is not None else 0,
        "has_collaborative_filtering": model_manager.bert_recommender.user_interactions_df is not None,
        "embedding_dimension": 384,
        "supported_recommendation_types": ["bert", "collaborative", "hybrid"]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
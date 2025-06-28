import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional, Tuple
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTJobRecommenderSystem:
    def __init__(self, jobs_df, user_interactions_df=None, bert_model_name='all-MiniLM-L6-v2'):
        """
        Enhanced Job Recommender System with BERT integration
        
        Parameters:
        jobs_df (DataFrame): DataFrame containing job listings
        user_interactions_df (DataFrame): Optional DataFrame containing user-job interactions
        bert_model_name (str): Name of the BERT model to use for embeddings
        """
        self.jobs_df = jobs_df.copy()
        
        # Adding a job ID if not present
        if 'Job_ID' not in self.jobs_df.columns:
            self.jobs_df['Job_ID'] = range(len(self.jobs_df))
        
        self.user_interactions_df = user_interactions_df
        
        # BERT model initialization
        self.bert_model_name = bert_model_name
        self.bert_model = None
        self.job_embeddings = None
        
        # Traditional components (keeping compatibility)
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.hybrid_weights = {'content': 0.6, 'collaborative': 0.4}  # Give more weight to BERT content
        
        # Initialize BERT model
        self._initialize_bert_model()
    
    def _initialize_bert_model(self):
        """Initialize the BERT model for embeddings"""
        try:
            logger.info(f"Loading BERT model: {self.bert_model_name}")
            self.bert_model = SentenceTransformer(self.bert_model_name)
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess job data for recommendation"""
        # Handle salary - convert to numeric, or create a binary indicator
        if 'Job Salary' in self.jobs_df.columns:
            try:
                # Try direct conversion to numeric
                self.jobs_df['Salary_Numeric'] = pd.to_numeric(self.jobs_df['Job Salary'], errors='coerce')
                
                if self.jobs_df['Salary_Numeric'].isna().all():
                    logger.warning("Could not convert salary values to numeric. Using binary indicator.")
                    self.jobs_df['Salary_Disclosed'] = self.jobs_df['Job Salary'].apply(
                        lambda x: 0 if str(x).lower().strip() in ['not disclosed', 'not disclosed by recruiter', 'na', 'n/a', ''] else 1
                    )
                else:
                    median_salary = self.jobs_df['Salary_Numeric'].median()
                    self.jobs_df['Salary_Numeric'] = self.jobs_df['Salary_Numeric'].fillna(median_salary)
                    
                    # Normalize the numeric salary
                    scaler = MinMaxScaler()
                    self.jobs_df['Normalized_Salary'] = scaler.fit_transform(self.jobs_df[['Salary_Numeric']])
            except Exception as e:
                logger.warning(f"Error processing salary data: {e}")
        
        # Handle experience - convert to numeric if possible
        if 'Job Experience Required' in self.jobs_df.columns:
            try:
                def extract_years(exp_text):
                    if pd.isna(exp_text) or not isinstance(exp_text, str):
                        return 0
                    
                    exp_text = exp_text.lower().strip()
                    if '-' in exp_text:
                        parts = exp_text.split('-')
                        if len(parts) == 2:
                            try:
                                min_exp = float(''.join(c for c in parts[0] if c.isdigit() or c == '.'))
                                max_exp = float(''.join(c for c in parts[1] if c.isdigit() or c == '.'))
                                return (min_exp + max_exp) / 2
                            except ValueError:
                                return 0
                    else:
                        try:
                            return float(''.join(c for c in exp_text if c.isdigit() or c == '.'))
                        except ValueError:
                            return 0
                
                self.jobs_df['Experience_Numeric'] = self.jobs_df['Job Experience Required'].apply(extract_years)
                
                # Normalize the numeric experience
                scaler = MinMaxScaler()
                self.jobs_df['Normalized_Experience'] = scaler.fit_transform(self.jobs_df[['Experience_Numeric']])
            except Exception as e:
                logger.warning(f"Error processing experience data: {e}")
        
        # Create combined text feature for BERT embedding
        text_features = []
        
        # Add all text features that exist in the dataframe
        for col in ['Job Title', 'Key Skills', 'Role Category', 'Functional Area', 'Industry']:
            if col in self.jobs_df.columns:
                text_features.append(self.jobs_df[col].fillna('').astype(str))
        
        # Combine all text features
        if text_features:
            self.jobs_df['Combined_Features'] = pd.Series(
                ' '.join(str(val) for val in vals) for vals in zip(*text_features)
            )
        else:
            logger.warning("No text features found for content-based filtering")
            self.jobs_df['Combined_Features'] = ""
    
    def build_bert_embeddings(self, batch_size=32):
        """Build BERT embeddings for all job descriptions"""
        logger.info("Building BERT embeddings for jobs...")
        
        if self.bert_model is None:
            raise ValueError("BERT model not initialized")
        
        self.preprocess_data()
        
        # Get job descriptions
        job_descriptions = self.jobs_df['Combined_Features'].tolist()
        
        # Generate embeddings in batches to handle memory efficiently
        embeddings = []
        for i in range(0, len(job_descriptions), batch_size):
            batch = job_descriptions[i:i + batch_size]
            batch_embeddings = self.bert_model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(job_descriptions)} jobs")
        
        # Concatenate all embeddings
        self.job_embeddings = np.vstack(embeddings)
        logger.info(f"BERT embeddings created with shape: {self.job_embeddings.shape}")
        
        return self.job_embeddings
    
    def get_bert_recommendations(self, job_id=None, user_profile_text=None, top_n=5):
        """
        Get recommendations using BERT embeddings
        
        Parameters:
        job_id: ID of a job to find similar jobs for
        user_profile_text: Text description of user profile/requirements
        top_n: Number of recommendations to return
        """
        if self.job_embeddings is None:
            self.build_bert_embeddings()
        
        if job_id is not None:
            # Job-to-job similarity
            if job_id not in self.jobs_df['Job_ID'].values:
                logger.error(f"Job ID {job_id} not found in dataset")
                return pd.DataFrame()
            
            # Find the index of the job
            job_idx = self.jobs_df[self.jobs_df['Job_ID'] == job_id].index[0]
            
            # Calculate similarity with all other jobs
            similarities = cosine_similarity(
                self.job_embeddings[job_idx:job_idx+1], 
                self.job_embeddings
            ).flatten()
            
            # Get top similar jobs (excluding the input job)
            similar_indices = similarities.argsort()[::-1]
            similar_indices = [idx for idx in similar_indices if idx != job_idx][:top_n]
            
            return self.jobs_df.iloc[similar_indices]
        
        elif user_profile_text is not None:
            # User profile to job similarity
            user_embedding = self.bert_model.encode([user_profile_text])
            
            # Calculate similarity with all jobs
            similarities = cosine_similarity(user_embedding, self.job_embeddings).flatten()
            
            # Get top similar jobs
            similar_indices = similarities.argsort()[::-1][:top_n]
            
            return self.jobs_df.iloc[similar_indices]
        
        else:
            raise ValueError("Either job_id or user_profile_text must be provided")
    
    def build_collaborative_model(self):
        """Build collaborative filtering model (keeping existing functionality)"""
        if self.user_interactions_df is None:
            logger.warning("No user interaction data available for collaborative filtering")
            return None
        
        # Create user-item matrix
        user_item_matrix = pd.pivot_table(
            self.user_interactions_df,
            values='Rating',
            index='User_ID',
            columns='Job_ID',
            fill_value=0
        )
        
        self.user_item_matrix = user_item_matrix
        
        # Calculate item-item similarity matrix
        item_item_similarity = cosine_similarity(user_item_matrix.T)
        self.similarity_matrix = pd.DataFrame(
            item_item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        return self.similarity_matrix
    
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Get collaborative filtering recommendations (keeping existing functionality)"""
        if self.similarity_matrix is None:
            if self.user_interactions_df is None:
                logger.warning("Cannot provide collaborative recommendations without user interaction data")
                return pd.DataFrame()
            self.build_collaborative_model()
        
        if user_id not in self.user_item_matrix.index:
            logger.warning(f"User {user_id} not found in interaction data")
            return pd.DataFrame()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_jobs = user_ratings[user_ratings > 0].index
        
        predicted_ratings = {}
        
        for job_id in self.similarity_matrix.columns:
            if job_id not in rated_jobs:
                similar_jobs = self.similarity_matrix[job_id]
                similar_jobs_rated = similar_jobs[rated_jobs]
                
                if len(similar_jobs_rated) > 0:
                    numerator = sum(similar_jobs_rated * user_ratings[rated_jobs])
                    denominator = sum(abs(similar_jobs_rated))
                    
                    if denominator > 0:
                        predicted_ratings[job_id] = numerator / denominator
                    else:
                        predicted_ratings[job_id] = 0
        
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        top_job_ids = [job_id for job_id, _ in sorted_predictions[:top_n]]
        
        return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]
    
    def get_hybrid_recommendations(self, user_id=None, job_id=None, user_profile_text=None, top_n=5):
        """
        Get hybrid recommendations combining BERT-based content filtering and collaborative filtering
        
        Parameters:
        user_id: User ID for collaborative filtering
        job_id: Job ID for content-based filtering
        user_profile_text: User profile text for BERT-based matching
        top_n: Number of recommendations to return
        """
        bert_recommendations = pd.DataFrame()
        collab_recommendations = pd.DataFrame()
        
        # Get BERT-based recommendations
        if job_id is not None or user_profile_text is not None:
            bert_recommendations = self.get_bert_recommendations(
                job_id=job_id, 
                user_profile_text=user_profile_text, 
                top_n=top_n
            )
        
        # Get collaborative filtering recommendations
        if user_id is not None and self.user_interactions_df is not None:
            collab_recommendations = self.get_collaborative_recommendations(user_id, top_n=top_n)
        
        # Combine recommendations
        if not bert_recommendations.empty and not collab_recommendations.empty:
            bert_jobs = set(bert_recommendations['Job_ID'])
            collab_jobs = set(collab_recommendations['Job_ID'])
            
            all_recommended_jobs = bert_jobs.union(collab_jobs)
            
            hybrid_scores = {}
            for job in all_recommended_jobs:
                score = 0
                if job in bert_jobs:
                    rank = list(bert_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['content'] * (1 - (rank / len(bert_jobs)))
                
                if job in collab_jobs:
                    rank = list(collab_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['collaborative'] * (1 - (rank / len(collab_jobs)))
                
                hybrid_scores[job] = score
            
            sorted_jobs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_job_ids = [job_id for job_id, _ in sorted_jobs]
            
            return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]
        
        elif not bert_recommendations.empty:
            return bert_recommendations
        elif not collab_recommendations.empty:
            return collab_recommendations
        else:
            return self.jobs_df.head(top_n)
    
    def recommend_jobs_for_user_profile(self, user_profile: Dict, top_n=5):
        """
        Recommend jobs based on user profile using BERT embeddings
        
        Parameters:
        user_profile: Dictionary containing user profile information
        top_n: Number of recommendations to return
        """
        # Create a comprehensive text description from user profile
        profile_text_parts = []
        
        # Add skills
        if 'skills' in user_profile:
            if isinstance(user_profile['skills'], list):
                skills_text = ' '.join(user_profile['skills'])
            else:
                skills_text = str(user_profile['skills'])
            profile_text_parts.append(f"Skills: {skills_text}")
        
        # Add experience
        if 'experience' in user_profile:
            profile_text_parts.append(f"Experience: {user_profile['experience']} years")
        
        # Add role category
        if 'role_category' in user_profile:
            profile_text_parts.append(f"Role: {user_profile['role_category']}")
        
        # Add industry
        if 'industry' in user_profile:
            profile_text_parts.append(f"Industry: {user_profile['industry']}")
        
        # Add functional area
        if 'functional_area' in user_profile:
            profile_text_parts.append(f"Functional Area: {user_profile['functional_area']}")
        
        # Add job title if specified
        if 'job_title' in user_profile:
            profile_text_parts.append(f"Desired Job Title: {user_profile['job_title']}")
        
        # Combine all parts
        user_profile_text = '. '.join(profile_text_parts)
        
        # Get BERT-based recommendations
        return self.get_bert_recommendations(user_profile_text=user_profile_text, top_n=top_n)
    
    def analyze_skill_gaps(self, user_skills):
        """Analyze skill gaps based on job market"""
        # Simple implementation - you can enhance this
        all_skills = []
        for _, job in self.jobs_df.iterrows():
            skills = str(job.get('Key Skills', '')).split(',')
            all_skills.extend([s.strip() for s in skills])
        
        skill_counts = pd.Series(all_skills).value_counts()
        
        skill_gaps = []
        for skill in skill_counts.head(10).index:
            if skill.lower() not in [s.lower() for s in user_skills]:
                skill_gaps.append({
                    'skill': skill,
                    'frequency': skill_counts[skill],
                    'importance': 'high' if skill_counts[skill] > 50 else 'medium'
                })
        
        return {
            'skill_gaps': skill_gaps[:5],
            'match_percentage': 75.0,  # Placeholder
            'recommendations_count': len(skill_gaps)
        }
    
    def get_similar_jobs(self, job_id, top_n=5):
        """Get jobs similar to a specific job"""
        return self.get_bert_recommendations(job_id=job_id, top_n=top_n)
    
    def retrain_model(self):
        """Retrain the model with new data"""
        self.build_bert_embeddings()
        if self.user_interactions_df is not None:
            self.build_collaborative_model()
    
    def save_model(self, filepath):
        """Save the model to a file"""
        model_data = {
            'jobs_df': self.jobs_df,
            'user_interactions_df': self.user_interactions_df,
            'job_embeddings': self.job_embeddings,
            'bert_model_name': self.bert_model_name,
            'hybrid_weights': self.hybrid_weights,
            'user_item_matrix': self.user_item_matrix,
            'similarity_matrix': self.similarity_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            jobs_df=model_data['jobs_df'], 
            user_interactions_df=model_data['user_interactions_df'],
            bert_model_name=model_data['bert_model_name']
        )
        
        # Restore embeddings and other data
        instance.job_embeddings = model_data['job_embeddings']
        instance.hybrid_weights = model_data['hybrid_weights']
        instance.user_item_matrix = model_data['user_item_matrix']
        instance.similarity_matrix = model_data['similarity_matrix']
        
        logger.info(f"Model loaded from {filepath}")
        return instance

# Integration function for FastAPI
def create_bert_prediction_function(model_instance):
    """
    Create a prediction function compatible with FastAPI
    """
    def predict_recommendations(user_data, num_recommendations=5):
        """
        Enhanced prediction function with BERT integration
        """
        try:
            # Get BERT-based recommendations
            recommendations_df = model_instance.recommend_jobs_for_user_profile(
                user_data, 
                top_n=num_recommendations
            )
            
            # Format recommendations for API response
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                recommendations.append({
                    'job_id': int(row['Job_ID']),
                    'job_title': str(row.get('Job Title', 'N/A')),
                    'industry': str(row.get('Industry', 'N/A')),
                    'functional_area': str(row.get('Functional Area', 'N/A')),
                    'experience_required': str(row.get('Job Experience Required', 'N/A')),
                    'key_skills': str(row.get('Key Skills', 'N/A')),
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in predict_recommendations: {e}")
            return []
    
    return predict_recommendations

# Example usage and migration from existing system
if __name__ == "__main__":
    # Example of how to migrate from existing system
    try:
        # Load existing data (assuming you have the CSV file)
        jobs_df = pd.read_csv('jobs.csv')  # Update path as needed
        
        # Create mock user interactions (as in your original code)
        sample_size = min(500, len(jobs_df))
        job_ids = jobs_df.index[:sample_size].tolist()
        
        sample_interactions = []
        for user_id in range(1, 51):  # 50 users
            num_interactions = np.random.randint(5, 21)
            selected_job_ids = np.random.choice(job_ids, size=min(num_interactions, len(job_ids)), replace=False)
            
            for job_id in selected_job_ids:
                rating = np.random.randint(1, 6)
                sample_interactions.append({
                    'User_ID': user_id,
                    'Job_ID': job_id,
                    'Rating': rating
                })
        
        user_interactions_df = pd.DataFrame(sample_interactions)
        
        # Initialize BERT-enhanced recommender
        bert_recommender = BERTJobRecommenderSystem(jobs_df, user_interactions_df)
        
        # Build BERT embeddings
        bert_recommender.build_bert_embeddings()
        
        # Build collaborative model
        bert_recommender.build_collaborative_model()
        
        # Test recommendations
        sample_job_id = jobs_df.index[0]
        sample_user_id = 1
        
        print("BERT-Enhanced Recommendations:") 
        
        # BERT-based content recommendations
        bert_recs = bert_recommender.get_bert_recommendations(job_id=sample_job_id, top_n=3)
        if not bert_recs.empty and 'Job Title' in bert_recs.columns:
            print(f"\nBERT Content-based recommendations for job {sample_job_id}:")
            print(bert_recs[['Job Title']].head(3))
        
        # User profile-based recommendations
        user_profile = {
            'skills': ['Python', 'Data Analysis', 'Machine Learning'],
            'experience': 3,
            'role_category': 'Data Science',
            'industry': 'Technology',
            'functional_area': 'Analytics'
        }
        
        profile_recs = bert_recommender.recommend_jobs_for_user_profile(user_profile, top_n=3)
        if not profile_recs.empty and 'Job Title' in profile_recs.columns:
            print("\nBERT Profile-based recommendations:")
            print(profile_recs[['Job Title']].head(3))
        
        # Hybrid recommendations
        hybrid_recs = bert_recommender.get_hybrid_recommendations(
            user_id=sample_user_id, 
            job_id=sample_job_id, 
            top_n=3
        )
        if not hybrid_recs.empty and 'Job Title' in hybrid_recs.columns:
            print("\nBERT Hybrid recommendations:")
            print(hybrid_recs[['Job Title']].head(3))
        
        # Save the enhanced model
        bert_recommender.save_model('models/bert_job_recommender.pkl')
        
        # Create prediction function for FastAPI
        predict_func = create_bert_prediction_function(bert_recommender)
        
        # Test the prediction function
        test_user_data = {
            'skills': ['Python', 'Machine Learning'],
            'experience': 2,
            'role_category': 'Data Science'
        }
        
        predictions = predict_func(test_user_data, 3)
        print(f"\nAPI-compatible predictions: {predictions}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        import traceback
        traceback.print_exc()

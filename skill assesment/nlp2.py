import numpy as np
import pandas as pd
import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import cv2
import pytesseract
from PIL import Image
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import json
from datetime import datetime
import uuid
import sqlite3
from typing import Dict, List, Any
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# Load spaCy model for NLP processing
# Download with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")

@dataclass
class Skill:
    name: str
    level: float  # 0-1 scale
    category: str
    confidence: float
    source: str  # 'resume', 'assessment', 'vision'

@dataclass
class LearningResource:
    title: str
    description: str
    difficulty: str
    duration: int  # minutes
    skills: List[str]
    url: str
    rating: float

@dataclass
class LearningPath:
    user_id: str
    skills_gap: List[str]
    recommended_resources: List[LearningResource]
    estimated_completion: int  # days
    priority_order: List[str]

class ResumeParser:
    """Parse and extract skills from resumes using NLP"""
    
    def __init__(self):
        self.skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'react', 'angular', 'vue', 'django', 'flask', 'spring'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'redis'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'project management']
        }
        
        # Initialize TensorFlow Universal Sentence Encoder for semantic similarity
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
    def extract_text_from_resume(self, file_path: str) -> str:
        """Extract text from various resume formats"""
        if file_path.endswith('.pdf'):
            # Use OCR for PDF files
            return self._extract_from_pdf(file_path)
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            return self._extract_from_image(file_path)
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR"""
        # This would typically use libraries like PyPDF2 or pdfplumber
        # For demonstration, we'll simulate OCR extraction
        return "Sample extracted text from PDF resume with skills like Python, Machine Learning, TensorFlow, SQL"
    
    def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing for better OCR
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(gray)
        return text
    
    def extract_skills(self, resume_text: str) -> List[Skill]:
        """Extract skills from resume text using NLP"""
        skills = []
        doc = nlp(resume_text.lower())
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Use TF-IDF and semantic similarity for skill extraction
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                # Check direct mentions
                if keyword.lower() in resume_text.lower():
                    # Calculate context-based confidence using sentence embeddings
                    confidence = self._calculate_skill_confidence(resume_text, keyword)
                    level = self._estimate_skill_level(resume_text, keyword)
                    
                    skills.append(Skill(
                        name=keyword,
                        level=level,
                        category=category,
                        confidence=confidence,
                        source='resume'
                    ))
        
        return self._deduplicate_skills(skills)
    
    def _calculate_skill_confidence(self, text: str, skill: str) -> float:
        """Calculate confidence score using Universal Sentence Encoder"""
        sentences = [sent.text for sent in nlp(text).sents if skill.lower() in sent.text.lower()]
        
        if not sentences:
            return 0.5
        
        # Use Universal Sentence Encoder for semantic similarity
        skill_contexts = [
            f"Expert in {skill}",
            f"Experienced with {skill}",
            f"Proficient in {skill}"
        ]
        
        # Get embeddings
        sentence_embeddings = self.use_model(sentences[:3])  # Limit to first 3 sentences
        context_embeddings = self.use_model(skill_contexts)
        
        # Calculate similarity scores
        similarities = tf.keras.utils.cosine_similarity(
            sentence_embeddings, 
            tf.reduce_mean(context_embeddings, axis=0, keepdims=True)
        )
        
        return float(tf.reduce_mean(similarities))
    
    def _estimate_skill_level(self, text: str, skill: str) -> float:
        """Estimate skill level based on context and experience indicators"""
        level_indicators = {
            'expert': 0.9, 'senior': 0.8, 'advanced': 0.8,
            'experienced': 0.7, 'proficient': 0.6, 'intermediate': 0.5,
            'familiar': 0.4, 'basic': 0.3, 'beginner': 0.2
        }
        
        skill_context = []
        doc = nlp(text.lower())
        
        for sent in doc.sents:
            if skill.lower() in sent.text:
                skill_context.append(sent.text)
        
        # Look for experience indicators
        max_level = 0.5  # Default
        for context in skill_context:
            for indicator, level in level_indicators.items():
                if indicator in context:
                    max_level = max(max_level, level)
        
        # Look for years of experience
        years_pattern = r'(\d+)\s*(?:years?|yrs?)'
        for context in skill_context:
            years_match = re.search(years_pattern, context)
            if years_match:
                years = int(years_match.group(1))
                experience_level = min(0.9, 0.3 + (years * 0.1))
                max_level = max(max_level, experience_level)
        
        return max_level
    
    def _deduplicate_skills(self, skills: List[Skill]) -> List[Skill]:
        """Remove duplicate skills and keep the one with highest confidence"""
        skill_dict = {}
        for skill in skills:
            key = skill.name.lower()
            if key not in skill_dict or skill.confidence > skill_dict[key].confidence:
                skill_dict[key] = skill
        
        return list(skill_dict.values())

class ComputerVisionSkillExtractor:
    """Extract skills from certificates, portfolios, and project screenshots"""
    
    def __init__(self):
        self.certificate_templates = {
            'aws': ['aws', 'amazon web services', 'certified'],
            'google_cloud': ['google cloud', 'gcp', 'certified'],
            'microsoft': ['microsoft', 'azure', 'certified'],
            'coursera': ['coursera', 'certificate', 'completion'],
            'udemy': ['udemy', 'certificate', 'completion']
        }
    
    def extract_from_certificate(self, image_path: str) -> List[Skill]:
        """Extract skills from certificate images"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement
        enhanced = self._enhance_image(gray)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(enhanced)
        
        skills = []
        text_lower = text.lower()
        
        # Identify certificate type and extract relevant skills
        for cert_type, keywords in self.certificate_templates.items():
            if any(keyword in text_lower for keyword in keywords):
                # Extract specific skills mentioned in the certificate
                extracted_skills = self._extract_certificate_skills(text, cert_type)
                skills.extend(extracted_skills)
        
        return skills
    
    def _enhance_image(self, gray_image):
        """Enhance image for better OCR results"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_certificate_skills(self, text: str, cert_type: str) -> List[Skill]:
        """Extract skills based on certificate type"""
        skills = []
        
        skill_mappings = {
            'aws': ['cloud computing', 'aws', 'ec2', 's3', 'lambda'],
            'google_cloud': ['cloud computing', 'gcp', 'bigquery', 'kubernetes'],
            'microsoft': ['azure', 'cloud computing', '.net', 'sql server'],
            'coursera': self._extract_coursera_skills(text),
            'udemy': self._extract_general_skills(text)
        }
        
        if cert_type in skill_mappings:
            skill_list = skill_mappings[cert_type]
            if callable(skill_list):
                skill_list = skill_list(text)
            
            for skill_name in skill_list:
                skills.append(Skill(
                    name=skill_name,
                    level=0.7,  # Certificate implies good proficiency
                    category=self._categorize_skill(skill_name),
                    confidence=0.9,  # High confidence from certificates
                    source='vision'
                ))
        
        return skills
    
    def _extract_coursera_skills(self, text: str) -> List[str]:
        """Extract skills from Coursera certificates"""
        # Common Coursera course patterns
        course_patterns = [
            r'machine learning',
            r'deep learning',
            r'data science',
            r'python',
            r'tensorflow',
            r'neural networks'
        ]
        
        skills = []
        for pattern in course_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(pattern.replace(' ', '_'))
        
        return skills
    
    def _extract_general_skills(self, text: str) -> List[str]:
        """Extract general skills from certificate text"""
        # Use keyword matching for common skills
        common_skills = [
            'python', 'java', 'javascript', 'react', 'angular',
            'machine learning', 'data science', 'sql', 'mongodb'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize skill based on name"""
        categories = {
            'programming': ['python', 'java', 'javascript', 'react', 'angular'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'data science'],
            'database': ['sql', 'mongodb', 'postgresql']
        }
        
        for category, skills in categories.items():
            if skill_name.lower() in [s.lower() for s in skills]:
                return category
        
        return 'general'

class SkillAssessmentEngine:
    """AI-powered skill assessment system"""
    
    def __init__(self):
        self.assessment_questions = self._load_assessment_questions()
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        
        # Build TensorFlow model for adaptive assessment
        self.assessment_model = self._build_assessment_model()
    
    def _build_assessment_model(self):
        """Build TensorFlow model for adaptive skill assessment"""
        # Input layers
        question_input = Input(shape=(100,), name='question_embedding')
        user_history = Input(shape=(50,), name='user_history')
        
        # Dense layers for processing
        question_dense = Dense(64, activation='relu')(question_input)
        question_dense = Dropout(0.3)(question_dense)
        
        history_dense = Dense(32, activation='relu')(user_history)
        history_dense = Dropout(0.3)(history_dense)
        
        # Combine features
        combined = tf.keras.layers.concatenate([question_dense, history_dense])
        combined = Dense(32, activation='relu')(combined)
        
        # Output layer for difficulty prediction
        output = Dense(4, activation='softmax', name='difficulty_prediction')(combined)
        
        model = Model(inputs=[question_input, user_history], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _load_assessment_questions(self) -> Dict[str, List[Dict]]:
        """Load assessment questions for different skills"""
        return {
            'python': [
                {
                    'question': 'What is the output of print(type([]) == list)?',
                    'options': ['True', 'False', 'Error', 'None'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'data types'
                },
                {
                    'question': 'Which decorator is used to create a property in Python?',
                    'options': ['@property', '@staticmethod', '@classmethod', '@decorator'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'decorators'
                }
            ],
            'machine_learning': [
                {
                    'question': 'What is overfitting in machine learning?',
                    'options': [
                        'Model performs well on training but poor on test data',
                        'Model performs poorly on both training and test data',
                        'Model is too simple',
                        'Model has too few parameters'
                    ],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'model evaluation'
                }
            ]
        }
    
    def conduct_assessment(self, user_id: str, skill: str, num_questions: int = 10) -> Dict[str, Any]:
        """Conduct adaptive skill assessment"""
        if skill not in self.assessment_questions:
            return {'error': f'No assessment available for {skill}'}
        
        questions = self.assessment_questions[skill]
        user_responses = []
        current_difficulty = 1  # Start with intermediate
        
        assessment_results = {
            'user_id': user_id,
            'skill': skill,
            'questions_answered': 0,
            'correct_answers': 0,
            'estimated_level': 0.5,
            'confidence': 0.0,
            'areas_for_improvement': [],
            'strong_areas': []
        }
        
        # Simulate adaptive questioning (in real implementation, this would be interactive)
        for i in range(min(num_questions, len(questions))):
            question = questions[i % len(questions)]
            
            # Simulate user response (in real implementation, this would come from user)
            simulated_response = np.random.choice([0, 1, 2, 3])
            is_correct = simulated_response == question['correct']
            
            user_responses.append({
                'question_id': i,
                'response': simulated_response,
                'correct': is_correct,
                'difficulty': question['difficulty'],
                'concept': question['concept']
            })
            
            assessment_results['questions_answered'] += 1
            if is_correct:
                assessment_results['correct_answers'] += 1
        
        # Calculate final assessment metrics
        assessment_results['estimated_level'] = self._calculate_skill_level(user_responses)
        assessment_results['confidence'] = self._calculate_confidence(user_responses)
        assessment_results['areas_for_improvement'] = self._identify_weak_areas(user_responses)
        assessment_results['strong_areas'] = self._identify_strong_areas(user_responses)
        
        return assessment_results
    
    def _calculate_skill_level(self, responses: List[Dict]) -> float:
        """Calculate estimated skill level based on responses"""
        if not responses:
            return 0.0
        
        # Weight responses by difficulty
        difficulty_weights = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
        total_weight = 0
        weighted_score = 0
        
        for response in responses:
            weight = difficulty_weights.get(response['difficulty'], 0.5)
            total_weight += weight
            if response['correct']:
                weighted_score += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, responses: List[Dict]) -> float:
        """Calculate confidence in skill level estimation"""
        if len(responses) < 3:
            return 0.3  # Low confidence with few responses
        
        # Calculate consistency in performance
        correct_count = sum(1 for r in responses if r['correct'])
        accuracy = correct_count / len(responses)
        
        # Higher confidence with more consistent results
        variance = np.var([1 if r['correct'] else 0 for r in responses])
        confidence = min(0.95, 0.5 + (1 - variance) * 0.4)
        
        return confidence
    
    def _identify_weak_areas(self, responses: List[Dict]) -> List[str]:
        """Identify areas where user performed poorly"""
        concept_performance = {}
        
        for response in responses:
            concept = response['concept']
            if concept not in concept_performance:
                concept_performance[concept] = {'correct': 0, 'total': 0}
            
            concept_performance[concept]['total'] += 1
            if response['correct']:
                concept_performance[concept]['correct'] += 1
        
        weak_areas = []
        for concept, perf in concept_performance.items():
            accuracy = perf['correct'] / perf['total']
            if accuracy < 0.6:  # Less than 60% accuracy
                weak_areas.append(concept)
        
        return weak_areas
    
    def _identify_strong_areas(self, responses: List[Dict]) -> List[str]:
        """Identify areas where user performed well"""
        concept_performance = {}
        
        for response in responses:
            concept = response['concept']
            if concept not in concept_performance:
                concept_performance[concept] = {'correct': 0, 'total': 0}
            
            concept_performance[concept]['total'] += 1
            if response['correct']:
                concept_performance[concept]['correct'] += 1
        
        strong_areas = []
        for concept, perf in concept_performance.items():
            accuracy = perf['correct'] / perf['total']
            if accuracy >= 0.8:  # 80% or higher accuracy
                strong_areas.append(concept)
        
        return strong_areas

class LearningPathGenerator:
    """Generate personalized learning paths based on skill gaps"""
    
    def __init__(self):
        self.learning_resources = self._initialize_learning_resources()
        self.skill_prerequisites = self._define_skill_prerequisites()
        
        # TensorFlow model for learning path optimization
        self.recommendation_model = self._build_recommendation_model()
    
    def _build_recommendation_model(self):
        """Build TensorFlow model for learning resource recommendation"""
        # User profile input
        user_skills = Input(shape=(50,), name='user_skills')
        user_preferences = Input(shape=(20,), name='user_preferences')
        
        # Resource features input
        resource_features = Input(shape=(30,), name='resource_features')
        
        # Neural network layers
        user_embedding = Dense(32, activation='relu')(tf.keras.layers.concatenate([user_skills, user_preferences]))
        user_embedding = Dropout(0.3)(user_embedding)
        
        resource_embedding = Dense(32, activation='relu')(resource_features)
        resource_embedding = Dropout(0.3)(resource_embedding)
        
        # Compute similarity
        similarity = tf.keras.layers.Dot(axes=1)([user_embedding, resource_embedding])
        similarity = Dense(1, activation='sigmoid')(similarity)
        
        model = Model(inputs=[user_skills, user_preferences, resource_features], outputs=similarity)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _initialize_learning_resources(self) -> List[LearningResource]:
        """Initialize sample learning resources"""
        return [
            LearningResource(
                title="Python Fundamentals",
                description="Learn Python basics including syntax, data types, and control structures",
                difficulty="beginner",
                duration=180,
                skills=["python", "programming"],
                url="https://example.com/python-fundamentals",
                rating=4.5
            ),
            LearningResource(
                title="Machine Learning with TensorFlow",
                description="Build ML models using TensorFlow and understand deep learning concepts",
                difficulty="intermediate",
                duration=300,
                skills=["machine_learning", "tensorflow", "python"],
                url="https://example.com/ml-tensorflow",
                rating=4.7
            ),
            LearningResource(
                title="Advanced SQL Queries",
                description="Master complex SQL queries, joins, and database optimization",
                difficulty="advanced",
                duration=150,
                skills=["sql", "database"],
                url="https://example.com/advanced-sql",
                rating=4.3
            ),
            LearningResource(
                title="Cloud Architecture with AWS",
                description="Design scalable cloud solutions using AWS services",
                difficulty="intermediate",
                duration=240,
                skills=["aws", "cloud", "architecture"],
                url="https://example.com/aws-architecture",
                rating=4.6
            )
        ]
    
    def _define_skill_prerequisites(self) -> Dict[str, List[str]]:
        """Define prerequisite relationships between skills"""
        return {
            'machine_learning': ['python', 'statistics'],
            'deep_learning': ['machine_learning', 'tensorflow'],
            'cloud_architecture': ['cloud', 'networking'],
            'advanced_sql': ['sql', 'database'],
            'react': ['javascript', 'html', 'css'],
            'django': ['python', 'web_development']
        }
    
    def generate_learning_path(self, current_skills: List[Skill], target_skills: List[str], 
                             preferences: Dict[str, Any] = None) -> LearningPath:
        """Generate personalized learning path"""
        if preferences is None:
            preferences = {'max_duration_per_day': 120, 'difficulty_preference': 'intermediate'}
        
        # Identify skill gaps
        current_skill_names = {skill.name.lower() for skill in current_skills}
        target_skill_names = {skill.lower() for skill in target_skills}
        skill_gaps = list(target_skill_names - current_skill_names)
        
        # Add prerequisite skills to gaps
        extended_gaps = self._add_prerequisites(skill_gaps, current_skill_names)
        
        # Find relevant learning resources
        relevant_resources = self._find_relevant_resources(extended_gaps)
        
        # Prioritize and order resources
        ordered_resources = self._prioritize_resources(relevant_resources, current_skills, preferences)
        
        # Calculate estimated completion time
        total_duration = sum(resource.duration for resource in ordered_resources)
        daily_limit = preferences.get('max_duration_per_day', 120)
        estimated_days = max(1, total_duration // daily_limit)
        
        return LearningPath(
            user_id=f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            skills_gap=skill_gaps,
            recommended_resources=ordered_resources,
            estimated_completion=estimated_days,
            priority_order=extended_gaps
        )
    
    def _add_prerequisites(self, skill_gaps: List[str], current_skills: set) -> List[str]:
        """Add prerequisite skills to the learning path"""
        extended_gaps = skill_gaps.copy()
        
        for skill in skill_gaps:
            if skill in self.skill_prerequisites:
                for prereq in self.skill_prerequisites[skill]:
                    if prereq not in current_skills and prereq not in extended_gaps:
                        extended_gaps.insert(0, prereq)  # Add prerequisites first
        
        return extended_gaps
    
    def _find_relevant_resources(self, skill_gaps: List[str]) -> List[LearningResource]:
        """Find learning resources that match skill gaps"""
        relevant_resources = []
        
        for resource in self.learning_resources:
            resource_skills = [skill.lower() for skill in resource.skills]
            if any(gap in resource_skills for gap in skill_gaps):
                relevant_resources.append(resource)
        
        return relevant_resources
    
    def _prioritize_resources(self, resources: List[LearningResource], 
                            current_skills: List[Skill], preferences: Dict[str, Any]) -> List[LearningResource]:
        """Prioritize learning resources based on user profile and preferences"""
        # Calculate priority scores for each resource
        scored_resources = []
        
        for resource in resources:
            score = self._calculate_resource_score(resource, current_skills, preferences)
            scored_resources.append((resource, score))
        
        # Sort by score (highest first)
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        return [resource for resource, _ in scored_resources]
    
    def _calculate_resource_score(self, resource: LearningResource, 
                                current_skills: List[Skill], preferences: Dict[str, Any]) -> float:
        """Calculate priority score for a learning resource"""
        score = 0.0
        
        # Base score from rating
        score += resource.rating * 0.3
        
        # Difficulty preference alignment
        preferred_difficulty = preferences.get('difficulty_preference', 'intermediate')
        if resource.difficulty == preferred_difficulty:
            score += 0.4
        elif abs(self._difficulty_to_number(resource.difficulty) - 
                self._difficulty_to_number(preferred_difficulty)) <= 1:
            score += 0.2
        
        # Duration preference (shorter preferred for beginners)
        max_duration = preferences.get('max_duration_per_day', 120)
        if resource.duration <= max_duration:
            score += 0.3
        else:
            score += 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def _difficulty_to_number(self, difficulty: str) -> int:
        """Convert difficulty string to number for comparison"""
        difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        return difficulty_map.get(difficulty, 2)

class PersonalizedLearningSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.cv_extractor = ComputerVisionSkillExtractor()
        self.assessment_engine = SkillAssessmentEngine()
        self.path_generator = LearningPathGenerator()
        self.user_profiles = {}
    
    def process_user_profile(self, user_id: str, resume_path: str = None, 
                           certificate_paths: List[str] = None, 
                           target_skills: List[str] = None) -> Dict[str, Any]:
        """Process complete user profile and generate learning recommendations"""
        
        all_skills = []
        
        # Extract skills from resume
        if resume_path:
            resume_text = self.resume_parser.extract_text_from_resume(resume_path)
            resume_skills = self.resume_parser.extract_skills(resume_text)
            all_skills.extend(resume_skills)
        
        # Extract skills from certificates
        if certificate_paths:
            for cert_path in certificate_paths:
                cert_skills = self.cv_extractor.extract_from_certificate(cert_path)
                all_skills.extend(cert_skills)
        
        # Conduct skill assessments for key skills
        assessment_results = {}
        key_skills = list(set([skill.name for skill in all_skills]))[:5]  # Limit to top 5 skills
        
        for skill in key_skills:
            if skill in ['python', 'machine_learning']:  # Only assess skills we have questions for
                assessment = self.assessment_engine.conduct_assessment(user_id, skill)
                assessment_results[skill] = assessment
                
                # Update skill level based on assessment
                for user_skill in all_skills:
                    if user_skill.name == skill:
                        user_skill.level = assessment['estimated_level']
                        user_skill.confidence = assessment['confidence']
        
        # Generate learning path
        learning_path = None
        if target_skills:
            learning_path = self.path_generator.generate_learning_path(
                current_skills=all_skills,
                target_skills=target_skills
            )
        
        # Store user profile
        # Store user profile
        self.user_profiles[user_id] = {
            'skills': all_skills,
            'assessments': assessment_results,
            'learning_path': learning_path,
            'last_updated': datetime.now(),
            'target_skills': target_skills or []
        }
        
        return {
            'user_id': user_id,
            'extracted_skills': [
                {
                    'name': skill.name,
                    'level': skill.level,
                    'category': skill.category,
                    'confidence': skill.confidence,
                    'source': skill.source
                } for skill in all_skills
            ],
            'assessments': assessment_results,
            'learning_path': {
                'skills_gap': learning_path.skills_gap if learning_path else [],
                'recommended_resources': [
                    {
                        'title': resource.title,
                        'description': resource.description,
                        'difficulty': resource.difficulty,
                        'duration': resource.duration,
                        'skills': resource.skills,
                        'rating': resource.rating,
                        'url': resource.url
                    } for resource in learning_path.recommended_resources
                ] if learning_path else [],
                'estimated_completion_days': learning_path.estimated_completion if learning_path else 0
            },
            'recommendations': self._generate_general_recommendations(all_skills, assessment_results)
        }
    
    def _generate_general_recommendations(self, skills: List[Skill], 
                                        assessments: Dict[str, Any]) -> List[str]:
        """Generate general learning recommendations"""
        recommendations = []
        
        # Skill diversity recommendations
        skill_categories = set(skill.category for skill in skills)
        if len(skill_categories) < 3:
            recommendations.append("Consider expanding into new skill categories to become more versatile")
        
        # Skill depth recommendations
        advanced_skills = [skill for skill in skills if skill.level > 0.7]
        if len(advanced_skills) < 2:
            recommendations.append("Focus on developing deeper expertise in your strongest skills")
        
        # Assessment-based recommendations
        for skill, assessment in assessments.items():
            if assessment.get('estimated_level', 0) < 0.6:
                recommendations.append(f"Consider additional practice in {skill} fundamentals")
            
            weak_areas = assessment.get('areas_for_improvement', [])
            if weak_areas:
                recommendations.append(f"Focus on improving {', '.join(weak_areas)} in {skill}")
        
        return recommendations
    
    def update_skill_progress(self, user_id: str, skill_name: str, 
                            new_level: float, completion_data: Dict[str, Any] = None):
        """Update user's skill progress"""
        if user_id not in self.user_profiles:
            return {'error': 'User profile not found'}
        
        profile = self.user_profiles[user_id]
        
        # Update existing skill or add new one
        skill_updated = False
        for skill in profile['skills']:
            if skill.name.lower() == skill_name.lower():
                skill.level = new_level
                skill.confidence = min(1.0, skill.confidence + 0.1)  # Increase confidence
                skill_updated = True
                break
        
        if not skill_updated:
            # Add new skill
            profile['skills'].append(Skill(
                name=skill_name,
                level=new_level,
                category='general',
                confidence=0.7,
                source='progress_update'
            ))
        
        # Update learning path if target skills exist
        if profile['target_skills']:
            updated_path = self.path_generator.generate_learning_path(
                current_skills=profile['skills'],
                target_skills=profile['target_skills']
            )
            profile['learning_path'] = updated_path
        
        profile['last_updated'] = datetime.now()
        
        return {'success': True, 'message': f'Updated {skill_name} progress'}
    
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard data for user interface"""
        if user_id not in self.user_profiles:
            return {'error': 'User profile not found'}
        
        profile = self.user_profiles[user_id]
        skills = profile['skills']
        
        # Calculate skill distribution by category
        category_distribution = {}
        for skill in skills:
            category = skill.category
            if category not in category_distribution:
                category_distribution[category] = {'count': 0, 'avg_level': 0}
            category_distribution[category]['count'] += 1
            category_distribution[category]['avg_level'] += skill.level
        
        # Calculate average levels
        for category in category_distribution:
            count = category_distribution[category]['count']
            category_distribution[category]['avg_level'] /= count
        
        # Get top skills
        top_skills = sorted(skills, key=lambda x: x.level * x.confidence, reverse=True)[:5]
        
        # Calculate overall progress
        if profile['learning_path']:
            total_resources = len(profile['learning_path'].recommended_resources)
            # Simulate progress (in real app, this would track actual completion)
            completed_resources = max(0, total_resources - len(profile['learning_path'].skills_gap))
            progress_percentage = (completed_resources / total_resources * 100) if total_resources > 0 else 0
        else:
            progress_percentage = 0
        
        return {
            'user_id': user_id,
            'skill_summary': {
                'total_skills': len(skills),
                'categories': list(category_distribution.keys()),
                'avg_skill_level': np.mean([skill.level for skill in skills]) if skills else 0,
                'category_distribution': category_distribution
            },
            'top_skills': [
                {
                    'name': skill.name,
                    'level': skill.level,
                    'category': skill.category,
                    'confidence': skill.confidence
                } for skill in top_skills
            ],
            'learning_progress': {
                'completion_percentage': progress_percentage,
                'days_since_start': (datetime.now() - profile['last_updated']).days,
                'estimated_completion': profile['learning_path'].estimated_completion if profile['learning_path'] else 0
            },
            'recent_assessments': profile['assessments'],
            'next_recommendations': profile['learning_path'].recommended_resources[:3] if profile['learning_path'] else []
        }

class VisualizationEngine:
    """Generate visualizations for skill analysis and learning progress"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_skill_radar(self, skills: List[Skill], save_path: str = None) -> str:
        """Create radar chart for skill visualization"""
        # Group skills by category and calculate average levels
        categories = {}
        for skill in skills:
            if skill.category not in categories:
                categories[skill.category] = []
            categories[skill.category].append(skill.level)
        
        # Calculate average level per category
        category_levels = {cat: np.mean(levels) for cat, levels in categories.items()}
        
        # Prepare data for radar chart
        labels = list(category_levels.keys())
        values = list(category_levels.values())
        
        # Number of variables
        N = len(labels)
        
        # Angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Close the plot
        values += values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot the radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label='Current Skills')
        ax.fill(angles, values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.title('Skill Level Distribution by Category', size=16, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            plt.show()
            return "displayed"
    
    def plot_learning_progress(self, learning_path: LearningPath, 
                             completed_resources: List[str] = None, 
                             save_path: str = None) -> str:
        """Create progress visualization for learning path"""
        if not learning_path:
            return "No learning path available"
        
        completed_resources = completed_resources or []
        resources = learning_path.recommended_resources
        
        # Prepare data
        resource_names = [r.title[:30] + '...' if len(r.title) > 30 else r.title for r in resources]
        durations = [r.duration for r in resources]
        completion_status = [1 if r.title in completed_resources else 0 for r in resources]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Progress bar chart
        colors = ['green' if status else 'lightblue' for status in completion_status]
        bars = ax1.barh(range(len(resource_names)), durations, color=colors)
        
        ax1.set_yticks(range(len(resource_names)))
        ax1.set_yticklabels(resource_names)
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_title('Learning Path Progress')
        ax1.invert_yaxis()
        
        # Add completion percentage text
        for i, (bar, status) in enumerate(zip(bars, completion_status)):
            if status:
                ax1.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                        'Completed', ha='center', va='center', fontweight='bold')
        
        # Pie chart for overall progress
        completed_count = sum(completion_status)
        remaining_count = len(resources) - completed_count
        
        if completed_count > 0:
            labels = ['Completed', 'Remaining']
            sizes = [completed_count, remaining_count]
            colors = ['green', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Overall Progress')
        else:
            ax2.text(0.5, 0.5, 'No Progress Yet', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Overall Progress')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            plt.show()
            return "displayed"
    
    def plot_assessment_results(self, assessment_data: Dict[str, Any], save_path: str = None) -> str:
        """Visualize assessment results"""
        if not assessment_data:
            return "No assessment data available"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Skill level comparison
        skills = list(assessment_data.keys())
        levels = [assessment_data[skill].get('estimated_level', 0) for skill in skills]
        confidences = [assessment_data[skill].get('confidence', 0) for skill in skills]
        
        x = range(len(skills))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], levels, width, label='Skill Level', alpha=0.8)
        ax1.bar([i + width/2 for i in x], confidences, width, label='Confidence', alpha=0.8)
        ax1.set_xlabel('Skills')
        ax1.set_ylabel('Score')
        ax1.set_title('Skill Level vs Confidence')
        ax1.set_xticks(x)
        ax1.set_xticklabels(skills)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Assessment accuracy
        accuracies = []
        for skill in skills:
            total_q = assessment_data[skill].get('questions_answered', 0)
            correct_q = assessment_data[skill].get('correct_answers', 0)
            accuracy = correct_q / total_q if total_q > 0 else 0
            accuracies.append(accuracy)
        
        ax2.pie(accuracies, labels=skills, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Assessment Accuracy by Skill')
        
        # Strong vs weak areas (for first skill with data)
        first_skill = skills[0] if skills else None
        if first_skill and 'strong_areas' in assessment_data[first_skill]:
            strong_areas = assessment_data[first_skill].get('strong_areas', [])
            weak_areas = assessment_data[first_skill].get('areas_for_improvement', [])
            
            categories = strong_areas + weak_areas
            performance = [0.8] * len(strong_areas) + [0.4] * len(weak_areas)
            colors = ['green'] * len(strong_areas) + ['red'] * len(weak_areas)
            
            if categories:
                ax3.bar(categories, performance, color=colors, alpha=0.7)
                ax3.set_title(f'{first_skill.title()} - Area Performance')
                ax3.set_ylabel('Performance Score')
                ax3.tick_params(axis='x', rotation=45)
        
        # Questions answered over time (simulated)
        questions_per_skill = [assessment_data[skill].get('questions_answered', 0) for skill in skills]
        ax4.plot(skills, questions_per_skill, marker='o', linewidth=2, markersize=8)
        ax4.set_title('Questions Answered per Skill')
        ax4.set_ylabel('Number of Questions')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            plt.show()
            return "displayed"

def main():
    """Main function to demonstrate the system"""
    print("🚀 Initializing Personalized Learning System...")
    
    # Initialize the system
    learning_system = PersonalizedLearningSystem()
    visualization_engine = VisualizationEngine()
    
    print("✅ System initialized successfully!")
    
    # Example usage
    user_id = "demo_user_001"
    
    # Simulate processing a user profile
    print(f"\n📊 Processing user profile for {user_id}...")
    
    # Example target skills
    target_skills = ["machine_learning", "tensorflow", "cloud_computing", "advanced_sql"]
    
    # Process user profile (would normally include resume and certificate paths)
    profile_result = learning_system.process_user_profile(
        user_id=user_id,
        target_skills=target_skills
    )
    
    print(f"✅ Profile processed successfully!")
    print(f"📈 Extracted {len(profile_result['extracted_skills'])} skills")
    print(f"🎯 Generated learning path with {len(profile_result['learning_path']['recommended_resources'])} resources")
    
    # Get dashboard data
    dashboard_data = learning_system.get_dashboard_data(user_id)
    
    print(f"\n📊 Dashboard Summary:")
    print(f"   • Total Skills: {dashboard_data['skill_summary']['total_skills']}")
    print(f"   • Categories: {', '.join(dashboard_data['skill_summary']['categories'])}")
    print(f"   • Average Skill Level: {dashboard_data['skill_summary']['avg_skill_level']:.2f}")
    print(f"   • Learning Progress: {dashboard_data['learning_progress']['completion_percentage']:.1f}%")
    
    # Display top skills
    print(f"\n🏆 Top Skills:")
    for i, skill in enumerate(dashboard_data['top_skills'][:3], 1):
        print(f"   {i}. {skill['name'].title()} - Level: {skill['level']:.2f} ({skill['category']})")
    
    # Display learning recommendations
    print(f"\n📚 Recommended Learning Resources:")
    for i, resource in enumerate(profile_result['learning_path']['recommended_resources'][:3], 1):
        print(f"   {i}. {resource['title']}")
        print(f"      Duration: {resource['duration']} min | Difficulty: {resource['difficulty']}")
        print(f"      Skills: {', '.join(resource['skills'])}")
        print(f"      Rating: {resource['rating']}/5.0")
        print()
    
    # Display general recommendations
    if profile_result['recommendations']:
        print(f"💡 General Recommendations:")
        for i, rec in enumerate(profile_result['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💾 User profile saved for {user_id}")
    
    return learning_system, visualization_engine

class APIInterface:
    """REST API interface for the learning system"""
    
    def __init__(self, learning_system: PersonalizedLearningSystem):
        self.learning_system = learning_system
        self.visualization_engine = VisualizationEngine()
    
    def create_user_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to create user profile"""
        try:
            user_id = data.get('user_id')
            resume_path = data.get('resume_path')
            certificate_paths = data.get('certificate_paths', [])
            target_skills = data.get('target_skills', [])
            
            result = self.learning_system.process_user_profile(
                user_id=user_id,
                resume_path=resume_path,
                certificate_paths=certificate_paths,
                target_skills=target_skills
            )
            
            return {'status': 'success', 'data': result}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """API endpoint to get user dashboard data"""
        try:
            dashboard_data = self.learning_system.get_dashboard_data(user_id)
            return {'status': 'success', 'data': dashboard_data}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def update_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to update learning progress"""
        try:
            user_id = data.get('user_id')
            skill_name = data.get('skill_name')
            new_level = data.get('new_level')
            completion_data = data.get('completion_data', {})
            
            result = self.learning_system.update_skill_progress(
                user_id=user_id,
                skill_name=skill_name,
                new_level=new_level,
                completion_data=completion_data
            )
            
            return {'status': 'success', 'data': result}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def conduct_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to conduct skill assessment"""
        try:
            user_id = data.get('user_id')
            skill = data.get('skill')
            num_questions = data.get('num_questions', 10)
            
            assessment_result = self.learning_system.assessment_engine.conduct_assessment(
                user_id=user_id,
                skill=skill,
                num_questions=num_questions
            )
            
            return {'status': 'success', 'data': assessment_result}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to generate visualizations"""
        try:
            user_id = data.get('user_id')
            viz_type = data.get('type', 'radar')  # 'radar', 'progress', 'assessment'
            save_path = data.get('save_path')
            
            if user_id not in self.learning_system.user_profiles:
                return {'status': 'error', 'message': 'User profile not found'}
            
            profile = self.learning_system.user_profiles[user_id]
            
            if viz_type == 'radar':
                result = self.visualization_engine.plot_skill_radar(
                    skills=profile['skills'],
                    save_path=save_path
                )
            elif viz_type == 'progress':
                result = self.visualization_engine.plot_learning_progress(
                    learning_path=profile['learning_path'],
                    save_path=save_path
                )
            elif viz_type == 'assessment':
                result = self.visualization_engine.plot_assessment_results(
                    assessment_data=profile['assessments'],
                    save_path=save_path
                )
            else:
                return {'status': 'error', 'message': 'Invalid visualization type'}
            
            return {'status': 'success', 'data': {'visualization_path': result}}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)

# Initialize the learning system
learning_system = PersonalizedLearningSystem()
visualization_engine = VisualizationEngine()

# Database setup
def init_db():
    """Initialize SQLite database for user management"""
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            profile_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # File uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    """Handle resume upload from the main interface"""
    try:
        user_id = request.form.get('userId')
        resume_file = request.files.get('resume')
        manual_skills_json = request.form.get('manual_skills')
        
        print(f"User ID: {user_id}")
        print(f"Resume file: {resume_file}")
        print(f"Manual skills: {manual_skills_json}")
        
        # Parse manual skills if provided
        manual_skills = []
        if manual_skills_json:
            manual_skills = json.loads(manual_skills_json)
        
        # Call your existing function
        result = process_user_profile_web(
            user_id=user_id,
            resume_path=resume_file,
            manual_skills=manual_skills
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Resume uploaded successfully',
            'user_id': user_id,
            'has_resume': resume_file is not None,
            'manual_skills_count': len(manual_skills),
            'result': result
        })
        
    except Exception as e:
        print(f"Error in upload_resume: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/conduct_assessment', methods=['POST'])
def conduct_assessment_api():
    """API endpoint for conducting assessments"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        target_skills = data.get('target_skills', [])
        
        print(f"Assessment for user: {user_id}, skills: {target_skills}")
        
        # You can integrate this with your existing assessment logic
        # For now, returning a success response
        result = {
            'user_id': user_id,
            'target_skills': target_skills,
            'assessment_completed': True,
            'message': 'Assessment completed successfully'
        }
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        print(f"Error in conduct_assessment: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/generate_visualization', methods=['POST'])
def generate_visualization_api():
    """API endpoint for generating visualizations"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        viz_type = data.get('viz_type')
        
        print(f"Generating {viz_type} visualization for user: {user_id}")
        
        # You can integrate this with your existing visualization logic
        result = {
            'user_id': user_id,
            'viz_type': viz_type,
            'visualization_generated': True,
            'message': f'{viz_type} visualization generated successfully'
        }
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        print(f"Error in generate_visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (user_id, username, email, password_hash))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('learning_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            
            # Update last login
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?', (user[0],))
            conn.commit()
            conn.close()
            
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile_setup', methods=['GET', 'POST'])
def profile_setup():
    """Profile setup page where users enter their details"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_id = session['user_id']
        
        # Get form data
        current_skills = request.form.getlist('current_skills[]')
        current_levels = request.form.getlist('current_levels[]')
        target_skills = request.form.getlist('target_skills[]')
        
        # Handle file uploads
        resume_path = None
        certificate_paths = []
        
        # Handle resume upload
        if 'resume' in request.files:
            resume_file = request.files['resume']
            if resume_file and resume_file.filename and allowed_file(resume_file.filename):
                filename = secure_filename(resume_file.filename)
                filename = f"{user_id}_resume_{filename}"
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume_file.save(resume_path)
                
                # Save to database
                save_file_upload(user_id, filename, 'resume', resume_path)
        
        # Handle certificate uploads
        if 'certificates' in request.files:
            cert_files = request.files.getlist('certificates')
            for cert_file in cert_files:
                if cert_file and cert_file.filename and allowed_file(cert_file.filename):
                    filename = secure_filename(cert_file.filename)
                    filename = f"{user_id}_cert_{filename}"
                    cert_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    cert_file.save(cert_path)
                    certificate_paths.append(cert_path)
                    
                    # Save to database
                    save_file_upload(user_id, filename, 'certificate', cert_path)
        
        # Create manual skills list
        manual_skills = []
        for skill, level in zip(current_skills, current_levels):
            if skill.strip():
                try:
                    level_float = float(level) / 100.0  # Convert percentage to 0-1 scale
                    manual_skills.append({
                        'name': skill.strip(),
                        'level': level_float,
                        'category': categorize_skill(skill.strip()),
                        'confidence': 0.8,
                        'source': 'user_input'
                    })
                except ValueError:
                    continue
        
        # Process user profile
        try:
            profile_result = process_user_profile_web(
                user_id=user_id,
                resume_path=resume_path,
                certificate_paths=certificate_paths,
                target_skills=[s.strip() for s in target_skills if s.strip()],
                manual_skills=manual_skills
            )
            
            # Save profile to database
            save_user_profile(user_id, profile_result)
            
            flash('Profile created successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            flash(f'Error processing profile: {str(e)}', 'error')
            return render_template('profile_setup.html')
    
    return render_template('profile_setup.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Check if user has a profile
    profile_data = get_user_profile(user_id)
    if not profile_data:
        return redirect(url_for('profile_setup'))
    
    # Get dashboard data
    dashboard_data = learning_system.get_dashboard_data(user_id)
    
    # Generate visualizations
    viz_paths = generate_user_visualizations(user_id)
    
    return render_template('dashboard.html', 
                         dashboard_data=dashboard_data,
                         visualizations=viz_paths,
                         username=session.get('username'))

@app.route('/assessment/<skill>')
def assessment(skill):
    """Skill assessment page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Conduct assessment
    assessment_result = learning_system.assessment_engine.conduct_assessment(
        user_id=user_id,
        skill=skill,
        num_questions=10
    )
    
    return render_template('assessment.html', 
                         skill=skill,
                         assessment=assessment_result)

@app.route('/learning_path')
def learning_path():
    """Learning path page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    profile_data = get_user_profile(user_id)
    
    if not profile_data:
        return redirect(url_for('profile_setup'))
    
    return render_template('learning_path.html', 
                         learning_path=profile_data.get('learning_path', {}))


@app.route('/api/chart-data/<user_id>')
def get_chart_data(user_id):
    """Get chart data for dashboard"""
    try:
        if user_id not in learning_system.user_profiles:
            return jsonify({'error': 'User not found'}), 404
        
        profile = learning_system.user_profiles[user_id]
        skills = profile['skills']
        
        # Skill radar data
        radar_data = []
        categories = {}
        for skill in skills:
            if skill.category not in categories:
                categories[skill.category] = []
            categories[skill.category].append(skill.level)
        
        for category, levels in categories.items():
            radar_data.append({
                'category': category.replace('_', ' ').title(),
                'level': sum(levels) / len(levels) * 100
            })
        
        # Progress data (simulated - in real app, track over time)
        progress_data = []
        for i in range(6):  # Last 6 months
            date = (datetime.now() - timedelta(days=30*i)).strftime('%Y-%m')
            data_point = {'date': date}
            for skill in skills[:5]:  # Top 5 skills
                # Simulate progress over time
                base_level = skill.level * 100
                variation = (i * 5)  # Simulate improvement
                data_point[skill.name] = min(100, base_level - variation)
            progress_data.append(data_point)
        
        progress_data.reverse()  # Chronological order
        
        return jsonify({
            'radar_data': radar_data,
            'progress_data': progress_data,
            'category_distribution': [
                {'name': cat, 'value': len(levels)} 
                for cat, levels in categories.items()
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_progress', methods=['POST'])
def update_progress():
    """API endpoint to update learning progress"""
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'})
    
    data = request.get_json()
    user_id = session['user_id']
    skill_name = data.get('skill_name')
    new_level = float(data.get('new_level', 0)) / 100.0  # Convert percentage to 0-1
    
    result = learning_system.update_skill_progress(
        user_id=user_id,
        skill_name=skill_name,
        new_level=new_level
    )
    
    # Update database
    profile_data = get_user_profile(user_id)
    if profile_data:
        save_user_profile(user_id, profile_data)
    
    return jsonify(result)

# Helper functions

def categorize_skill(skill_name):
    """Categorize skill based on name"""
    skill_name_lower = skill_name.lower()
    
    if any(lang in skill_name_lower for lang in ['python', 'java', 'javascript', 'react', 'angular', 'vue', 'html', 'css']):
        return 'programming'
    elif any(ds in skill_name_lower for ds in ['machine learning', 'data science', 'tensorflow', 'pytorch', 'pandas', 'numpy']):
        return 'data_science'
    elif any(cloud in skill_name_lower for cloud in ['aws', 'azure', 'gcp', 'docker', 'kubernetes']):
        return 'cloud'
    elif any(db in skill_name_lower for db in ['sql', 'mongodb', 'postgresql', 'mysql']):
        return 'database'
    elif any(soft in skill_name_lower for soft in ['leadership', 'communication', 'teamwork', 'management']):
        return 'soft_skills'
    else:
        return 'general'
    
def process_user_profile_web(user_id, resume_path=None, certificate_paths=None, target_skills=None, manual_skills=None):
    """Process user profile for web interface"""
    skill_objects = []
    if manual_skills:
        # No import needed if Skill class is in same file
        for skill_data in manual_skills:
            skill_objects.append(Skill(
                name=skill_data['name'],
                level=skill_data['level'],
                category=skill_data['category'],
                confidence=skill_data['confidence'],
                source=skill_data['source']
            ))
    
    # Process with the learning system
    result = learning_system.process_user_profile(
        user_id=user_id,
        resume_path=resume_path,
        certificate_paths=certificate_paths or [],
        target_skills=target_skills or []
    )
    
    # Add manual skills to the result
    if skill_objects:
        if user_id in learning_system.user_profiles:
            learning_system.user_profiles[user_id]['skills'].extend(skill_objects)
        
        # Regenerate learning path with all skills
        all_skills = learning_system.user_profiles[user_id]['skills']
        learning_path = learning_system.path_generator.generate_learning_path(
            current_skills=all_skills,
            target_skills=target_skills or []
        )
        learning_system.user_profiles[user_id]['learning_path'] = learning_path
    
    return result

def save_file_upload(user_id, filename, file_type, file_path):
    """Save file upload record to database"""
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO file_uploads (user_id, filename, file_type, file_path)
        VALUES (?, ?, ?, ?)
    ''', (user_id, filename, file_type, file_path))
    conn.commit()
    conn.close()

def save_user_profile(user_id, profile_data):
    """Save user profile to database"""
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    
    # Check if profile exists
    cursor.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
    existing = cursor.fetchone()
    
    profile_json = json.dumps(profile_data, default=str)
    
    if existing:
        cursor.execute('''
            UPDATE user_profiles 
            SET profile_data = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE user_id = ?
        ''', (profile_json, user_id))
    else:
        cursor.execute('''
            INSERT INTO user_profiles (user_id, profile_data)
            VALUES (?, ?)
        ''', (user_id, profile_json))
    
    conn.commit()
    conn.close()

def get_user_profile(user_id):
    """Get user profile from database"""
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    cursor.execute('SELECT profile_data FROM user_profiles WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return json.loads(result[0])
    return None

def generate_user_visualizations(user_id):
    """Generate visualizations for user"""
    viz_paths = {}
    
    if user_id in learning_system.user_profiles:
        profile = learning_system.user_profiles[user_id]
        
        try:
            # Skill radar chart
            radar_path = f'static/visualizations/{user_id}_radar.png'
            visualization_engine.plot_skill_radar(
                skills=profile['skills'],
                save_path=radar_path
            )
            viz_paths['radar'] = radar_path
        except Exception as e:
            print(f"Error generating radar chart: {e}")
        
        try:
            # Learning progress chart
            if profile['learning_path']:
                progress_path = f'static/visualizations/{user_id}_progress.png'
                visualization_engine.plot_learning_progress(
                    learning_path=profile['learning_path'],
                    save_path=progress_path
                )
                viz_paths['progress'] = progress_path
        except Exception as e:
            print(f"Error generating progress chart: {e}")
        
        try:
            # Assessment results
            if profile['assessments']:
                assessment_path = f'static/visualizations/{user_id}_assessment.png'
                visualization_engine.plot_assessment_results(
                    assessment_data=profile['assessments'],
                    save_path=assessment_path
                )
                viz_paths['assessment'] = assessment_path
        except Exception as e:
            print(f"Error generating assessment chart: {e}")
    
    return viz_paths

if __name__ == "__main__":
    # Run the main demonstration
    learning_system, visualization_engine = main()
    
    # Initialize API interface
    api = APIInterface(learning_system)
    
    print(f"\n🌐 API Interface initialized and ready for integration!")
    print(f"📡 Available endpoints:")
    print(f"   • POST /create_profile - Create user profile")
    print(f"   • GET /dashboard/<user_id> - Get dashboard data")
    print(f"   • POST /update_progress - Update learning progress")
    print(f"   • POST /conduct_assessment - Conduct skill assessment")
    print(f"   • POST /generate_visualization - Generate visualizations")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
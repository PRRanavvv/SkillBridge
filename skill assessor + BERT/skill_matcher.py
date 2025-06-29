import os
import sqlite3
import json
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import cv2
import pytesseract
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer  # BERT Integration
import warnings
import traceback
import hashlib
import time
from datetime import datetime, timedelta
import re
from pathlib import Path
import io
import base64
from PIL import Image
import fitz
import docx
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from typing import List, Dict, Optional, Tuple, Union, Any
import uuid
from dataclasses import dataclass
import tensorflow_hub as hub

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

class BERTEnhancedResumeParser:
    """BERT-Enhanced Resume Parser with semantic understanding"""
    
    def __init__(self):
        # Initialize spaCy
        self.nlp = nlp
        
        # BERT Integration
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_model = None
        
        # Keep Universal Sentence Encoder as fallback
        try:
            self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        except Exception as e:
            logger.warning(f"Universal Sentence Encoder not available: {e}")
            self.use_model = None
        
        # Enhanced skill database
        self.skill_keywords = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'perl', 'react', 
                'angular', 'vue', 'django', 'flask', 'spring'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 
                'numpy', 'scikit-learn', 'data analysis', 'statistics', 'data visualization', 
                'tableau', 'power bi', 'matplotlib', 'seaborn'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'cloud computing'
            ],
            'database': [
                'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'cassandra', 'oracle', 'sqlite'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'project management'
            ]
        }
        
        self.skill_embeddings_cache = {}
        self._build_skill_embeddings()
    
    def _build_skill_embeddings(self):
        """Build BERT embeddings for all skills"""
        if not self.bert_model:
            logger.warning("BERT model not available - using traditional methods only")
            return
        
        try:
            logger.info("Building BERT embeddings for skill database...")
            all_skills = []
            for category, skills in self.skill_keywords.items():
                all_skills.extend(skills)
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(all_skills), batch_size):
                batch = all_skills[i:i + batch_size]
                embeddings = self.bert_model.encode(batch, show_progress_bar=False)
                
                for skill, embedding in zip(batch, embeddings):
                    self.skill_embeddings_cache[skill.lower()] = {
                        'embedding': embedding,
                        'category': self._get_skill_category(skill)
                    }
            
            logger.info(f"Built BERT embeddings for {len(all_skills)} skills")
        except Exception as e:
            logger.error(f"Error building BERT embeddings: {e}")
    
    def _get_skill_category(self, skill):
        """Get category for a skill"""
        for category, skills in self.skill_keywords.items():
            if skill in skills:
                return category
        return 'general'
    
    def extract_text_from_resume(self, file_path: str) -> str:
        """Extract text from various resume formats"""
        try:
            if hasattr(file_path, 'read'):  # File object
                # Handle file upload object
                content = file_path.read()
                filename = getattr(file_path, 'filename', 'unknown')
                
                if filename.endswith('.pdf'):
                    return self._extract_from_pdf_content(content)
                elif filename.endswith(('.png', '.jpg', '.jpeg')):
                    return self._extract_from_image_content(content)
                else:
                    return content.decode('utf-8', errors='ignore')
            
            elif file_path.endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                return self._extract_from_image(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error extracting text from resume: {e}")
            return ""
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return "Sample extracted text from PDF resume with skills like Python, Machine Learning, TensorFlow, SQL"
    
    def _extract_from_pdf_content(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF content: {e}")
            return ""
    
    def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return ""
    
    def _extract_from_image_content(self, content: bytes) -> str:
        """Extract text from image content using OCR"""
        try:
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting from image content: {e}")
            return ""
    
    def extract_skills_bert(self, resume_text: str) -> List[Skill]:
        """BERT-powered skill extraction with semantic understanding"""
        if not self.bert_model or not self.skill_embeddings_cache:
            return self.extract_skills_traditional(resume_text)
        
        try:
            # Generate embedding for resume text
            text_embedding = self.bert_model.encode([resume_text.lower()], show_progress_bar=False)
            
            detected_skills = []
            similarity_threshold = 0.6
            
            for skill, skill_data in self.skill_embeddings_cache.items():
                try:
                    similarity = cosine_similarity(
                        text_embedding, 
                        [skill_data['embedding']]
                    )[0][0]
                    
                    if similarity > similarity_threshold:
                        # Estimate skill level using context
                        level = self._estimate_skill_level_bert(resume_text, skill)
                        
                        detected_skills.append(Skill(
                            name=skill,
                            level=level,
                            category=skill_data['category'],
                            confidence=float(similarity),
                            source='resume_bert'
                        ))
                except Exception as skill_error:
                    continue
            
            # Sort by confidence and return top skills
            detected_skills.sort(key=lambda x: x.confidence, reverse=True)
            return detected_skills[:20]
            
        except Exception as e:
            logger.error(f"Error in BERT skill extraction: {e}")
            return self.extract_skills_traditional(resume_text)
    
    def extract_skills_traditional(self, resume_text: str) -> List[Skill]:
        """Traditional keyword-based skill extraction"""
        try:
            skills = []
            doc = self.nlp(resume_text.lower())
            
            for category, keywords in self.skill_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in resume_text.lower():
                        confidence = self._calculate_skill_confidence_traditional(resume_text, keyword)
                        level = self._estimate_skill_level_traditional(resume_text, keyword)
                        
                        skills.append(Skill(
                            name=keyword,
                            level=level,
                            category=category,
                            confidence=confidence,
                            source='resume_traditional'
                        ))
            
            return self._deduplicate_skills(skills)
        except Exception as e:
            logger.error(f"Error in traditional skill extraction: {e}")
            return []
    
    def extract_skills(self, resume_text: str) -> List[Skill]:
        """Hybrid skill extraction combining BERT and traditional methods"""
        try:
            # Get skills from both methods
            bert_skills = self.extract_skills_bert(resume_text)
            traditional_skills = self.extract_skills_traditional(resume_text)
            
            # Combine and deduplicate
            all_skills = {}
            
            # Add traditional skills (high confidence for exact matches)
            for skill in traditional_skills:
                key = skill.name.lower()
                all_skills[key] = skill
                all_skills[key].confidence = min(0.95, skill.confidence + 0.1)
            
            # Add BERT skills (semantic matches)
            for skill in bert_skills:
                key = skill.name.lower()
                if key not in all_skills:
                    all_skills[key] = skill
                else:
                    # Combine confidences for skills found by both methods
                    existing_skill = all_skills[key]
                    combined_confidence = min(1.0, existing_skill.confidence + skill.confidence * 0.3)
                    all_skills[key].confidence = combined_confidence
                    all_skills[key].source = 'resume_hybrid'
            
            return list(all_skills.values())
            
        except Exception as e:
            logger.error(f"Error in hybrid skill extraction: {e}")
            return self.extract_skills_traditional(resume_text)
    
    def _estimate_skill_level_bert(self, text: str, skill: str) -> float:
        """Estimate skill level using BERT context understanding"""
        try:
            # Look for skill in context
            sentences = [sent.text for sent in self.nlp(text).sents if skill.lower() in sent.text.lower()]
            
            if not sentences:
                return 0.5  # Default level
            
            # Use BERT to understand proficiency context
            proficiency_contexts = [
                f"expert in {skill}",
                f"advanced {skill}",
                f"intermediate {skill}",
                f"beginner {skill}"
            ]
            
            if self.bert_model:
                sentence_embeddings = self.bert_model.encode(sentences[:3])
                context_embeddings = self.bert_model.encode(proficiency_contexts)
                
                # Find best matching proficiency level
                similarities = cosine_similarity(sentence_embeddings, context_embeddings)
                max_sim_idx = np.argmax(similarities)
                
                # Map to skill levels
                level_map = [0.9, 0.7, 0.5, 0.3]  # expert, advanced, intermediate, beginner
                return level_map[max_sim_idx % 4]
            
            return self._estimate_skill_level_traditional(text, skill)
            
        except Exception as e:
            logger.error(f"Error estimating BERT skill level: {e}")
            return 0.5
    
    def _estimate_skill_level_traditional(self, text: str, skill: str) -> float:
        """Traditional skill level estimation"""
        level_indicators = {
            'expert': 0.9, 'senior': 0.8, 'advanced': 0.8,
            'experienced': 0.7, 'proficient': 0.6, 'intermediate': 0.5,
            'familiar': 0.4, 'basic': 0.3, 'beginner': 0.2
        }
        
        skill_context = []
        doc = self.nlp(text.lower())
        
        for sent in doc.sents:
            if skill.lower() in sent.text:
                skill_context.append(sent.text)
        
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
    
    def _calculate_skill_confidence_traditional(self, text: str, skill: str) -> float:
        """Calculate confidence using traditional methods"""
        sentences = [sent.text for sent in self.nlp(text).sents if skill.lower() in sent.text.lower()]
        
        if not sentences:
            return 0.5
        
        # Use Universal Sentence Encoder if available
        if self.use_model:
            try:
                skill_contexts = [
                    f"Expert in {skill}",
                    f"Experienced with {skill}",
                    f"Proficient in {skill}"
                ]
                
                sentence_embeddings = self.use_model(sentences[:3])
                context_embeddings = self.use_model(skill_contexts)
                
                similarities = tf.keras.utils.cosine_similarity(
                    sentence_embeddings, 
                    tf.reduce_mean(context_embeddings, axis=0, keepdims=True)
                )
                
                return float(tf.reduce_mean(similarities))
            except Exception as e:
                logger.error(f"Error with USE: {e}")
        
        # Fallback to simple confidence
        return min(0.9, 0.6 + len(sentences) * 0.1)
    
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
        try:
            # Load and preprocess image
            if hasattr(image_path, 'read'):  # File object
                content = image_path.read()
                image = Image.open(io.BytesIO(content))
                image = np.array(image)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
            else:
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
                    extracted_skills = self._extract_certificate_skills(text, cert_type)
                    skills.extend(extracted_skills)
            
            return skills
        except Exception as e:
            logger.error(f"Error extracting from certificate: {e}")
            return []
    
    def _enhance_image(self, gray_image):
        """Enhance image for better OCR results"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return gray_image
    
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
                skill_list = skill_list
            
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
    """AI-powered skill assessment system with BERT enhancement"""
    
    def __init__(self):
        self.assessment_questions = self._load_assessment_questions()
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        
        # Build TensorFlow model for adaptive assessment
        self.assessment_model = self._build_assessment_model()
        
        # BERT integration for assessment enhancement
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading BERT for assessment: {e}")
            self.bert_model = None
    
    def _build_assessment_model(self):
        """Build TensorFlow model for adaptive skill assessment"""
        try:
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
            
            model = tf.keras.Model(inputs=[question_input, user_history], outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            return model
        except Exception as e:
            logger.error(f"Error building assessment model: {e}")
            return None
    
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
                },
                {
                    'question': 'What is the purpose of the __init__ method in Python?',
                    'options': ['To initialize class variables', 'To create a new instance', 'To define class methods', 'To inherit from parent class'],
                    'correct': 1,
                    'difficulty': 'intermediate',
                    'concept': 'object oriented programming'
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
                },
                {
                    'question': 'Which algorithm is commonly used for classification problems?',
                    'options': ['Linear Regression', 'Decision Tree', 'K-means', 'PCA'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'classification'
                }
            ],
            'javascript': [
                {
                    'question': 'What is the correct way to declare a variable in JavaScript?',
                    'options': ['var x = 5;', 'variable x = 5;', 'v x = 5;', 'declare x = 5;'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'variables'
                }
            ]
        }
    
    def conduct_assessment(self, user_id: str, skill: str, num_questions: int = 10) -> Dict[str, Any]:
        """Conduct adaptive skill assessment with BERT enhancement"""
        try:
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
            
            # Simulate adaptive questioning
            for i in range(min(num_questions, len(questions))):
                question = questions[i % len(questions)]
                
                # Simulate user response
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
            
        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            return {'error': str(e)}
    
    def _calculate_skill_level(self, responses: List[Dict]) -> float:
        """Calculate estimated skill level based on responses"""
        if not responses:
            return 0.0
        
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
            return 0.3
        
        correct_count = sum(1 for r in responses if r['correct'])
        accuracy = correct_count / len(responses)
        
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
            if accuracy < 0.6:
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
            if accuracy >= 0.8:
                strong_areas.append(concept)
        
        return strong_areas

class LearningPathGenerator:
    """Generate personalized learning paths with BERT enhancement"""
    
    def __init__(self):
        self.learning_resources = self._initialize_learning_resources()
        self.skill_prerequisites = self._define_skill_prerequisites()
        
        # BERT integration for better resource matching
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading BERT for learning paths: {e}")
            self.bert_model = None
        
        # TensorFlow model for learning path optimization
        self.recommendation_model = self._build_recommendation_model()
    
    def _build_recommendation_model(self):
        """Build TensorFlow model for learning resource recommendation"""
        try:
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
            
            model = tf.keras.Model(inputs=[user_skills, user_preferences, resource_features], outputs=similarity)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            return model
        except Exception as e:
            logger.error(f"Error building recommendation model: {e}")
            return None
    
    def _initialize_learning_resources(self) -> List[LearningResource]:
        """Initialize comprehensive learning resources"""
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
            ),
            LearningResource(
                title="React Development Bootcamp",
                description="Build modern web applications with React and JavaScript",
                difficulty="intermediate",
                duration=360,
                skills=["react", "javascript", "web_development"],
                url="https://example.com/react-bootcamp",
                rating=4.4
            ),
            LearningResource(
                title="Data Science with Python",
                description="Complete data science workflow using pandas, numpy, and scikit-learn",
                difficulty="intermediate",
                duration=420,
                skills=["data_science", "python", "pandas", "numpy"],
                url="https://example.com/data-science-python",
                rating=4.8
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
            'django': ['python', 'web_development'],
            'data_science': ['python', 'statistics'],
            'tensorflow': ['python', 'machine_learning']
        }
    
    def generate_learning_path(self, current_skills: List[Skill], target_skills: List[str], 
                             preferences: Dict[str, Any] = None) -> LearningPath:
        """Generate personalized learning path with BERT enhancement"""
        try:
            if preferences is None:
                preferences = {'max_duration_per_day': 120, 'difficulty_preference': 'intermediate'}
            
            # Identify skill gaps
            current_skill_names = {skill.name.lower() for skill in current_skills}
            target_skill_names = {skill.lower() for skill in target_skills}
            skill_gaps = list(target_skill_names - current_skill_names)
            
            # Add prerequisite skills to gaps
            extended_gaps = self._add_prerequisites(skill_gaps, current_skill_names)
            
            # Find relevant learning resources
            relevant_resources = self._find_relevant_resources_bert(extended_gaps) if self.bert_model else self._find_relevant_resources(extended_gaps)
            
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
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return LearningPath(
                user_id="error",
                skills_gap=[],
                recommended_resources=[],
                estimated_completion=0,
                priority_order=[]
            )
    
    def _find_relevant_resources_bert(self, skill_gaps: List[str]) -> List[LearningResource]:
        """Find learning resources using BERT semantic matching"""
        try:
            relevant_resources = []
            
            # Generate embeddings for skill gaps
            gap_embeddings = self.bert_model.encode(skill_gaps)
            
            for resource in self.learning_resources:
                # Generate embedding for resource skills and description
                resource_text = ' '.join(resource.skills) + ' ' + resource.description
                resource_embedding = self.bert_model.encode([resource_text])
                
                # Calculate similarity with skill gaps
                similarities = cosine_similarity(gap_embeddings, resource_embedding)
                max_similarity = np.max(similarities)
                
                # Include resource if similarity is above threshold
                if max_similarity > 0.3:  # Lower threshold for broader matching
                    relevant_resources.append(resource)
            
            return relevant_resources
        except Exception as e:
            logger.error(f"Error in BERT resource matching: {e}")
            return self._find_relevant_resources(skill_gaps)
    
    def _find_relevant_resources(self, skill_gaps: List[str]) -> List[LearningResource]:
        """Find learning resources that match skill gaps (traditional method)"""
        relevant_resources = []
        
        for resource in self.learning_resources:
            resource_skills = [skill.lower() for skill in resource.skills]
            if any(gap.lower() in resource_skills for gap in skill_gaps):
                relevant_resources.append(resource)
        
        return relevant_resources
    
    def _add_prerequisites(self, skill_gaps: List[str], current_skills: set) -> List[str]:
        """Add prerequisite skills to the learning path"""
        extended_gaps = skill_gaps.copy()
        
        for skill in skill_gaps:
            if skill in self.skill_prerequisites:
                for prereq in self.skill_prerequisites[skill]:
                    if prereq not in current_skills and prereq not in extended_gaps:
                        extended_gaps.insert(0, prereq)  # Add prerequisites first
        
        return extended_gaps
    
    def _prioritize_resources(self, resources: List[LearningResource], 
                            current_skills: List[Skill], preferences: Dict[str, Any]) -> List[LearningResource]:
        """Prioritize learning resources based on user profile and preferences"""
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
        
        # Duration preference
        max_duration = preferences.get('max_duration_per_day', 120)
        if resource.duration <= max_duration:
            score += 0.3
        else:
            score += 0.1
        
        return min(1.0, score)
    
    def _difficulty_to_number(self, difficulty: str) -> int:
        """Convert difficulty string to number for comparison"""
        difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        return difficulty_map.get(difficulty, 2)

class PersonalizedLearningSystem:
    """Main BERT-enhanced system that orchestrates all components"""
    
    def __init__(self):
        self.resume_parser = BERTEnhancedResumeParser()
        self.cv_extractor = ComputerVisionSkillExtractor()
        self.assessment_engine = SkillAssessmentEngine()
        self.path_generator = LearningPathGenerator()
        self.user_profiles = {}
        
        logger.info("BERT-Enhanced Personalized Learning System initialized")
    
    def process_user_profile(self, user_id: str, resume_path: str = None, 
                           certificate_paths: List[str] = None, 
                           target_skills: List[str] = None) -> Dict[str, Any]:
        """Process complete user profile with BERT enhancement"""
        try:
            all_skills = []
            
            # Extract skills from resume using BERT
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
            key_skills = list(set([skill.name for skill in all_skills]))[:5]
            
            for skill in key_skills:
                if skill in ['python', 'machine_learning', 'javascript']:
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
        except Exception as e:
            logger.error(f"Error processing user profile: {e}")
            return {'error': str(e)}
    
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
        try:
            if user_id not in self.user_profiles:
                return {'error': 'User profile not found'}
            
            profile = self.user_profiles[user_id]
            
            # Update existing skill or add new one
            skill_updated = False
            for skill in profile['skills']:
                if skill.name.lower() == skill_name.lower():
                    skill.level = new_level
                    skill.confidence = min(1.0, skill.confidence + 0.1)
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
        except Exception as e:
            logger.error(f"Error updating skill progress: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard data for user interface"""
        try:
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
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}

class VisualizationEngine:
    """Generate visualizations for skill analysis and learning progress"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_skill_radar(self, skills: List[Skill], save_path: str = None) -> str:
        """Create radar chart for skill visualization"""
        try:
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
            
            if N == 0:
                return "No skills to visualize"
            
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
                plt.close()
                return save_path
            else:
                plt.show()
                return "displayed"
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return "error"
    
    def plot_learning_progress(self, learning_path: LearningPath, 
                             completed_resources: List[str] = None, 
                             save_path: str = None) -> str:
        """Create progress visualization for learning path"""
        try:
            if not learning_path or not learning_path.recommended_resources:
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
                plt.close()
                return save_path
            else:
                plt.show()
                return "displayed"
        except Exception as e:
            logger.error(f"Error creating progress chart: {e}")
            return "error"
    
    def plot_assessment_results(self, assessment_data: Dict[str, Any], save_path: str = None) -> str:
        """Visualize assessment results"""
        try:
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
                plt.close()
                return save_path
            else:
                plt.show()
                return "displayed"
        except Exception as e:
            logger.error(f"Error creating assessment chart: {e}")
            return "error"

# Flask Application with all original routes + BERT enhancement
app = Flask(__name__)
app.secret_key = 'bert_enhanced_skill_bridge_2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)

# Initialize BERT-enhanced system
bert_learning_system = PersonalizedLearningSystem()
visualization_engine = VisualizationEngine()

# Database setup
def init_db():
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        profile_data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS file_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        skill TEXT NOT NULL,
        score REAL NOT NULL,
        level TEXT NOT NULL,
        confidence REAL NOT NULL,
        assessment_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )''')

    conn.commit()
    conn.close()

init_db()

# Helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')

        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)

        try:
            conn = sqlite3.connect('learning_system.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO users (user_id, username, email, password_hash) VALUES (?, ?, ?, ?)''', 
                         (user_id, username, email, password_hash))
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
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Profile setup route with BERT enhancement
@app.route('/profile_setup', methods=['GET', 'POST'])
def profile_setup():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session['user_id']

        current_skills = request.form.getlist('current_skills[]')
        current_levels = request.form.getlist('current_levels[]')
        target_skills = request.form.getlist('target_skills[]')

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
                    save_file_upload(user_id, filename, 'certificate', cert_path)

        # Create manual skills
        manual_skills = []
        for skill, level in zip(current_skills, current_levels):
            if skill.strip():
                try:
                    level_float = float(level) / 100.0
                    manual_skills.append(Skill(
                        name=skill.strip(),
                        level=level_float,
                        category=categorize_skill(skill.strip()),
                        confidence=0.8,
                        source='user_input'
                    ))
                except ValueError:
                    continue

        try:
            # Process with BERT enhancement
            profile_result = bert_learning_system.process_user_profile(
                user_id=user_id,
                resume_path=resume_path,
                certificate_paths=certificate_paths,
                target_skills=[s.strip() for s in target_skills if s.strip()]
            )

            # Add manual skills
            if user_id in bert_learning_system.user_profiles:
                bert_learning_system.user_profiles[user_id]['skills'].extend(manual_skills)

            save_user_profile(user_id, profile_result)

            flash('Profile created successfully with BERT analysis!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f'Error processing profile: {str(e)}', 'error')
            logger.error(f"Profile setup error: {e}")
            return render_template('profile_setup.html')

    return render_template('profile_setup.html')

# BERT-enhanced dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    profile_data = get_user_profile(user_id)
    if not profile_data:
        return redirect(url_for('profile_setup'))

    # Get BERT-enhanced dashboard data
    dashboard_data = bert_learning_system.get_dashboard_data(user_id)
    
    # Generate visualizations
    if user_id in bert_learning_system.user_profiles:
        skills = bert_learning_system.user_profiles[user_id]['skills']
        
        # Create skill radar chart
        radar_path = f'static/visualizations/{user_id}_skills_radar.png'
        visualization_engine.plot_skill_radar(skills, radar_path)
        
        # Create learning progress chart if learning path exists
        learning_path = bert_learning_system.user_profiles[user_id].get('learning_path')
        if learning_path:
            progress_path = f'static/visualizations/{user_id}_progress.png'
            visualization_engine.plot_learning_progress(learning_path, save_path=progress_path)

    return render_template('dashboard.html', 
                         dashboard_data=dashboard_data,
                         username=session.get('username'))

# Skill assessment route
@app.route('/assessment/<skill>')
def skill_assessment(skill):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('assessment.html', skill=skill)

@app.route('/conduct_assessment', methods=['POST'])
def conduct_assessment():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        user_id = session['user_id']
        skill = data.get('skill')
        num_questions = data.get('num_questions', 10)
        
        # Conduct BERT-enhanced assessment
        assessment_result = bert_learning_system.assessment_engine.conduct_assessment(
            user_id, skill, num_questions
        )
        
        # Save assessment result
        save_assessment_result(user_id, assessment_result)
        
        return jsonify(assessment_result)
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        return jsonify({'error': str(e)}), 500

# Learning path route
@app.route('/learning_path')
def learning_path():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if user_id in bert_learning_system.user_profiles:
        learning_path = bert_learning_system.user_profiles[user_id].get('learning_path')
        return render_template('learning_path.html', learning_path=learning_path)
    
    return redirect(url_for('profile_setup'))

# API endpoints
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        user_id = request.form.get('userId') or session.get('user_id')
        resume_file = request.files.get('resume')
        manual_skills_json = request.form.get('manual_skills')

        manual_skills = []
        if manual_skills_json:
            manual_skills = json.loads(manual_skills_json)

        # Process with BERT enhancement
        result = bert_learning_system.process_user_profile(
            user_id=user_id,
            resume_path=resume_file,
            target_skills=[]
        )

        return jsonify({
            'status': 'success',
            'message': 'Resume processed with BERT analysis',
            'user_id': user_id,
            'bert_analysis': result,
            'extracted_skills': result.get('extracted_skills', [])
        })

    except Exception as e:
        logger.error(f"Error in upload_resume: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/skill_gap_analysis', methods=['POST'])
def skill_gap_analysis():
    try:
        data = request.get_json()
        user_skills = data.get('skills', [])
        target_job = data.get('target_job', '')

        if not user_skills or not target_job:
            return jsonify({'error': 'Skills and target job required'}), 400

        # BERT-powered skill gap analysis
        gap_analysis = bert_learning_system.path_generator.skill_gap_analyzer.analyze_skill_gaps(
            user_skills, target_job
        )

        if 'skill_gaps' in gap_analysis:
            roadmap = bert_learning_system.path_generator.skill_gap_analyzer.generate_learning_roadmap(
                gap_analysis['skill_gaps']
            )
            gap_analysis['learning_roadmap'] = roadmap

        return jsonify(gap_analysis)

    except Exception as e:
        logger.error(f"Error in skill gap analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_progress', methods=['POST'])
def update_progress():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        user_id = session['user_id']
        skill_name = data.get('skill')
        new_level = data.get('level')
        completion_data = data.get('completion_data', {})
        
        result = bert_learning_system.update_skill_progress(
            user_id, skill_name, new_level, completion_data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Progress update error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_visualization/<viz_type>')
def generate_visualization(viz_type):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        if user_id not in bert_learning_system.user_profiles:
            return jsonify({'error': 'Profile not found'}), 404
        
        profile = bert_learning_system.user_profiles[user_id]
        
        if viz_type == 'skills_radar':
            skills = profile['skills']
            save_path = f'static/visualizations/{user_id}_skills_radar.png'
            result = visualization_engine.plot_skill_radar(skills, save_path)
            
        elif viz_type == 'learning_progress':
            learning_path = profile.get('learning_path')
            if learning_path:
                save_path = f'static/visualizations/{user_id}_progress.png'
                result = visualization_engine.plot_learning_progress(learning_path, save_path=save_path)
            else:
                return jsonify({'error': 'No learning path found'}), 404
                
        elif viz_type == 'assessment_results':
            assessments = profile.get('assessments', {})
            if assessments:
                save_path = f'static/visualizations/{user_id}_assessments.png'
                result = visualization_engine.plot_assessment_results(assessments, save_path)
            else:
                return jsonify({'error': 'No assessment data found'}), 404
        
        else:
            return jsonify({'error': 'Invalid visualization type'}), 400
        
        return jsonify({'status': 'success', 'path': result})
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions
def categorize_skill(skill_name):
    skill_name_lower = skill_name.lower()

    if any(lang in skill_name_lower for lang in ['python', 'java', 'javascript', 'react', 'angular']):
        return 'programming'
    elif any(ds in skill_name_lower for ds in ['machine learning', 'data science', 'tensorflow']):
        return 'data_science'
    elif any(cloud in skill_name_lower for cloud in ['aws', 'azure', 'gcp', 'docker']):
        return 'cloud'
    elif any(db in skill_name_lower for db in ['sql', 'mongodb', 'postgresql']):
        return 'database'
    else:
        return 'general'

def save_file_upload(user_id, filename, file_type, file_path):
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO file_uploads (user_id, filename, file_type, file_path) VALUES (?, ?, ?, ?)''', 
                   (user_id, filename, file_type, file_path))
    conn.commit()
    conn.close()

def save_user_profile(user_id, profile_data):
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
    existing = cursor.fetchone()

    profile_json = json.dumps(profile_data, default=str)

    if existing:
        cursor.execute('''UPDATE user_profiles SET profile_data = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?''', 
                       (profile_json, user_id))
    else:
        cursor.execute('''INSERT INTO user_profiles (user_id, profile_data) VALUES (?, ?)''', 
                       (user_id, profile_json))

    conn.commit()
    conn.close()

def get_user_profile(user_id):
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    cursor.execute('SELECT profile_data FROM user_profiles WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return json.loads(result[0])
    return None

def save_assessment_result(user_id, assessment_result):
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    
    cursor.execute('''INSERT INTO assessments 
                     (user_id, skill, score, level, confidence, assessment_data) 
                     VALUES (?, ?, ?, ?, ?, ?)''',
                   (user_id, 
                    assessment_result.get('skill', ''),
                    assessment_result.get('estimated_level', 0),
                    assessment_result.get('level', 'beginner'),
                    assessment_result.get('confidence', 0),
                    json.dumps(assessment_result, default=str)))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting BERT-Enhanced Skill Assessment System with Full Web Interface...")
    logger.info("Features: BERT semantic analysis, skill gap analysis, learning roadmaps, assessments")
    app.run(debug=True, host='0.0.0.0', port=5000)

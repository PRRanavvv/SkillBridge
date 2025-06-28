import os
import sqlite3
import json
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional, Tuple, Union, Any
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
import fitz  # PyMuPDF for PDF processing
import docx
from werkzeug.utils import secure_filename

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import warnings
# warnings.filterwarnings('ignore')


# Suppress warnings but keep critical ones
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_skill_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemError(Exception):
    """Base exception for system errors"""
    pass

class DataProcessingError(SystemError):
    """Exception for data processing errors"""
    pass

class ModelLoadError(SystemError):
    """Exception for model loading errors"""
    pass

class FileProcessingError(SystemError):
    """Exception for file processing errors"""
    pass

class DatabaseError(SystemError):
    """Exception for database errors"""
    pass

class BERTEnhancedResumeParser:
    """Enhanced Resume Parser with comprehensive error handling and BERT integration"""
    
    def __init__(self):
        self.nlp = None
        self.bert_model = None
        self.tfidf_vectorizer = None
        self.skill_embeddings_cache = {}
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'bert_extractions': 0,
            'traditional_extractions': 0
        }
        
        # Initialize components with error handling
        self._initialize_nlp_model()
        self._initialize_bert_model()
        self._initialize_tfidf()
        self._setup_skill_database()
        self._build_skill_embeddings()
    
    def _initialize_nlp_model(self):
        """Initialize spaCy model with comprehensive error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            logger.warning(f"spaCy model not found: {e}. Attempting to download...")
            try:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {download_error}")
                # Fallback to basic text processing
                self.nlp = None
                logger.warning("Using fallback text processing without spaCy")
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy: {e}")
            self.nlp = None
    
    def _initialize_bert_model(self):
        """Initialize BERT model with error handling and fallbacks"""
        try:
            logger.info("Loading BERT model: all-MiniLM-L6-v2")
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            try:
                # Try alternative model
                logger.info("Trying alternative BERT model: all-mpnet-base-v2")
                self.bert_model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("Alternative BERT model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback BERT model: {fallback_error}")
                self.bert_model = None
                logger.warning("BERT functionality disabled - using traditional methods only")
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer with error handling"""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            logger.info("TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
    
    def _setup_skill_database(self):
        """Setup comprehensive skill database with error handling"""
        try:
            self.skill_keywords = {
                'programming': [
                    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                    'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'perl', 'dart',
                    'objective-c', 'assembly', 'cobol', 'fortran', 'haskell', 'lua', 'shell'
                ],
                'web_development': [
                    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
                    'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'webpack', 'sass',
                    'less', 'tailwind', 'next.js', 'nuxt.js', 'gatsby', 'svelte', 'ember'
                ],
                'data_science': [
                    'machine learning', 'deep learning', 'data analysis', 'statistics',
                    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                    'data visualization', 'tableau', 'power bi', 'matplotlib', 'seaborn',
                    'plotly', 'jupyter', 'anaconda', 'spark', 'hadoop', 'nlp', 'computer vision'
                ],
                'databases': [
                    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                    'oracle', 'sqlite', 'elasticsearch', 'neo4j', 'dynamodb', 'firebase',
                    'mariadb', 'couchdb', 'influxdb', 'clickhouse'
                ],
                'cloud_devops': [
                    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
                    'terraform', 'ansible', 'linux', 'bash', 'ci/cd', 'helm', 'prometheus',
                    'grafana', 'nagios', 'puppet', 'chef', 'vagrant', 'gitlab'
                ],
                'mobile_development': [
                    'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
                    'cordova', 'phonegap', 'swift', 'kotlin', 'objective-c'
                ],
                'ai_ml': [
                    'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
                    'natural language processing', 'computer vision', 'reinforcement learning',
                    'generative ai', 'llm', 'bert', 'gpt', 'transformers', 'opencv'
                ],
                'cybersecurity': [
                    'cybersecurity', 'information security', 'penetration testing', 'ethical hacking',
                    'vulnerability assessment', 'security audit', 'firewall', 'encryption',
                    'malware analysis', 'incident response', 'risk assessment'
                ]
            }
            
            # Add skill variations and synonyms
            self._add_skill_variations()
            logger.info(f"Skill database initialized with {sum(len(skills) for skills in self.skill_keywords.values())} skills")
            
        except Exception as e:
            logger.error(f"Error setting up skill database: {e}")
            # Minimal fallback skill set
            self.skill_keywords = {
                'programming': ['python', 'java', 'javascript'],
                'web_development': ['html', 'css', 'react'],
                'data_science': ['machine learning', 'data analysis']
            }
    
    def _add_skill_variations(self):
        """Add skill variations and synonyms for better matching"""
        try:
            variations = {
                'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
                'python': ['py', 'python3', 'python2'],
                'machine learning': ['ml', 'artificial intelligence', 'ai'],
                'react': ['reactjs', 'react.js'],
                'node.js': ['nodejs', 'node'],
                'tensorflow': ['tf'],
                'pytorch': ['torch'],
                'kubernetes': ['k8s'],
                'docker': ['containerization'],
                'git': ['version control', 'github', 'gitlab', 'bitbucket']
            }
            
            for category, skills in self.skill_keywords.items():
                extended_skills = skills.copy()
                for skill in skills:
                    if skill in variations:
                        extended_skills.extend(variations[skill])
                self.skill_keywords[category] = list(set(extended_skills))
                
        except Exception as e:
            logger.error(f"Error adding skill variations: {e}")
    
    def _build_skill_embeddings(self):
        """Build BERT embeddings for all skills with comprehensive error handling"""
        if not self.bert_model:
            logger.warning("BERT model not available - skipping embedding generation")
            return
        
        try:
            logger.info("Building BERT embeddings for skill database...")
            all_skills = []
            skill_categories = {}
            
            for category, skills in self.skill_keywords.items():
                for skill in skills:
                    all_skills.append(skill)
                    skill_categories[skill] = category
            
            # Generate embeddings in batches
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(all_skills), batch_size):
                batch = all_skills[i:i + batch_size]
                try:
                    batch_embeddings = self.bert_model.encode(
                        batch, 
                        convert_to_tensor=False,
                        show_progress_bar=False
                    )
                    embeddings.extend(batch_embeddings)
                    
                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed {i + len(batch)}/{len(all_skills)} skill embeddings")
                        
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i//batch_size}: {batch_error}")
                    # Add zero embeddings for failed batch
                    embeddings.extend([np.zeros(384) for _ in batch])
            
            # Cache embeddings
            for skill, embedding in zip(all_skills, embeddings):
                self.skill_embeddings_cache[skill.lower()] = {
                    'embedding': embedding,
                    'category': skill_categories[skill]
                }
            
            logger.info(f"Built embeddings for {len(all_skills)} skills")
            
        except Exception as e:
            logger.error(f"Error building skill embeddings: {e}")
            self.skill_embeddings_cache = {}
    
    def extract_text_from_file(self, file_path_or_content, file_type=None):
        """Extract text from various file formats with comprehensive error handling"""
        try:
            if isinstance(file_path_or_content, str):
                # File path provided
                file_path = Path(file_path_or_content)
                if not file_path.exists():
                    raise FileProcessingError(f"File not found: {file_path}")
                
                file_type = file_type or file_path.suffix.lower()
                
                with open(file_path, 'rb') as file:
                    content = file.read()
            else:
                # File content provided
                content = file_path_or_content
                if not file_type:
                    raise FileProcessingError("File type must be specified when providing content")
            
            # Extract text based on file type
            if file_type in ['.pdf']:
                return self._extract_from_pdf(content)
            elif file_type in ['.docx', '.doc']:
                return self._extract_from_docx(content)
            elif file_type in ['.txt']:
                return self._extract_from_txt(content)
            elif file_type in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self._extract_from_image(content)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise FileProcessingError(f"Failed to extract text: {str(e)}")
    
    def _extract_from_pdf(self, content):
        """Extract text from PDF with error handling"""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return ""
    
    def _extract_from_docx(self, content):
        """Extract text from DOCX with error handling"""
        try:
            doc = docx.Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            return ""
    
    def _extract_from_txt(self, content):
        """Extract text from TXT with error handling"""
        try:
            if isinstance(content, bytes):
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                # If all fail, use error handling
                return content.decode('utf-8', errors='ignore')
            return str(content)
        except Exception as e:
            logger.error(f"Error extracting from TXT: {e}")
            return ""
    
    def _extract_from_image(self, content):
        """Extract text from image using OCR with error handling"""
        try:
            if not pytesseract:
                logger.warning("Tesseract not available for OCR")
                return ""
            
            # Convert bytes to image
            image = Image.open(io.BytesIO(content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            return text
            
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return ""
    
    def extract_skills_bert(self, text, confidence_threshold=0.6):
        """BERT-powered skill extraction with comprehensive error handling"""
        if not self.bert_model or not self.skill_embeddings_cache:
            logger.warning("BERT model or embeddings not available")
            return []
        
        try:
            # Clean and preprocess text
            text = self._clean_text(text)
            if not text.strip():
                return []
            
            # Generate embedding for input text
            text_embedding = self.bert_model.encode([text.lower()], show_progress_bar=False)
            
            # Find semantically similar skills
            detected_skills = []
            
            for skill, skill_data in self.skill_embeddings_cache.items():
                try:
                    similarity = cosine_similarity(
                        text_embedding, 
                        [skill_data['embedding']]
                    )[0][0]
                    
                    if similarity > confidence_threshold:
                        detected_skills.append({
                            'skill': skill,
                            'confidence': float(similarity),
                            'category': skill_data['category'],
                            'method': 'bert_semantic'
                        })
                        
                except Exception as skill_error:
                    logger.debug(f"Error processing skill {skill}: {skill_error}")
                    continue
            
            # Sort by confidence
            detected_skills.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.processing_stats['bert_extractions'] += 1
            return detected_skills[:20]  # Top 20 most relevant skills
            
        except Exception as e:
            logger.error(f"Error in BERT skill extraction: {e}")
            return []
    
    def extract_skills_traditional(self, text):
        """Traditional keyword-based skill extraction with error handling"""
        try:
            text_lower = self._clean_text(text).lower()
            if not text_lower.strip():
                return []
            
            detected_skills = []
            
            for category, skills in self.skill_keywords.items():
                for skill in skills:
                    try:
                        # Use regex for better matching
                        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                        if re.search(pattern, text_lower):
                            detected_skills.append({
                                'skill': skill,
                                'confidence': 1.0,
                                'category': category,
                                'method': 'keyword_match'
                            })
                    except Exception as skill_error:
                        logger.debug(f"Error processing skill {skill}: {skill_error}")
                        continue
            
            self.processing_stats['traditional_extractions'] += 1
            return detected_skills
            
        except Exception as e:
            logger.error(f"Error in traditional skill extraction: {e}")
            return []
    
    def _clean_text(self, text):
        """Clean and preprocess text with error handling"""
        try:
            if not text:
                return ""
            
            # Convert to string if not already
            text = str(text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep alphanumeric and common punctuation
            text = re.sub(r'[^\w\s\-\.\+\#]', ' ', text)
            
            # Remove extra spaces
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return str(text) if text else ""
    
    def hybrid_skill_extraction(self, text, bert_weight=0.7, traditional_weight=0.3):
        """Combine BERT and traditional approaches with comprehensive error handling"""
        try:
            self.processing_stats['total_processed'] += 1
            
            # Extract skills using both methods
            bert_skills = self.extract_skills_bert(text) if self.bert_model else []
            traditional_skills = self.extract_skills_traditional(text)
            
            # Merge and deduplicate
            all_skills = {}
            
            # Add traditional skills (high confidence for exact matches)
            for skill_info in traditional_skills:
                skill = skill_info['skill'].lower()
                all_skills[skill] = {
                    'skill': skill_info['skill'],
                    'confidence': 0.95 * traditional_weight,
                    'category': skill_info['category'],
                    'method': 'keyword',
                    'sources': ['traditional']
                }
            
            # Add BERT skills (semantic matches)
            for skill_info in bert_skills:
                skill = skill_info['skill'].lower()
                bert_confidence = skill_info['confidence'] * bert_weight
                
                if skill in all_skills:
                    # Combine confidences
                    existing_confidence = all_skills[skill]['confidence']
                    combined_confidence = min(1.0, existing_confidence + bert_confidence)
                    all_skills[skill]['confidence'] = combined_confidence
                    all_skills[skill]['sources'].append('bert')
                    all_skills[skill]['method'] = 'hybrid'
                else:
                    all_skills[skill] = {
                        'skill': skill_info['skill'],
                        'confidence': bert_confidence,
                        'category': skill_info['category'],
                        'method': 'semantic',
                        'sources': ['bert']
                    }
            
            # Sort by confidence and return
            final_skills = list(all_skills.values())
            final_skills.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.processing_stats['successful_extractions'] += 1
            return final_skills
            
        except Exception as e:
            logger.error(f"Error in hybrid skill extraction: {e}")
            self.processing_stats['failed_extractions'] += 1
            # Return traditional skills as fallback
            try:
                return self.extract_skills_traditional(text)
            except:
                return []
    
    def extract_personal_info(self, text):
        """Extract personal information with comprehensive error handling"""
        try:
            personal_info = {
                'name': '',
                'email': '',
                'phone': '',
                'location': '',
                'linkedin': '',
                'github': ''
            }
            
            if not text:
                return personal_info
            
            text = str(text)
            
            # Extract email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                personal_info['email'] = emails[0]
            
            # Extract phone
            phone_patterns = [
                r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}'
            ]
            
            for pattern in phone_patterns:
                phones = re.findall(pattern, text)
                if phones:
                    if isinstance(phones[0], tuple):
                        personal_info['phone'] = '-'.join(phones[0])
                    else:
                        personal_info['phone'] = phones[0]
                    break
            
            # Extract LinkedIn
            linkedin_pattern = r'linkedin\.com/in/([A-Za-z0-9-]+)'
            linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
            if linkedin_matches:
                personal_info['linkedin'] = f"linkedin.com/in/{linkedin_matches[0]}"
            
            # Extract GitHub
            github_pattern = r'github\.com/([A-Za-z0-9-]+)'
            github_matches = re.findall(github_pattern, text, re.IGNORECASE)
            if github_matches:
                personal_info['github'] = f"github.com/{github_matches[0]}"
            
            # Extract name (first few words, excluding common resume headers)
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and not any(header in line.lower() for header in 
                                  ['resume', 'cv', 'curriculum', 'contact', 'email', 'phone']):
                    words = line.split()
                    if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
                        personal_info['name'] = line
                        break
            
            return personal_info
            
        except Exception as e:
            logger.error(f"Error extracting personal info: {e}")
            return {
                'name': '', 'email': '', 'phone': '', 'location': '', 
                'linkedin': '', 'github': ''
            }
    
    def get_processing_stats(self):
        """Get processing statistics"""
        return self.processing_stats.copy()

class BERTSkillGapAnalyzer:
    """BERT-powered skill gap analysis with comprehensive error handling"""
    
    def __init__(self):
        self.bert_model = None
        self.job_market_skills = {}
        self.learning_resources = {}
        self.skill_difficulty_map = {}
        self.industry_trends = {}
        
        # Initialize components
        self._initialize_bert_model()
        self._setup_job_market_data()
        self._setup_learning_resources()
        self._setup_skill_difficulty_mapping()
        self._setup_industry_trends()
    
    def _initialize_bert_model(self):
        """Initialize BERT model with error handling"""
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model initialized for skill gap analysis")
        except Exception as e:
            logger.error(f"Error initializing BERT model: {e}")
            self.bert_model = None
    
    def _setup_job_market_data(self):
        """Setup comprehensive job market data"""
        try:
            self.job_market_skills = {
                'data_scientist': {
                    'required_skills': [
                        'Python', 'Machine Learning', 'Deep Learning', 'Statistics',
                        'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'SQL',
                        'Data Visualization', 'A/B Testing', 'Feature Engineering',
                        'Jupyter', 'Git', 'Linear Algebra', 'Probability'
                    ],
                    'preferred_skills': [
                        'PyTorch', 'Spark', 'Hadoop', 'AWS', 'Docker', 'MLOps',
                        'Tableau', 'Power BI', 'R', 'Scala', 'Kubernetes'
                    ],
                    'avg_salary': 95000,
                    'growth_rate': 0.22
                },
                'full_stack_developer': {
                    'required_skills': [
                        'JavaScript', 'React', 'Node.js', 'HTML', 'CSS', 'Python',
                        'SQL', 'Git', 'RESTful APIs', 'MongoDB', 'Express.js',
                        'Responsive Design', 'Version Control'
                    ],
                    'preferred_skills': [
                        'AWS', 'Docker', 'TypeScript', 'GraphQL', 'Redux',
                        'Vue.js', 'Angular', 'PostgreSQL', 'Redis', 'Webpack'
                    ],
                    'avg_salary': 78000,
                    'growth_rate': 0.15
                },
                'machine_learning_engineer': {
                    'required_skills': [
                        'Python', 'TensorFlow', 'PyTorch', 'Kubernetes', 'Docker',
                        'MLOps', 'Cloud Computing', 'Model Deployment', 'Git',
                        'Linux', 'CI/CD', 'Machine Learning', 'Deep Learning'
                    ],
                    'preferred_skills': [
                        'AWS', 'Azure', 'GCP', 'Airflow', 'Spark', 'Kafka',
                        'Monitoring', 'A/B Testing', 'Feature Stores', 'Kubeflow'
                    ],
                    'avg_salary': 105000,
                    'growth_rate': 0.28
                },
                'software_engineer': {
                    'required_skills': [
                        'Programming Languages', 'Data Structures', 'Algorithms',
                        'System Design', 'Git', 'Testing', 'Debugging', 'APIs',
                        'Database Design', 'Software Architecture'
                    ],
                    'preferred_skills': [
                        'Cloud Platforms', 'DevOps', 'Microservices', 'Agile',
                        'Code Review', 'Performance Optimization', 'Security'
                    ],
                    'avg_salary': 85000,
                    'growth_rate': 0.18
                },
                'devops_engineer': {
                    'required_skills': [
                        'Linux', 'Docker', 'Kubernetes', 'AWS', 'CI/CD', 'Git',
                        'Terraform', 'Ansible', 'Monitoring', 'Shell Scripting',
                        'Infrastructure as Code', 'Jenkins'
                    ],
                    'preferred_skills': [
                        'Azure', 'GCP', 'Helm', 'Prometheus', 'Grafana',
                        'ELK Stack', 'Service Mesh', 'GitOps', 'Security'
                    ],
                    'avg_salary': 92000,
                    'growth_rate': 0.25
                },
                'cybersecurity_analyst': {
                    'required_skills': [
                        'Network Security', 'Incident Response', 'Risk Assessment',
                        'Security Frameworks', 'Penetration Testing', 'SIEM',
                        'Vulnerability Assessment', 'Compliance', 'Forensics'
                    ],
                    'preferred_skills': [
                        'Cloud Security', 'Threat Intelligence', 'Malware Analysis',
                        'Security Automation', 'Zero Trust', 'DevSecOps'
                    ],
                    'avg_salary': 88000,
                    'growth_rate': 0.31
                }
            }
            logger.info(f"Job market data loaded for {len(self.job_market_skills)} roles")
        except Exception as e:
            logger.error(f"Error setting up job market data: {e}")
            self.job_market_skills = {}
    
    def _setup_learning_resources(self):
        """Setup comprehensive learning resources database"""
        try:
            self.learning_resources = {
                'python': {
                    'youtube': [
                        'Python Tutorial - Programming with Mosh',
                        'Automate the Boring Stuff with Python',
                        'Python Full Course - freeCodeCamp',
                        'Corey Schafer Python Tutorials'
                    ],
                    'articles': [
                        'Python.org Official Tutorial',
                        'Real Python',
                        'GeeksforGeeks Python',
                        'Python Tricks by Dan Bader'
                    ],
                    'practice': [
                        'HackerRank Python',
                        'LeetCode Python',
                        'Codewars Python',
                        'Python Challenge'
                    ],
                    'projects': [
                        'Build a Web Scraper',
                        'Create a REST API with Flask',
                        'Data Analysis with Pandas',
                        'Build a GUI Application'
                    ],
                    'books': [
                        'Python Crash Course',
                        'Effective Python',
                        'Fluent Python'
                    ]
                },
                'machine learning': {
                    'youtube': [
                        'Andrew Ng Machine Learning Course',
                        '3Blue1Brown Neural Networks',
                        'StatQuest Machine Learning',
                        'Krish Naik ML Playlist'
                    ],
                    'articles': [
                        'Towards Data Science',
                        'Machine Learning Mastery',
                        'Analytics Vidhya',
                        'KDnuggets'
                    ],
                    'practice': [
                        'Kaggle Competitions',
                        'Google Colab Notebooks',
                        'ML Course Labs',
                        'Hands-On ML Exercises'
                    ],
                    'projects': [
                        'Iris Classification',
                        'House Price Prediction',
                        'Customer Segmentation',
                        'Recommendation System'
                    ],
                    'books': [
                        'Hands-On Machine Learning',
                        'Pattern Recognition and Machine Learning',
                        'The Elements of Statistical Learning'
                    ]
                },
                'react': {
                    'youtube': [
                        'React Tutorial - Traversy Media',
                        'React Course - freeCodeCamp',
                        'React Hooks Tutorial',
                        'Modern React with Redux'
                    ],
                    'articles': [
                        'React Official Documentation',
                        'React Patterns',
                        'Overreacted by Dan Abramov',
                        'React Best Practices'
                    ],
                    'practice': [
                        'React Challenges',
                        'Frontend Mentor',
                        'React Exercises',
                        'Component Library Building'
                    ],
                    'projects': [
                        'Todo App with Hooks',
                        'Weather App',
                        'E-commerce Site',
                        'Social Media Dashboard'
                    ],
                    'books': [
                        'Learning React',
                        'React Up & Running',
                        'Fullstack React'
                    ]
                }
                # Add more skills as needed
            }
            
            # Generate generic resources for skills not explicitly defined
            self._generate_generic_resources()
            
        except Exception as e:
            logger.error(f"Error setting up learning resources: {e}")
            self.learning_resources = {}
    
    def _generate_generic_resources(self):
        """Generate generic learning resources for skills not explicitly defined"""
        generic_template = {
            'youtube': ['{skill} Tutorial for Beginners', '{skill} Complete Course'],
            'articles': ['Learn {skill} - Official Documentation', '{skill} Best Practices'],
            'practice': ['{skill} Practice Problems', '{skill} Coding Challenges'],
            'projects': ['Build a {skill} Project', '{skill} Portfolio Project'],
            'books': ['{skill} Guide', 'Mastering {skill}']
        }
        
        # This template will be used for skills not explicitly defined
        self.generic_resource_template = generic_template
    
    def _setup_skill_difficulty_mapping(self):
        """Setup skill difficulty and learning time estimates"""
        try:
            self.skill_difficulty_map = {
                # Beginner (1-2 months)
                'html': {'difficulty': 'beginner', 'time_weeks': 4},
                'css': {'difficulty': 'beginner', 'time_weeks': 6},
                'git': {'difficulty': 'beginner', 'time_weeks': 3},
                'sql': {'difficulty': 'beginner', 'time_weeks': 8},
                
                # Intermediate (2-4 months)
                'javascript': {'difficulty': 'intermediate', 'time_weeks': 12},
                'python': {'difficulty': 'intermediate', 'time_weeks': 10},
                'react': {'difficulty': 'intermediate', 'time_weeks': 14},
                'node.js': {'difficulty': 'intermediate', 'time_weeks': 12},
                'docker': {'difficulty': 'intermediate', 'time_weeks': 8},
                
                # Advanced (4-6 months)
                'machine learning': {'difficulty': 'advanced', 'time_weeks': 20},
                'deep learning': {'difficulty': 'advanced', 'time_weeks': 24},
                'kubernetes': {'difficulty': 'advanced', 'time_weeks': 16},
                'system design': {'difficulty': 'advanced', 'time_weeks': 18},
                
                # Expert (6+ months)
                'mlops': {'difficulty': 'expert', 'time_weeks': 28},
                'distributed systems': {'difficulty': 'expert', 'time_weeks': 32},
                'cybersecurity': {'difficulty': 'expert', 'time_weeks': 30}
            }
        except Exception as e:
            logger.error(f"Error setting up skill difficulty mapping: {e}")
            self.skill_difficulty_map = {}
    
    def _setup_industry_trends(self):
        """Setup industry trends and demand data"""
        try:
            self.industry_trends = {
                'ai_ml': {
                    'growth_rate': 0.35,
                    'hot_skills': ['LLM', 'Generative AI', 'MLOps', 'Computer Vision'],
                    'emerging_skills': ['Prompt Engineering', 'AI Ethics', 'Model Interpretability']
                },
                'web_development': {
                    'growth_rate': 0.15,
                    'hot_skills': ['React', 'TypeScript', 'Next.js', 'GraphQL'],
                    'emerging_skills': ['Web3', 'JAMstack', 'Micro-frontends']
                },
                'cloud_computing': {
                    'growth_rate': 0.28,
                    'hot_skills': ['AWS', 'Kubernetes', 'Serverless', 'DevOps'],
                    'emerging_skills': ['Edge Computing', 'Multi-cloud', 'FinOps']
                },
                'cybersecurity': {
                    'growth_rate': 0.31,
                    'hot_skills': ['Zero Trust', 'Cloud Security', 'DevSecOps'],
                    'emerging_skills': ['AI Security', 'Quantum Cryptography', 'Privacy Engineering']
                }
            }
        except Exception as e:
            logger.error(f"Error setting up industry trends: {e}")
            self.industry_trends = {}
    
    def analyze_skill_gaps(self, user_skills, target_job, user_experience_level='intermediate'):
        """Comprehensive skill gap analysis with error handling"""
        try:
            if not user_skills:
                return {'error': 'No user skills provided'}
            
            target_job_key = target_job.lower().replace(' ', '_')
            if target_job_key not in self.job_market_skills:
                return {'error': f'Job role {target_job} not found in database'}
            
            job_data = self.job_market_skills[target_job_key]
            required_skills = job_data['required_skills']
            preferred_skills = job_data.get('preferred_skills', [])
            
            # Analyze skills using BERT if available
            if self.bert_model:
                analysis = self._bert_skill_analysis(user_skills, required_skills, preferred_skills)
            else:
                analysis = self._traditional_skill_analysis(user_skills, required_skills, preferred_skills)
            
            # Add job market context
            analysis.update({
                'target_job': target_job,
                'job_market_info': {
                    'avg_salary': job_data.get('avg_salary', 0),
                    'growth_rate': job_data.get('growth_rate', 0),
                    'total_required_skills': len(required_skills),
                    'total_preferred_skills': len(preferred_skills)
                },
                'user_experience_level': user_experience_level
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in skill gap analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _bert_skill_analysis(self, user_skills, required_skills, preferred_skills):
        """BERT-powered skill analysis"""
        try:
            user_skills_text = ' '.join(user_skills)
            user_embedding = self.bert_model.encode([user_skills_text])
            
            skill_analysis = []
            
            # Analyze required skills
            for skill in required_skills:
                skill_embedding = self.bert_model.encode([skill])
                similarity = cosine_similarity(user_embedding, skill_embedding)[0][0]
                
                status = 'strong' if similarity > 0.7 else 'weak' if similarity > 0.4 else 'missing'
                priority = 'high' if similarity < 0.4 else 'medium' if similarity < 0.7 else 'low'
                
                skill_analysis.append({
                    'skill': skill,
                    'similarity': float(similarity),
                    'status': status,
                    'priority': priority,
                    'skill_type': 'required',
                    'learning_time': self._get_learning_time(skill)
                })
            
            # Analyze preferred skills
            for skill in preferred_skills:
                skill_embedding = self.bert_model.encode([skill])
                similarity = cosine_similarity(user_embedding, skill_embedding)[0][0]
                
                status = 'strong' if similarity > 0.7 else 'weak' if similarity > 0.4 else 'missing'
                priority = 'medium' if similarity < 0.4 else 'low'
                
                skill_analysis.append({
                    'skill': skill,
                    'similarity': float(similarity),
                    'status': status,
                    'priority': priority,
                    'skill_type': 'preferred',
                    'learning_time': self._get_learning_time(skill)
                })
            
            # Calculate overall match
            required_similarities = [s['similarity'] for s in skill_analysis if s['skill_type'] == 'required']
            overall_match = np.mean(required_similarities) * 100 if required_similarities else 0
            
            return {
                'overall_match_percentage': round(overall_match, 2),
                'skill_analysis': skill_analysis,
                'skill_gaps': [s for s in skill_analysis if s['status'] in ['weak', 'missing']],
                'strong_skills': [s for s in skill_analysis if s['status'] == 'strong'],
                'analysis_method': 'bert_semantic'
            }
            
        except Exception as e:
            logger.error(f"Error in BERT skill analysis: {e}")
            return self._traditional_skill_analysis(user_skills, required_skills, preferred_skills)
    
    def _traditional_skill_analysis(self, user_skills, required_skills, preferred_skills):
        """Traditional keyword-based skill analysis as fallback"""
        try:
            user_skills_lower = [skill.lower() for skill in user_skills]
            skill_analysis = []
            
            # Analyze required skills
            for skill in required_skills:
                skill_lower = skill.lower()
                has_skill = any(skill_lower in user_skill or user_skill in skill_lower 
                              for user_skill in user_skills_lower)
                
                status = 'strong' if has_skill else 'missing'
                priority = 'low' if has_skill else 'high'
                similarity = 1.0 if has_skill else 0.0
                
                skill_analysis.append({
                    'skill': skill,
                    'similarity': similarity,
                    'status': status,
                    'priority': priority,
                    'skill_type': 'required',
                    'learning_time': self._get_learning_time(skill)
                })
            
            # Analyze preferred skills
            for skill in preferred_skills:
                skill_lower = skill.lower()
                has_skill = any(skill_lower in user_skill or user_skill in skill_lower 
                              for user_skill in user_skills_lower)
                
                status = 'strong' if has_skill else 'missing'
                priority = 'low' if has_skill else 'medium'
                similarity = 1.0 if has_skill else 0.0
                
                skill_analysis.append({
                    'skill': skill,
                    'similarity': similarity,
                    'status': status,
                    'priority': priority,
                    'skill_type': 'preferred',
                    'learning_time': self._get_learning_time(skill)
                })
            
            # Calculate overall match
            required_matches = sum(1 for s in skill_analysis 
                                 if s['skill_type'] == 'required' and s['status'] == 'strong')
            total_required = len(required_skills)
            overall_match = (required_matches / total_required * 100) if total_required > 0 else 0
            
            return {
                'overall_match_percentage': round(overall_match, 2),
                'skill_analysis': skill_analysis,
                'skill_gaps': [s for s in skill_analysis if s['status'] == 'missing'],
                'strong_skills': [s for s in skill_analysis if s['status'] == 'strong'],
                'analysis_method': 'keyword_matching'
            }
            
        except Exception as e:
            logger.error(f"Error in traditional skill analysis: {e}")
            return {
                'error': 'Skill analysis failed',
                'overall_match_percentage': 0,
                'skill_analysis': [],
                'skill_gaps': [],
                'strong_skills': []
            }
    
    def _get_learning_time(self, skill):
        """Get estimated learning time for a skill"""
        try:
            skill_lower = skill.lower()
            if skill_lower in self.skill_difficulty_map:
                return self.skill_difficulty_map[skill_lower]
            
            # Default estimates based on common patterns
            if any(keyword in skill_lower for keyword in ['basic', 'html', 'css']):
                return {'difficulty': 'beginner', 'time_weeks': 4}
            elif any(keyword in skill_lower for keyword in ['advanced', 'machine learning', 'deep learning']):
                return {'difficulty': 'advanced', 'time_weeks': 20}
            else:
                return {'difficulty': 'intermediate', 'time_weeks': 12}
                
        except Exception as e:
            logger.error(f"Error getting learning time for {skill}: {e}")
            return {'difficulty': 'intermediate', 'time_weeks': 12}
    
    def generate_learning_roadmap(self, skill_gaps, user_level='beginner', target_timeline_months=6):
        """Generate comprehensive learning roadmap with error handling"""
        try:
            if not skill_gaps:
                return {
                    'message': 'No skill gaps identified - you\'re well-prepared for this role!',
                    'phases': [],
                    'total_estimated_time': '0 months',
                    'resources': {}
                }
            
            # Sort skills by priority and learning time
            sorted_gaps = sorted(skill_gaps, key=lambda x: (
                0 if x['priority'] == 'high' else 1 if x['priority'] == 'medium' else 2,
                x.get('learning_time', {}).get('time_weeks', 12)
            ))
            
            # Create learning phases
            phases = []
            current_phase = 1
            current_phase_skills = []
            current_phase_weeks = 0
            max_phase_weeks = (target_timeline_months * 4) // 3  # Divide timeline into 3 phases
            
            for skill_gap in sorted_gaps:
                skill_weeks = skill_gap.get('learning_time', {}).get('time_weeks', 12)
                
                # If adding this skill would exceed phase limit, start new phase
                if current_phase_weeks + skill_weeks > max_phase_weeks and current_phase_skills:
                    phases.append({
                        'phase': current_phase,
                        'title': self._get_phase_title(current_phase),
                        'duration_weeks': current_phase_weeks,
                        'skills': current_phase_skills.copy(),
                        'description': self._get_phase_description(current_phase, current_phase_skills)
                    })
                    
                    current_phase += 1
                    current_phase_skills = []
                    current_phase_weeks = 0
                
                current_phase_skills.append(skill_gap['skill'])
                current_phase_weeks += skill_weeks
            
            # Add remaining skills to final phase
            if current_phase_skills:
                phases.append({
                    'phase': current_phase,
                    'title': self._get_phase_title(current_phase),
                    'duration_weeks': current_phase_weeks,
                    'skills': current_phase_skills,
                    'description': self._get_phase_description(current_phase, current_phase_skills)
                })
            
            # Add portfolio/practice phase
            phases.append({
                'phase': len(phases) + 1,
                'title': 'Portfolio Development & Job Preparation',
                'duration_weeks': 4,
                'skills': ['Project Building', 'Portfolio Creation', 'Interview Preparation', 'Networking'],
                'description': 'Apply learned skills in real projects and prepare for job applications'
            })
            
            # Generate resources
            resources = {}
            for skill_gap in skill_gaps:
                skill = skill_gap['skill'].lower()
                resources[skill] = self._get_skill_resources(skill)
            
            # Calculate total time
            total_weeks = sum(phase['duration_weeks'] for phase in phases)
            total_months = max(1, round(total_weeks / 4))
            
            return {
                'total_estimated_time': f'{total_months} months',
                'total_weeks': total_weeks,
                'phases': phases,
                'resources': resources,
                'learning_tips': self._get_learning_tips(user_level),
                'success_metrics': self._get_success_metrics()
            }
            
        except Exception as e:
            logger.error(f"Error generating learning roadmap: {e}")
            return {
                'error': 'Failed to generate roadmap',
                'total_estimated_time': '6 months',
                'phases': [],
                'resources': {}
            }
    
    def _get_phase_title(self, phase_number):
        """Get appropriate title for learning phase"""
        titles = {
            1: 'Foundation Skills',
            2: 'Core Competencies',
            3: 'Advanced Skills',
            4: 'Specialization',
            5: 'Mastery'
        }
        return titles.get(phase_number, f'Phase {phase_number}')
    
    def _get_phase_description(self, phase_number, skills):
        """Get description for learning phase"""
        if phase_number == 1:
            return f'Build foundational knowledge in {", ".join(skills[:2])} and related core concepts'
        elif phase_number == 2:
            return f'Develop practical skills in {", ".join(skills[:2])} with hands-on projects'
        else:
            return f'Master advanced concepts in {", ".join(skills[:2])} and apply to real-world scenarios'
    
    def _get_skill_resources(self, skill):
        """Get learning resources for a specific skill"""
        try:
            if skill in self.learning_resources:
                return self.learning_resources[skill]
            
            # Generate generic resources using template
            skill_title = skill.replace('_', ' ').title()
            generic_resources = {}
            
            for resource_type, templates in self.generic_resource_template.items():
                generic_resources[resource_type] = [
                    template.format(skill=skill_title) for template in templates
                ]
            
            return generic_resources
            
        except Exception as e:
            logger.error(f"Error getting resources for {skill}: {e}")
            return {
                'youtube': [f'{skill.title()} Tutorial'],
                'articles': [f'Learn {skill.title()}'],
                'practice': [f'{skill.title()} Exercises'],
                'projects': [f'{skill.title()} Project']
            }
    
    def _get_learning_tips(self, user_level):
        """Get personalized learning tips based on user level"""
        tips = {
            'beginner': [
                'Start with fundamentals and don\'t rush',
                'Practice coding every day, even if just 30 minutes',
                'Join online communities and forums for support',
                'Build small projects to apply what you learn',
                'Don\'t be afraid to ask questions'
            ],
            'intermediate': [
                'Focus on building substantial projects',
                'Contribute to open source projects',
                'Learn best practices and design patterns',
                'Network with other professionals',
                'Consider getting certifications'
            ],
            'advanced': [
                'Mentor others to solidify your knowledge',
                'Stay updated with industry trends',
                'Specialize in niche areas',
                'Speak at conferences or write technical blogs',
                'Lead technical projects at work'
            ]
        }
        return tips.get(user_level, tips['intermediate'])
    
    def _get_success_metrics(self):
        """Get success metrics for tracking progress"""
        return {
            'weekly_goals': [
                'Complete assigned learning materials',
                'Build one small project or exercise',
                'Review and practice previous week\'s concepts',
                'Connect with one new person in the field'
            ],
            'monthly_milestones': [
                'Complete a substantial project',
                'Pass a relevant certification or assessment',
                'Contribute to an open source project',
                'Update portfolio with new work'
            ],
            'progress_indicators': [
                'Ability to explain concepts to others',
                'Confidence in using tools independently',
                'Recognition from peers or mentors',
                'Job interview success rate improvement'
            ]
        }

# Continue with the rest of the comprehensive system...
# [The code continues with the enhanced SkillAssessmentEngine, PersonalizedLearningSystem, and Flask application with full error handling]

class BERTEnhancedSkillAssessmentEngine:
    """Enhanced Skill Assessment Engine with comprehensive BERT integration and error handling"""
    
    def __init__(self):
        self.bert_model = None
        self.skill_gap_analyzer = None
        self.tf_model = None
        self.assessment_history = []
        self.model_performance_stats = {
            'total_assessments': 0,
            'successful_assessments': 0,
            'bert_assessments': 0,
            'traditional_assessments': 0,
            'average_confidence': 0.0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components with error handling"""
        try:
            # Initialize BERT model
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model initialized for skill assessment")
        except Exception as e:
            logger.error(f"Error initializing BERT model: {e}")
            self.bert_model = None
        
        try:
            # Initialize skill gap analyzer
            self.skill_gap_analyzer = BERTSkillGapAnalyzer()
            logger.info("Skill gap analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing skill gap analyzer: {e}")
            self.skill_gap_analyzer = None
        
        try:
            # Build TensorFlow model
            self.build_tensorflow_model()
            logger.info("TensorFlow assessment model built")
        except Exception as e:
            logger.error(f"Error building TensorFlow model: {e}")
            self.tf_model = None
    
    def build_tensorflow_model(self):
        """Build TensorFlow model for skill level assessment with error handling"""
        try:
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(10,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(5, activation='softmax')  # 5 skill levels
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.tf_model = model
            return model
            
        except Exception as e:
            logger.error(f"Error building TensorFlow model: {e}")
            self.tf_model = None
            return None
    
    def assess_skill_level_bert(self, user_responses, skill_domain, context_info=None):
        """BERT-enhanced skill level assessment with comprehensive error handling"""
        try:
            self.model_performance_stats['total_assessments'] += 1
            
            if not user_responses:
                return self._fallback_assessment(skill_domain, 'No responses provided')
            
            # Combine user responses into text
            if isinstance(user_responses, list):
                response_text = ' '.join(str(response) for response in user_responses)
            else:
                response_text = str(user_responses)
            
            if not self.bert_model:
                return self._traditional_assessment(user_responses, skill_domain)
            
            # Generate BERT embedding
            try:
                response_embedding = self.bert_model.encode([response_text], show_progress_bar=False)
                self.model_performance_stats['bert_assessments'] += 1
            except Exception as bert_error:
                logger.error(f"BERT encoding error: {bert_error}")
                return self._traditional_assessment(user_responses, skill_domain)
            
            # Create feature vector for TensorFlow model
            try:
                features = self._create_feature_vector(response_embedding[0], user_responses, context_info)
                
                # Predict skill level using TensorFlow
                if self.tf_model:
                    prediction = self.tf_model.predict([features], verbose=0)
                    skill_level = np.argmax(prediction[0])
                    confidence = float(np.max(prediction[0]))
                else:
                    # Fallback assessment based on response quality
                    skill_level, confidence = self._heuristic_assessment(response_text, user_responses)
                
            except Exception as tf_error:
                logger.error(f"TensorFlow prediction error: {tf_error}")
                skill_level, confidence = self._heuristic_assessment(response_text, user_responses)
            
            level_names = ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert']
            
            assessment_result = {
                'skill_level': level_names[skill_level],
                'level_numeric': skill_level,
                'confidence': confidence,
                'assessment_method': 'bert_enhanced',
                'skill_domain': skill_domain,
                'response_analysis': self._analyze_response_quality(response_text),
                'recommendations': self._get_skill_recommendations(skill_level, skill_domain),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update statistics
            self.model_performance_stats['successful_assessments'] += 1
            self.model_performance_stats['average_confidence'] = (
                (self.model_performance_stats['average_confidence'] * 
                 (self.model_performance_stats['successful_assessments'] - 1) + confidence) /
                self.model_performance_stats['successful_assessments']
            )
            
            # Store assessment history
            self.assessment_history.append(assessment_result)
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"Error in BERT skill assessment: {e}")
            return self._fallback_assessment(skill_domain, str(e))
    
    def _create_feature_vector(self, bert_embedding, user_responses, context_info):
        """Create feature vector for TensorFlow model"""
        try:
            # Use first 8 BERT dimensions
            bert_features = bert_embedding[:8]
            
            # Add response-based features
            response_features = [
                len(user_responses) if isinstance(user_responses, list) else 1,
                len(str(user_responses))
            ]
            
            # Combine features
            features = np.concatenate([bert_features, response_features])
            
            # Ensure feature vector has exactly 10 dimensions
            if len(features) < 10:
                features = np.pad(features, (0, 10 - len(features)), 'constant')
            elif len(features) > 10:
                features = features[:10]
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            # Return default feature vector
            return np.zeros(10)
    
    def _heuristic_assessment(self, response_text, user_responses):
        """Heuristic-based assessment as fallback"""
        try:
            # Simple heuristics based on response characteristics
            response_length = len(response_text)
            num_responses = len(user_responses) if isinstance(user_responses, list) else 1
            
            # Calculate skill level based on response quality
            if response_length < 50:
                skill_level = 0  # Beginner
                confidence = 0.6
            elif response_length < 150:
                skill_level = 1  # Basic
                confidence = 0.7
            elif response_length < 300:
                skill_level = 2  # Intermediate
                confidence = 0.75
            elif response_length < 500:
                skill_level = 3  # Advanced
                confidence = 0.8
            else:
                skill_level = 4  # Expert
                confidence = 0.85
            
            # Adjust based on number of responses
            if num_responses > 5:
                confidence = min(0.95, confidence + 0.1)
            
            return skill_level, confidence
            
        except Exception as e:
            logger.error(f"Error in heuristic assessment: {e}")
            return 1, 0.5  # Default to Basic level
    

    def _traditional_assessment(self, user_responses, skill_domain):
        """Traditional assessment method as fallback"""
        try:
            self.model_performance_stats['traditional_assessments'] += 1
            
            num_responses = len(user_responses) if isinstance(user_responses, list) else 1
            
            # Simple assessment based on response count and length
            if num_responses < 3:
                skill_level = 0  # Beginner
                confidence = 0.6
            elif num_responses < 6:
                skill_level = 1  # Basic
                confidence = 0.7
            elif num_responses < 10:
                skill_level = 2  # Intermediate
                confidence = 0.75
            else:
                skill_level = 3  # Advanced
                confidence = 0.8
            
            level_names = ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert']
            
            return {
                'skill_level': level_names[skill_level],
                'level_numeric': skill_level,
                'confidence': confidence,
                'assessment_method': 'traditional_fallback',
                'skill_domain': skill_domain
            }
            
        except Exception as e:
            logger.error(f"Error in traditional assessment: {e}")
            return {
                'skill_level': 'Basic',
                'level_numeric': 1,
                'confidence': 0.5,
                'assessment_method': 'fallback',
                'skill_domain': skill_domain
            }

    def _fallback_assessment(self, skill_domain, error_msg):
        """Final fallback assessment"""
        return {
            'skill_level': 'Basic',
            'level_numeric': 1,
            'confidence': 0.5,
            'assessment_method': 'error_fallback',
            'skill_domain': skill_domain,
            'error': error_msg
        }
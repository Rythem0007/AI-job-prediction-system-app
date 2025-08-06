import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Optional import for advanced NLP
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
import re
from collections import Counter
import streamlit as st

class TraditionalMLModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.nlp = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the model and load spaCy"""
        try:
            # Try to load spaCy model if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    st.warning("⚠️ spaCy English model not found. Using basic text processing.")
                    self.nlp = None
            else:
                self.nlp = None
            
            # Generate synthetic training data for demonstration
            self._create_training_data()
            self.is_initialized = True
            
        except Exception as e:
            raise Exception(f"Failed to initialize traditional ML model: {str(e)}")
    
    def _create_training_data(self):
        """Create synthetic training data for the model"""
        # This would typically be replaced with real training data
        sample_resumes = [
            "Python developer with 5 years experience in Django, Flask, machine learning",
            "Java software engineer with Spring Boot, microservices, REST APIs",
            "Data scientist with Python, R, machine learning, deep learning, TensorFlow",
            "Frontend developer with React, JavaScript, HTML, CSS, responsive design",
            "Full stack developer with Python, JavaScript, React, Node.js, databases",
            "DevOps engineer with AWS, Docker, Kubernetes, CI/CD, automation",
            "Mobile developer with Android, Kotlin, Java, mobile app development",
            "Backend developer with Python, Django, PostgreSQL, Redis, APIs"
        ]
        
        sample_jobs = [
            "Looking for Python developer with Django experience for web development",
            "Java developer needed for enterprise applications with Spring framework",
            "Data scientist role requiring machine learning and Python skills",
            "Frontend developer position for React and JavaScript development",
            "Full stack position requiring both frontend and backend skills",
            "DevOps engineer for cloud infrastructure and automation",
            "Android developer for mobile application development",
            "Backend developer for API development and database management"
        ]
        
        # Create features and labels
        X_features = []
        y_labels = []
        
        for i, resume in enumerate(sample_resumes):
            for j, job in enumerate(sample_jobs):
                features = self._extract_features(resume, job)
                X_features.append(features)
                # Higher similarity for matching pairs
                y_labels.append(1 if i == j else np.random.choice([0, 1], p=[0.8, 0.2]))
        
        # Train the model
        X_train = np.array(X_features)
        y_train = np.array(y_labels)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Fit TF-IDF on combined text
        combined_texts = sample_resumes + sample_jobs
        self.tfidf_vectorizer.fit(combined_texts)
    
    def _extract_features(self, resume_text, job_text):
        """Extract traditional ML features"""
        features = []
        
        # Text similarity features
        resume_clean = self._clean_text(resume_text)
        job_clean = self._clean_text(job_text)
        
        # TF-IDF similarity
        try:
            texts = [resume_clean, job_clean]
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        features.append(tfidf_similarity)
        
        # Keyword matching
        resume_words = set(resume_clean.lower().split())
        job_words = set(job_clean.lower().split())
        
        # Jaccard similarity
        intersection = len(resume_words.intersection(job_words))
        union = len(resume_words.union(job_words))
        jaccard_similarity = intersection / union if union > 0 else 0
        features.append(jaccard_similarity)
        
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'django', 'flask', 'spring', 'nodejs', 'express',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
            'aws', 'docker', 'kubernetes', 'git', 'linux',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'html', 'css', 'bootstrap', 'jquery'
        ]
        
        resume_skills = sum(1 for skill in tech_skills if skill in resume_clean.lower())
        job_skills = sum(1 for skill in tech_skills if skill in job_clean.lower())
        skill_match_ratio = resume_skills / max(job_skills, 1)
        features.append(skill_match_ratio)
        
        # Text length features
        features.append(len(resume_text.split()))
        features.append(len(job_text.split()))
        
        # Experience indicators
        experience_patterns = [r'\d+\s*years?', r'\d+\+\s*years?', r'experience', r'experienced']
        resume_exp = sum(1 for pattern in experience_patterns 
                        if re.search(pattern, resume_clean.lower()))
        job_exp = sum(1 for pattern in experience_patterns 
                     if re.search(pattern, job_clean.lower()))
        features.extend([resume_exp, job_exp])
        
        return features
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_skills(self, text):
        """Extract skills from text using pattern matching and NLP"""
        skills = set()
        text_lower = text.lower()
        
        # Technical skills dictionary
        skill_patterns = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
            'tools': ['git', 'jenkins', 'jira', 'linux', 'unix', 'bash']
        }
        
        for category, skill_list in skill_patterns.items():
            for skill in skill_list:
                if skill in text_lower:
                    skills.add(skill)
        
        # Use spaCy for entity recognition if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT']:  # Organizations and products might be technologies
                        skills.add(ent.text.lower())
            except:
                pass  # Continue without spaCy processing
        
        return list(skills)
    
    def predict(self, resume_text, job_description):
        """Make prediction using traditional ML approach"""
        if not self.is_initialized:
            raise Exception("Model not initialized. Call initialize() first.")
        
        try:
            # Extract features
            features = self._extract_features(resume_text, job_description)
            features_scaled = self.scaler.transform([features])
            
            # Get prediction probability
            match_probability = self.classifier.predict_proba(features_scaled)[0][1]
            
            # Extract skills
            resume_skills = self._extract_skills(resume_text)
            job_skills = self._extract_skills(job_description)
            
            # Find matched and missing skills
            matched_skills = list(set(resume_skills) & set(job_skills))
            missing_skills = list(set(job_skills) - set(resume_skills))
            
            # Calculate confidence based on feature importance
            feature_importance = self.classifier.feature_importances_
            confidence = np.average(np.abs(features), weights=feature_importance)
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Skill matching score
            skill_match_score = len(matched_skills) / max(len(job_skills), 1) if job_skills else 0
            
            # Combine scores
            final_score = (match_probability * 0.7 + skill_match_score * 0.3)
            
            return {
                'match_score': final_score,
                'confidence': confidence,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'resume_skills': resume_skills,
                'job_skills': job_skills,
                'tfidf_similarity': features[0] if features else 0,
                'jaccard_similarity': features[1] if len(features) > 1 else 0,
                'skill_coverage': skill_match_score,
                'model_type': 'Traditional ML (Random Forest + TF-IDF)',
                'features_used': [
                    'TF-IDF Similarity',
                    'Jaccard Similarity', 
                    'Skill Matching',
                    'Text Length Analysis',
                    'Experience Indicators'
                ]
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

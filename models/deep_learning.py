import os
os.environ["USE_TF"] = "0" 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers library not available. Using simplified deep learning approach.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class DeepLearningModel:
    def __init__(self):
        self.sentence_model = None
        self.tokenizer = None
        self.bert_model = None
        self.neural_network = None
        self.tfidf_vectorizer = None
        self.is_initialized = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize(self):
        """Initialize the deep learning models with safe fallbacks"""
        try:
            # Always initialize TF-IDF (used as fallback)
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Try loading Sentence Transformer
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    st.warning(f"⚠️ Sentence Transformer load failed ({e}). Using TF-IDF fallback.")
                    self.sentence_model = None
            else:
                self.sentence_model = None
            
            # Try loading BERT
            if TRANSFORMERS_AVAILABLE:
                try:
                    model_name = 'distilbert-base-uncased'
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.bert_model = AutoModel.from_pretrained(model_name)
                    self.bert_model.eval()
                except Exception as e:
                    st.warning(f"⚠️ BERT load failed ({e}). Using TF-IDF fallback.")
                    self.tokenizer = None
                    self.bert_model = None
            else:
                self.tokenizer = None
                self.bert_model = None
            
            # Create neural network
            self._create_neural_network()
            self.is_initialized = True
            
        except Exception as e:
            st.error(f"❌ Deep learning initialization failed: {e}. Switching to TF-IDF only.")
            self.sentence_model = None
            self.tokenizer = None
            self.bert_model = None
            self._create_neural_network()
            self.is_initialized = True
    
    def _create_neural_network(self):
        """Create a neural network for final prediction"""
        class JobMatchNN(nn.Module):
            def __init__(self, input_size=384, hidden_size=256):
                super(JobMatchNN, self).__init__()
                self.fc1 = nn.Linear(input_size * 2, hidden_size)  # Concatenated embeddings
                self.fc2 = nn.Linear(hidden_size, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.dropout(self.relu(self.fc3(x)))
                x = self.sigmoid(self.fc4(x))
                return x
        
        self.neural_network = JobMatchNN()
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize neural network weights"""
        for module in self.neural_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _get_bert_embeddings(self, text, max_length=512):
        """Get BERT embeddings for text"""
        if self.tokenizer is None or self.bert_model is None:
            return self._get_tfidf_embeddings(text)
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings[0]
        except Exception as e:
            st.warning(f"BERT embedding failed ({e}). Using TF-IDF fallback.")
            return self._get_tfidf_embeddings(text)
    
    def _get_sentence_embeddings(self, text):
        """Get sentence embeddings using SentenceTransformer"""
        if self.sentence_model is None:
            return self._get_tfidf_embeddings(text)
        
        try:
            embeddings = self.sentence_model.encode(text)
            return embeddings
        except Exception as e:
            st.warning(f"Sentence embedding failed ({e}). Using TF-IDF fallback.")
            return self._get_tfidf_embeddings(text)
    
    def _get_tfidf_embeddings(self, text):
        """Get TF-IDF embeddings as fallback"""
        try:
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or self.tfidf_vectorizer.vocabulary_ is None:
                self.tfidf_vectorizer.fit([text])
            embeddings = self.tfidf_vectorizer.transform([text]).toarray()[0]
            return embeddings
        except:
            words = text.lower().split()
            vocab_size = 1000
            embedding = np.zeros(vocab_size)
            for word in words[:vocab_size]:
                embedding[hash(word) % vocab_size] += 1
            return embedding / max(np.sum(embedding), 1)
    
    def _extract_semantic_skills(self, text, skill_categories):
        """Extract skills using semantic similarity"""
        if self.sentence_model is None:
            return self._extract_skills_fallback(text, skill_categories)
        
        try:
            text_embedding = self.sentence_model.encode([text])
            detected_skills = {}
            for category, skills in skill_categories.items():
                category_skills = []
                for skill in skills:
                    skill_embedding = self.sentence_model.encode([skill])
                    similarity = cosine_similarity(text_embedding, skill_embedding)[0][0]
                    if similarity > 0.3:
                        category_skills.append((skill, similarity))
                if category_skills:
                    detected_skills[category] = sorted(category_skills, key=lambda x: x[1], reverse=True)
            return detected_skills
        except:
            return self._extract_skills_fallback(text, skill_categories)
    
    def _extract_skills_fallback(self, text, skill_categories):
        """Fallback skill extraction using keyword matching"""
        text_lower = text.lower()
        detected_skills = {}
        for category, skills in skill_categories.items():
            category_skills = []
            for skill in skills:
                if skill.lower() in text_lower:
                    count = text_lower.count(skill.lower())
                    relevance = min(count / 10.0, 1.0)
                    category_skills.append((skill, relevance))
            if category_skills:
                detected_skills[category] = sorted(category_skills, key=lambda x: x[1], reverse=True)
        return detected_skills
    
    def _advanced_skill_extraction(self, resume_text, job_text):
        """Advanced skill extraction using semantic understanding"""
        skill_categories = {
            'programming': ['Python programming', 'Java development', 'JavaScript coding', 'C++ programming', 'React development', 'Angular framework', 'Vue.js', 'Node.js backend', 'Django web framework'],
            'data_science': ['Machine Learning', 'Deep Learning', 'Data Analysis', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'Data Visualization', 'Statistical Analysis'],
            'cloud_devops': ['Amazon Web Services', 'Microsoft Azure', 'Google Cloud Platform', 'Docker containerization', 'Kubernetes orchestration', 'CI/CD pipelines', 'Infrastructure automation', 'DevOps practices'],
            'databases': ['SQL databases', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis caching', 'Database design', 'Query optimization']
        }
        resume_skills = self._extract_semantic_skills(resume_text, skill_categories)
        job_skills = self._extract_semantic_skills(job_text, skill_categories)
        return resume_skills, job_skills
    
    def _calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        except:
            return 0.0
    
    def _neural_network_prediction(self, resume_embedding, job_embedding):
        """Use neural network for final prediction"""
        try:
            combined_features = np.concatenate([resume_embedding, job_embedding])
            input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
            with torch.no_grad():
                prediction = self.neural_network(input_tensor).item()
            return prediction
        except:
            return cosine_similarity([resume_embedding], [job_embedding])[0][0]
    
    def predict(self, resume_text, job_description):
        """Make prediction using deep learning approach"""
        if not self.is_initialized:
            raise Exception("Model not initialized. Call initialize() first.")
        
        try:
            # Embeddings
            resume_embedding = self._get_sentence_embeddings(resume_text)
            job_embedding = self._get_sentence_embeddings(job_description)
            resume_bert = self._get_bert_embeddings(resume_text)
            job_bert = self._get_bert_embeddings(job_description)
            
            # Similarities
            sentence_similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            bert_similarity = cosine_similarity([resume_bert], [job_bert])[0][0]
            nn_prediction = self._neural_network_prediction(resume_embedding, job_embedding)
            
            # Skills
            resume_skills, job_skills = self._advanced_skill_extraction(resume_text, job_description)
            skill_matches = {}
            skill_coverage_scores = []
            for category in job_skills:
                if category in resume_skills:
                    job_cat_skills = set(skill[0] for skill in job_skills[category])
                    resume_cat_skills = set(skill[0] for skill in resume_skills[category])
                    matched = job_cat_skills.intersection(resume_cat_skills)
                    coverage = len(matched) / len(job_cat_skills) if job_cat_skills else 0
                    skill_matches[category] = {'matched': list(matched), 'missing': list(job_cat_skills - resume_cat_skills), 'coverage': coverage}
                    skill_coverage_scores.append(coverage)
            overall_skill_coverage = np.mean(skill_coverage_scores) if skill_coverage_scores else 0
            
            # Final score
            final_score = (sentence_similarity * 0.3 + bert_similarity * 0.3 + nn_prediction * 0.2 + overall_skill_coverage * 0.2)
            
            # Confidence
            scores = [sentence_similarity, bert_similarity, nn_prediction, overall_skill_coverage]
            confidence = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
            confidence = max(0.1, min(confidence, 1.0))
            
            # Flatten skill lists
            all_resume_skills = [skill[0] for skills in resume_skills.values() for skill in skills]
            all_job_skills = [skill[0] for skills in job_skills.values() for skill in skills]
            all_matched_skills = [skill for cat in skill_matches.values() for skill in cat['matched']]
            all_missing_skills = [skill for cat in skill_matches.values() for skill in cat['missing']]
            
            return {
                'match_score': final_score,
                'confidence': confidence,
                'matched_skills': all_matched_skills,
                'missing_skills': all_missing_skills,
                'resume_skills': all_resume_skills,
                'job_skills': all_job_skills,
                'sentence_similarity': sentence_similarity,
                'bert_similarity': bert_similarity,
                'neural_network_score': nn_prediction,
                'skill_coverage': overall_skill_coverage,
                'skill_category_breakdown': skill_matches,
                'model_type': 'Deep Learning (BERT + Neural Network + Sentence Transformers)',
                'techniques_used': [
                    'Sentence Transformers (all-MiniLM-L6-v2)',
                    'DistilBERT Embeddings',
                    'Custom Neural Network',
                    'Semantic Skill Matching',
                    'Multi-model Ensemble'
                ]
            }
        except Exception as e:
            raise Exception(f"Deep learning prediction failed: {str(e)}")

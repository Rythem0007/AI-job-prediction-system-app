import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            lowercase=True
        )
        
        # Predefined skill categories and weights
        self.skill_categories = {
            'programming_languages': {
                'skills': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift', 'typescript'],
                'weight': 1.0
            },
            'web_technologies': {
                'skills': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring', 'bootstrap', 'jquery'],
                'weight': 0.9
            },
            'databases': {
                'skills': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite', 'cassandra'],
                'weight': 0.8
            },
            'cloud_devops': {
                'skills': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'linux', 'unix', 'terraform', 'ansible'],
                'weight': 0.9
            },
            'data_science': {
                'skills': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'r', 'matlab'],
                'weight': 1.0
            },
            'mobile_development': {
                'skills': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin'],
                'weight': 0.8
            }
        }
    
    def extract_all_features(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Extract comprehensive features from resume and job description"""
        features = {}
        
        # Basic text statistics
        features.update(self._extract_text_statistics(resume_text, job_description))
        
        # Similarity features
        features.update(self._extract_similarity_features(resume_text, job_description))
        
        # Skill matching features
        features.update(self._extract_skill_features(resume_text, job_description))
        
        # Experience features
        features.update(self._extract_experience_features(resume_text, job_description))
        
        # Education features
        features.update(self._extract_education_features(resume_text))
        
        # Keyword density features
        features.update(self._extract_keyword_features(resume_text, job_description))
        
        # Structural features
        features.update(self._extract_structural_features(resume_text, job_description))
        
        return features
    
    def _extract_text_statistics(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """Extract basic text statistics"""
        resume_words = len(resume_text.split())
        job_words = len(job_description.split())
        
        resume_chars = len(resume_text)
        job_chars = len(job_description)
        
        return {
            'resume_word_count': resume_words,
            'job_word_count': job_words,
            'word_count_ratio': resume_words / max(job_words, 1),
            'resume_char_count': resume_chars,
            'job_char_count': job_chars,
            'char_count_ratio': resume_chars / max(job_chars, 1),
            'resume_sentence_count': len(resume_text.split('.')),
            'job_sentence_count': len(job_description.split('.')),
        }
    
    def _extract_similarity_features(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """Extract various text similarity features"""
        # TF-IDF similarity
        try:
            texts = [resume_text, job_description]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Word overlap similarity
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        intersection = len(resume_words.intersection(job_words))
        union = len(resume_words.union(job_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        overlap_coefficient = intersection / min(len(resume_words), len(job_words)) if min(len(resume_words), len(job_words)) > 0 else 0
        
        return {
            'tfidf_similarity': tfidf_similarity,
            'jaccard_similarity': jaccard_similarity,
            'overlap_coefficient': overlap_coefficient,
            'common_word_count': intersection,
            'common_word_ratio': intersection / max(len(job_words), 1)
        }
    
    def _extract_skill_features(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Extract skill-based features"""
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        
        features = {}
        category_matches = {}
        
        for category, category_info in self.skill_categories.items():
            skills = category_info['skills']
            weight = category_info['weight']
            
            resume_category_skills = [skill for skill in skills if skill in resume_lower]
            job_category_skills = [skill for skill in skills if skill in job_lower]
            
            matched_skills = list(set(resume_category_skills) & set(job_category_skills))
            missing_skills = list(set(job_category_skills) - set(resume_category_skills))
            
            # Category-specific features
            features[f'{category}_resume_count'] = len(resume_category_skills)
            features[f'{category}_job_count'] = len(job_category_skills)
            features[f'{category}_matched_count'] = len(matched_skills)
            features[f'{category}_missing_count'] = len(missing_skills)
            features[f'{category}_coverage'] = len(matched_skills) / max(len(job_category_skills), 1)
            features[f'{category}_weighted_score'] = features[f'{category}_coverage'] * weight
            
            category_matches[category] = {
                'matched': matched_skills,
                'missing': missing_skills,
                'coverage': features[f'{category}_coverage']
            }
        
        # Overall skill features
        all_resume_skills = []
        all_job_skills = []
        all_matched_skills = []
        
        for category_info in category_matches.values():
            all_matched_skills.extend(category_info['matched'])
        
        for category, category_info in self.skill_categories.items():
            skills = category_info['skills']
            all_resume_skills.extend([skill for skill in skills if skill in resume_lower])
            all_job_skills.extend([skill for skill in skills if skill in job_lower])
        
        features.update({
            'total_resume_skills': len(set(all_resume_skills)),
            'total_job_skills': len(set(all_job_skills)),
            'total_matched_skills': len(set(all_matched_skills)),
            'overall_skill_coverage': len(set(all_matched_skills)) / max(len(set(all_job_skills)), 1),
            'skill_diversity_ratio': len(set(all_resume_skills)) / max(len(all_resume_skills), 1) if all_resume_skills else 0
        })
        
        features['skill_category_matches'] = category_matches
        
        return features
    
    def _extract_experience_features(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """Extract experience-related features"""
        # Experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*years?\s*with'
        ]
        
        resume_years = []
        job_years = []
        
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        
        for pattern in experience_patterns:
            resume_years.extend([int(match) for match in re.findall(pattern, resume_lower)])
            job_years.extend([int(match) for match in re.findall(pattern, job_lower)])
        
        # Experience keywords
        experience_keywords = ['experience', 'experienced', 'expertise', 'proficient', 'skilled', 'specialist']
        
        resume_exp_mentions = sum(1 for keyword in experience_keywords if keyword in resume_lower)
        job_exp_mentions = sum(1 for keyword in experience_keywords if keyword in job_lower)
        
        return {
            'resume_max_experience': max(resume_years) if resume_years else 0,
            'job_min_experience': min(job_years) if job_years else 0,
            'job_max_experience': max(job_years) if job_years else 0,
            'experience_match_ratio': (max(resume_years) if resume_years else 0) / max(max(job_years) if job_years else 1, 1),
            'resume_experience_mentions': resume_exp_mentions,
            'job_experience_mentions': job_exp_mentions,
            'experience_keyword_coverage': min(resume_exp_mentions / max(job_exp_mentions, 1), 1.0)
        }
    
    def _extract_education_features(self, resume_text: str) -> Dict[str, int]:
        """Extract education-related features"""
        education_levels = {
            'bachelor': r'\b(bachelor|b\.?tech|b\.?e|b\.?sc|b\.?a|bs|ba)\b',
            'master': r'\b(master|m\.?tech|m\.?e|m\.?sc|m\.?a|ms|ma|mba)\b',
            'phd': r'\b(phd|ph\.?d|doctorate|doctoral)\b',
            'certification': r'\b(certification|certificate|certified)\b'
        }
        
        resume_lower = resume_text.lower()
        education_features = {}
        
        for level, pattern in education_levels.items():
            count = len(re.findall(pattern, resume_lower))
            education_features[f'education_{level}'] = min(count, 1)  # Binary feature
        
        # Education score (weighted by level)
        weights = {'bachelor': 1, 'master': 2, 'phd': 3, 'certification': 0.5}
        education_score = sum(education_features[f'education_{level}'] * weights[level] 
                            for level in education_levels.keys())
        
        education_features['education_score'] = education_score
        
        return education_features
    
    def _extract_keyword_features(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """Extract keyword density and importance features"""
        # Extract important keywords from job description
        job_words = [word.lower() for word in job_description.split() 
                    if len(word) > 3 and word.isalpha()]
        job_word_freq = Counter(job_words)
        
        # Get top keywords from job description
        top_job_keywords = [word for word, _ in job_word_freq.most_common(20)]
        
        # Count how many top job keywords appear in resume
        resume_lower = resume_text.lower()
        keyword_matches = sum(1 for keyword in top_job_keywords if keyword in resume_lower)
        
        return {
            'keyword_match_count': keyword_matches,
            'keyword_match_ratio': keyword_matches / max(len(top_job_keywords), 1),
            'job_keyword_density': len(job_word_freq) / max(len(job_words), 1),
            'top_keyword_coverage': keyword_matches / min(20, len(top_job_keywords)) if top_job_keywords else 0
        }
    
    def _extract_structural_features(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """Extract structural features from the texts"""
        # Section indicators
        resume_sections = ['experience', 'education', 'skills', 'projects', 'summary', 'objective']
        resume_section_count = sum(1 for section in resume_sections 
                                 if section in resume_text.lower())
        
        # Contact information indicators
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'[\+]?[1-9]?[0-9]{2,3}[-.\s]?[0-9]{4,6}[-.\s]?[0-9]{4}',  # Phone
            r'linkedin\.com',  # LinkedIn
            r'github\.com'     # GitHub
        ]
        
        contact_info_count = sum(1 for pattern in contact_patterns 
                               if re.search(pattern, resume_text))
        
        # Bullet points and formatting
        bullet_indicators = ['-', '•', '*', '→']
        bullet_count = sum(resume_text.count(indicator) for indicator in bullet_indicators)
        
        return {
            'resume_section_count': resume_section_count,
            'contact_info_completeness': min(contact_info_count / 4, 1.0),  # Max 4 types of contact info
            'bullet_point_count': bullet_count,
            'formatting_score': min((resume_section_count + contact_info_count + min(bullet_count/10, 1)) / 3, 1.0),
            'text_structure_score': min(resume_section_count / 6, 1.0)  # 6 expected sections
        }
    
    def get_feature_importance(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance scores"""
        importance_weights = {
            'overall_skill_coverage': 0.25,
            'tfidf_similarity': 0.20,
            'keyword_match_ratio': 0.15,
            'experience_match_ratio': 0.15,
            'jaccard_similarity': 0.10,
            'education_score': 0.05,
            'formatting_score': 0.05,
            'contact_info_completeness': 0.05
        }
        
        feature_scores = {}
        for feature_name, weight in importance_weights.items():
            if feature_name in features:
                feature_scores[feature_name] = features[feature_name] * weight
        
        return feature_scores
    
    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Normalize numerical features to 0-1 range"""
        normalized = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if 'ratio' in key or 'coverage' in key or 'similarity' in key:
                    # Already normalized features
                    normalized[key] = max(0, min(value, 1))
                elif 'count' in key:
                    # Count features - normalize by reasonable maximums
                    if 'word' in key:
                        normalized[key] = min(value / 1000, 1)  # Max 1000 words
                    elif 'skill' in key:
                        normalized[key] = min(value / 20, 1)   # Max 20 skills
                    else:
                        normalized[key] = min(value / 10, 1)   # Generic max 10
                else:
                    # Other numerical features
                    normalized[key] = min(value / 10, 1) if value > 1 else value
            else:
                # Non-numerical features remain unchanged
                normalized[key] = value
        
        return normalized

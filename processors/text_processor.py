import re
import string
from typing import List, Dict, Set
import nltk
from collections import Counter

class TextProcessor:
    def __init__(self):
        # Try to download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass  # Continue without NLTK
                
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass  # Continue without stopwords
        
        # Define stopwords manually as fallback
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{2,3}[-.\s]?[0-9]{4,6}[-.\s]?[0-9]{4}', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text.lower())
        except:
            # Fallback tokenization
            tokens = text.lower().split()
        
        # Remove punctuation and empty tokens
        tokens = [token for token in tokens if token and token not in string.punctuation]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = self.stopwords
        
        return [token for token in tokens if token.lower() not in stop_words]
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """Extract top keywords from text"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens_no_stop = self.remove_stopwords(tokens)
        
        # Filter tokens (minimum length 3)
        filtered_tokens = [token for token in tokens_no_stop if len(token) >= 3]
        
        # Count frequency
        word_freq = Counter(filtered_tokens)
        
        return word_freq.most_common(top_n)
    
    def extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms and skills from text"""
        technical_patterns = [
            # Programming languages
            r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|rust|scala|kotlin|swift)\b',
            # Web technologies
            r'\b(html|css|react|angular|vue|nodejs|express|django|flask|spring|bootstrap)\b',
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite)\b',
            # Cloud and DevOps
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|linux|unix|bash)\b',
            # Data Science
            r'\b(machine\s*learning|deep\s*learning|tensorflow|pytorch|pandas|numpy|scikit-learn)\b',
            # Frameworks and Libraries
            r'\b(spring\s*boot|laravel|rails|express\.js|flask|django|react\.js|angular\.js)\b'
        ]
        
        technical_terms = set()
        text_lower = text.lower()
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            technical_terms.update(matches)
        
        return technical_terms
    
    def extract_experience_years(self, text: str) -> List[int]:
        """Extract years of experience mentioned in text"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*years?\s*with'
        ]
        
        years = []
        text_lower = text.lower()
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(match) for match in matches])
        
        return years
    
    def calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate various text similarity metrics"""
        # Clean texts
        clean_text1 = self.clean_text(text1)
        clean_text2 = self.clean_text(text2)
        
        # Tokenize
        tokens1 = set(self.tokenize(clean_text1))
        tokens2 = set(self.tokenize(clean_text2))
        
        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Overlap coefficient
        overlap_coefficient = intersection / min(len(tokens1), len(tokens2)) if min(len(tokens1), len(tokens2)) > 0 else 0
        
        # Word overlap ratio
        word_overlap = intersection / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'overlap_coefficient': overlap_coefficient,
            'word_overlap_ratio': word_overlap,
            'common_words': intersection
        }
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information from text"""
        education_patterns = [
            r'\b(bachelor|b\.?tech|b\.?e|b\.?sc|b\.?a|bs|ba)\b',
            r'\b(master|m\.?tech|m\.?e|m\.?sc|m\.?a|ms|ma|mba)\b',
            r'\b(phd|ph\.?d|doctorate|doctoral)\b',
            r'\b(diploma|certificate|certification)\b'
        ]
        
        education = []
        text_lower = text.lower()
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))
    
    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """Extract contact information from text"""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone pattern
        phone_pattern = r'[\+]?[1-9]?[0-9]{2,3}[-.\s]?[0-9]{4,6}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, text)
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9-]+'
        linkedin = re.findall(linkedin_pattern, text)
        
        # GitHub pattern
        github_pattern = r'github\.com/[A-Za-z0-9-]+'
        github = re.findall(github_pattern, text)
        
        return {
            'emails': emails,
            'phones': phones,
            'linkedin': linkedin,
            'github': github
        }
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get basic text statistics"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(text)
        
        return {
            'character_count': len(text),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'sentence_count': len(text.split('.')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
        }

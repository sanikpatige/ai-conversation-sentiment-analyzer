#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Engine
Uses VADER sentiment analysis with emotion detection and quality metrics extraction
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Optional
import re

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class EnhancedSentimentAnalyzer:
    """Advanced sentiment analyzer with emotion detection and quality metrics"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion keywords mapping
        self.emotion_keywords = {
            'joy': ['happy', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'perfect', 
                   'fantastic', 'delighted', 'thrilled', 'satisfied', 'pleased', 'glad'],
            'anger': ['angry', 'frustrated', 'annoyed', 'terrible', 'awful', 'horrible', 
                     'furious', 'mad', 'upset', 'disappointed', 'irritated', 'outraged'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'sorry',
                       'unfortunate', 'regret', 'poor', 'bad'],
            'fear': ['worried', 'concerned', 'anxious', 'scared', 'afraid', 'nervous', 'uncertain'],
            'surprise': ['surprised', 'unexpected', 'shocked', 'amazed', 'astonished', 'incredible'],
            'trust': ['reliable', 'trustworthy', 'confident', 'professional', 'competent',
                     'knowledgeable', 'helpful', 'supportive', 'dependable']
        }
        
        # Quality indicators
        self.quality_keywords = {
            'speed': ['quick', 'fast', 'slow', 'delayed', 'prompt', 'immediate', 'responsive', 
                     'rapid', 'swift', 'sluggish', 'lengthy'],
            'helpfulness': ['helpful', 'unhelpful', 'useful', 'useless', 'informative', 
                           'supportive', 'assist', 'guidance'],
            'professionalism': ['professional', 'unprofessional', 'courteous', 'rude', 'polite',
                               'respectful', 'impolite', 'friendly', 'considerate'],
            'resolution': ['solved', 'resolved', 'fixed', 'unresolved', 'pending', 'incomplete',
                          'successful', 'complete', 'finished']
        }
    
    def analyze(self, text: str, rating: Optional[int] = None) -> Dict:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            text: Text to analyze
            rating: Optional numeric rating (1-5) for hybrid analysis
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return self._empty_analysis()
        
        # Get VADER scores
        vader_scores = self.sia.polarity_scores(text)
        
        # Determine sentiment category
        sentiment = self._classify_sentiment(vader_scores['compound'], rating)
        
        # Detect emotions
        emotions = self._detect_emotions(text.lower())
        
        # Extract quality indicators
        quality_aspects = self._extract_quality_aspects(text.lower())
        
        # Extract keywords
        keywords = self._extract_keywords(text.lower())
        
        # Calculate confidence
        confidence = self._calculate_confidence(vader_scores, rating, text)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'vader_scores': {
                'positive': round(vader_scores['pos'], 3),
                'neutral': round(vader_scores['neu'], 3),
                'negative': round(vader_scores['neg'], 3),
                'compound': round(vader_scores['compound'], 3)
            },
            'emotions': emotions,
            'quality_aspects': quality_aspects,
            'keywords': keywords,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def _classify_sentiment(self, compound_score: float, rating: Optional[int] = None) -> str:
        """
        Classify sentiment using VADER compound score and optional rating
        
        Hybrid approach: If rating is provided, use it as a signal
        """
        # If rating is provided, use hybrid approach
        if rating is not None:
            if rating >= 4:
                return 'positive'
            elif rating <= 2:
                return 'negative'
            elif rating == 3:
                # For 3-star ratings, rely more on text sentiment
                if compound_score >= 0.1:
                    return 'positive'
                elif compound_score <= -0.1:
                    return 'negative'
                else:
                    return 'neutral'
        
        # Pure VADER classification
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotions present in the text"""
        detected_emotions = []
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if emotion not in detected_emotions:
                        detected_emotions.append(emotion)
                    break
        
        return detected_emotions
    
    def _extract_quality_aspects(self, text: str) -> Dict[str, List[str]]:
        """Extract quality-related aspects mentioned in feedback"""
        quality_mentions = {}
        
        for aspect, keywords in self.quality_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)
            
            if found_keywords:
                quality_mentions[aspect] = found_keywords
        
        return quality_mentions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text"""
        # Remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter keywords
        keywords = [
            word for word in words 
            if len(word) > 3 and word not in common_words
        ]
        
        # Get unique keywords (preserve order)
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)
        
        return unique_keywords[:10]  # Return top 10 keywords
    
    def _calculate_confidence(self, vader_scores: Dict, rating: Optional[int], text: str) -> float:
        """Calculate confidence score for the sentiment prediction"""
        # Base confidence from VADER compound score magnitude
        base_confidence = abs(vader_scores['compound'])
        
        # Adjust for text length (longer text = more confident)
        word_count = len(text.split())
        length_factor = min(word_count / 20, 1.0)  # Cap at 20 words
        
        # If rating matches sentiment, increase confidence
        rating_match_bonus = 0.0
        if rating is not None:
            sentiment = self._classify_sentiment(vader_scores['compound'], None)
            if (rating >= 4 and sentiment == 'positive') or \
               (rating <= 2 and sentiment == 'negative') or \
               (rating == 3 and sentiment == 'neutral'):
                rating_match_bonus = 0.15
        
        # Calculate final confidence
        confidence = min((base_confidence * 0.7) + (length_factor * 0.15) + rating_match_bonus, 1.0)
        
        return round(confidence, 2)
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis for invalid input"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'vader_scores': {
                'positive': 0.0,
                'neutral': 0.0,
                'negative': 0.0,
                'compound': 0.0
            },
            'emotions': [],
            'quality_aspects': {},
            'keywords': [],
            'text_length': 0,
            'word_count': 0
        }


# Example usage
if __name__ == '__main__':
    analyzer = EnhancedSentimentAnalyzer()
    
    # Test samples
    test_texts = [
        ("The support was absolutely fantastic! Very quick response and extremely helpful.", 5),
        ("Terrible experience. Slow response and unhelpful agent. Very disappointed.", 1),
        ("Average service. Nothing special but got the job done.", 3)
    ]
    
    print("\n" + "="*60)
    print("Sentiment Analyzer Test")
    print("="*60 + "\n")
    
    for text, rating in test_texts:
        result = analyzer.analyze(text, rating)
        print(f"Text: {text}")
        print(f"Rating: {rating} stars")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']}")
        print(f"Emotions: {', '.join(result['emotions'])}")
        print(f"VADER Compound: {result['vader_scores']['compound']}")
        print(f"Keywords: {', '.join(result['keywords'][:5])}")
        print("="*60 + "\n")

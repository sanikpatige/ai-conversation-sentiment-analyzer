#!/usr/bin/env python3
"""
AI Conversation Quality & Sentiment Analysis System
FastAPI backend with advanced NLP sentiment analysis
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import sqlite3
import json
import io
import pandas as pd

from sentiment_analyzer import EnhancedSentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="AI Conversation Sentiment Analyzer",
    description="API for conversation quality analysis with advanced NLP sentiment detection",
    version="2.0.0"
)


# Data Models
class RatingCreate(BaseModel):
    """Model for creating a new rating"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    feedback: str = Field(..., min_length=1, description="Text feedback for analysis")
    user_id: Optional[str] = Field(None, description="User who submitted rating")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional context")


class TextAnalysisRequest(BaseModel):
    """Model for analyzing text without saving"""
    text: str = Field(..., min_length=1, description="Text to analyze")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Optional rating for hybrid analysis")


class Rating(RatingCreate):
    """Model for a stored rating with analysis"""
    id: int
    timestamp: str
    sentiment_analysis: Dict


# Database Management
class Database:
    """SQLite database manager with sentiment analysis storage"""
    
    def __init__(self, db_path: str = "ratings.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                feedback TEXT NOT NULL,
                user_id TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_confidence REAL,
                vader_compound REAL,
                vader_positive REAL,
                vader_neutral REAL,
                vader_negative REAL,
                emotions TEXT,
                quality_aspects TEXT,
                keywords TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON ratings(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON ratings(rating)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON ratings(sentiment)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON ratings(conversation_id)")
        
        conn.commit()
        conn.close()
        
        print("✓ Database initialized successfully")
    
    def insert_rating(self, rating_data: RatingCreate, sentiment_analysis: Dict) -> Dict:
        """Insert a new rating with sentiment analysis"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO ratings (
                conversation_id, rating, feedback, user_id, metadata, timestamp,
                sentiment, sentiment_confidence, vader_compound, vader_positive,
                vader_neutral, vader_negative, emotions, quality_aspects, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rating_data.conversation_id,
            rating_data.rating,
            rating_data.feedback,
            rating_data.user_id,
            json.dumps(rating_data.metadata),
            timestamp,
            sentiment_analysis['sentiment'],
            sentiment_analysis['confidence'],
            sentiment_analysis['vader_scores']['compound'],
            sentiment_analysis['vader_scores']['positive'],
            sentiment_analysis['vader_scores']['neutral'],
            sentiment_analysis['vader_scores']['negative'],
            json.dumps(sentiment_analysis['emotions']),
            json.dumps(sentiment_analysis['quality_aspects']),
            json.dumps(sentiment_analysis['keywords'])
        ))
        
        rating_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "id": rating_id,
            "conversation_id": rating_data.conversation_id,
            "rating": rating_data.rating,
            "feedback": rating_data.feedback,
            "user_id": rating_data.user_id,
            "metadata": rating_data.metadata,
            "timestamp": timestamp,
            "sentiment_analysis": sentiment_analysis
      }
    
    def get_ratings(self, limit: int = 100, min_rating: Optional[int] = None,
                    max_rating: Optional[int] = None, sentiment: Optional[str] = None) -> List[Dict]:
        """Get ratings with optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM ratings WHERE 1=1"
        params = []
        
        if min_rating:
            query += " AND rating >= ?"
            params.append(min_rating)
        
        if max_rating:
            query += " AND rating <= ?"
            params.append(max_rating)
        
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._format_rating(row) for row in rows]
    
    def get_rating_by_id(self, rating_id: int) -> Optional[Dict]:
        """Get a specific rating by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM ratings WHERE id = ?", (rating_id,))
        row = cursor.fetchone()
        conn.close()
        
        return self._format_rating(row) if row else None
    
    def delete_rating(self, rating_id: int) -> bool:
        """Delete a rating"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM ratings WHERE id = ?", (rating_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_all_ratings(self) -> List[Dict]:
        """Get all ratings for export"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM ratings ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [self._format_rating(row) for row in rows]
    
    def _format_rating(self, row) -> Dict:
        """Format database row into rating dictionary"""
        rating = dict(row)
        
        # Parse JSON fields
        rating['metadata'] = json.loads(rating['metadata']) if rating.get('metadata') else {}
        rating['emotions'] = json.loads(rating['emotions']) if rating.get('emotions') else []
        rating['quality_aspects'] = json.loads(rating['quality_aspects']) if rating.get('quality_aspects') else {}
        rating['keywords'] = json.loads(rating['keywords']) if rating.get('keywords') else []
        
        # Build sentiment_analysis object
        rating['sentiment_analysis'] = {
            'sentiment': rating['sentiment'],
            'confidence': rating['sentiment_confidence'],
            'vader_scores': {
                'compound': rating['vader_compound'],
                'positive': rating['vader_positive'],
                'neutral': rating['vader_neutral'],
                'negative': rating['vader_negative']
            },
            'emotions': rating['emotions'],
            'quality_aspects': rating['quality_aspects'],
            'keywords': rating['keywords']
        }
        
        # Remove redundant fields
        for key in ['sentiment', 'sentiment_confidence', 'vader_compound', 'vader_positive',
                   'vader_neutral', 'vader_negative']:
            rating.pop(key, None)
        
        return rating


# Analytics Engine
class Analytics:
    """Analytics and statistics calculator with sentiment insights"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics"""
        ratings = self.db.get_all_ratings()
        
        if not ratings:
            return {
                "total_ratings": 0,
                "average_rating": 0,
                "rating_distribution": {},
                "sentiment_distribution": {},
                "average_confidence": 0
            }
        
        # Calculate statistics
        rating_values = [r['rating'] for r in ratings]
        avg_rating = sum(rating_values) / len(rating_values)
        
        # Rating distribution
        rating_dist = {str(i): 0 for i in range(1, 6)}
        for rating in rating_values:
            rating_dist[str(rating)] += 1
        
        # Sentiment distribution
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        confidence_sum = 0
        
        for rating in ratings:
            sentiment = rating['sentiment_analysis']['sentiment']
            sentiment_counts[sentiment] += 1
            confidence_sum += rating['sentiment_analysis']['confidence']
        
        avg_confidence = confidence_sum / len(ratings)
        
        return {
            "total_ratings": len(ratings),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_dist,
            "sentiment_distribution": sentiment_counts,
            "average_confidence": round(avg_confidence, 2)
        }
    
    def get_sentiment_summary(self) -> Dict:
        """Get detailed sentiment analysis summary"""
        ratings = self.db.get_all_ratings()
        
        if not ratings:
            return {
                "total_ratings": 0,
                "sentiment_distribution": {},
                "average_vader_compound": 0,
                "emotions_detected": {}
            }
        
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        vader_sum = 0
        emotion_counts = {}
        
        for rating in ratings:
            analysis = rating['sentiment_analysis']
            sentiment_counts[analysis['sentiment']] += 1
            vader_sum += analysis['vader_scores']['compound']
            
            # Count emotions
            for emotion in analysis['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_ratings": len(ratings),
            "sentiment_distribution": sentiment_counts,
            "average_vader_compound": round(vader_sum / len(ratings), 3),
            "average_confidence": round(sum(r['sentiment_analysis']['confidence'] for r in ratings) / len(ratings), 2),
            "emotions_detected": emotion_counts
        }
    
    def get_emotion_analysis(self) -> Dict:
        """Get detailed emotion analysis"""
        ratings = self.db.get_all_ratings()
        
        if not ratings:
            return {"total_ratings": 0, "emotions": {}}
        
        emotion_counts = {}
        emotion_by_rating = {i: {} for i in range(1, 6)}
        
        for rating in ratings:
            analysis = rating['sentiment_analysis']
            star_rating = rating['rating']
            
            for emotion in analysis['emotions']:
                # Overall count
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # By rating
                emotion_by_rating[star_rating][emotion] = \
                    emotion_by_rating[star_rating].get(emotion, 0) + 1
        
        return {
            "total_ratings": len(ratings),
            "emotion_counts": emotion_counts,
            "emotions_by_rating": emotion_by_rating
        }
    
    def get_quality_metrics(self) -> Dict:
        """Get quality aspects analysis"""
        ratings = self.db.get_all_ratings()
        
        if not ratings:
            return {"total_ratings": 0, "quality_aspects": {}}
        
        quality_counts = {
            'speed': {},
            'helpfulness': {},
            'professionalism': {},
            'resolution': {}
        }
        
        for rating in ratings:
            analysis = rating['sentiment_analysis']
            quality_aspects = analysis.get('quality_aspects', {})
            
            for aspect, keywords in quality_aspects.items():
                if aspect in quality_counts:
                    for keyword in keywords:
                        quality_counts[aspect][keyword] = \
                            quality_counts[aspect].get(keyword, 0) + 1
        
        return {
            "total_ratings": len(ratings),
            "quality_aspects": quality_counts,
            "summary": {
                "speed_mentions": sum(quality_counts['speed'].values()),
                "helpfulness_mentions": sum(quality_counts['helpfulness'].values()),
                "professionalism_mentions": sum(quality_counts['professionalism'].values()),
                "resolution_mentions": sum(quality_counts['resolution'].values())
            }
        }


# Initialize components
sentiment_analyzer = EnhancedSentimentAnalyzer()
db = Database()
analytics = Analytics(db)


# API Endpoints

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Conversation Sentiment Analyzer",
        "version": "2.0.0",
        "description": "Advanced NLP-powered conversation quality analysis",
        "docs": "/docs",
        "features": [
            "VADER sentiment analysis",
            "Emotion detection",
            "Quality metrics extraction",
            "Real-time analytics"
        ]
    }


@app.post("/ratings", response_model=Dict, status_code=201)
def create_rating(rating: RatingCreate):
    """
    Submit a new conversation rating with sentiment analysis
    
    - **conversation_id**: Unique identifier for the conversation
    - **rating**: Rating from 1 to 5 stars
    - **feedback**: Text feedback for sentiment analysis (required)
    - **user_id**: Optional user identifier
    - **metadata**: Optional additional context
    
    Returns the rating with complete sentiment analysis including:
    - Sentiment classification (positive/neutral/negative)
    - VADER scores
    - Detected emotions
    - Quality aspects mentioned
    - Keywords extracted
    """
    try:
        # Perform sentiment analysis
        sentiment_analysis = sentiment_analyzer.analyze(rating.feedback, rating.rating)
        
        # Store in database
        result = db.insert_rating(rating, sentiment_analysis)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating rating: {str(e)}")


@app.post("/analyze-text")
def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text sentiment without saving to database
    
    - **text**: Text to analyze
    - **rating**: Optional numeric rating for hybrid analysis
    
    Returns sentiment analysis results
    """
    try:
        analysis = sentiment_analyzer.analyze(request.text, request.rating)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.get("/ratings")
def get_ratings(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of ratings"),
    min_rating: Optional[int] = Query(None, ge=1, le=5, description="Minimum rating filter"),
    max_rating: Optional[int] = Query(None, ge=1, le=5, description="Maximum rating filter"),
    sentiment: Optional[str] = Query(None, regex="^(positive|neutral|negative)$", description="Filter by sentiment")
):
    """
    Get ratings with optional filters
    
    - **limit**: Maximum number of ratings (default: 100)
    - **min_rating**: Filter by minimum rating (1-5)
    - **max_rating**: Filter by maximum rating (1-5)
    - **sentiment**: Filter by sentiment (positive/neutral/negative)
    """
    try:
        ratings = db.get_ratings(
            limit=limit,
            min_rating=min_rating,
            max_rating=max_rating,
            sentiment=sentiment
        )
        return {
            "count": len(ratings),
            "ratings": ratings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching ratings: {str(e)}")


@app.get("/ratings/{rating_id}")
def get_rating(rating_id: int):
    """Get a specific rating by ID with full sentiment analysis"""
    rating = db.get_rating_by_id(rating_id)
    if not rating:
        raise HTTPException(status_code=404, detail=f"Rating {rating_id} not found")
    return rating


@app.delete("/ratings/{rating_id}")
def delete_rating(rating_id: int):
    """Delete a rating"""
    success = db.delete_rating(rating_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Rating {rating_id} not found")
    return {"message": f"Rating {rating_id} deleted successfully"}


@app.get("/analytics/summary")
def get_analytics_summary():
    """
    Get overall analytics summary
    
    Returns:
    - Total ratings
    - Average rating
    - Rating distribution
    - Sentiment distribution
    - Average confidence score
    """
    return analytics.get_summary()


@app.get("/analytics/sentiment-summary")
def get_sentiment_summary():
    """
    Get detailed sentiment analysis summary
    
    Returns:
    - Sentiment distribution
    - Average VADER compound score
    - Emotions detected across all ratings
    """
    return analytics.get_sentiment_summary()


@app.get("/analytics/emotions")
def get_emotion_analysis():
    """
    Get detailed emotion analysis
    
    Returns emotion counts and breakdown by rating level
    """
    return analytics.get_emotion_analysis()


@app.get("/analytics/quality-metrics")
def get_quality_metrics():
    """
    Get quality aspects analysis
    
    Returns mentions of speed, helpfulness, professionalism, and resolution
    """
    return analytics.get_quality_metrics()


@app.get("/export")
def export_data(format: str = Query("json", regex="^(json|csv)$", description="Export format")):
    """
    Export all ratings data with sentiment analysis
    
    - **format**: Export format - 'json' or 'csv' (default: json)
    """
    try:
        ratings = db.get_all_ratings()
        
        if format == "json":
            return JSONResponse(content={"ratings": ratings})
        
        elif format == "csv":
            # Flatten sentiment analysis for CSV
            flat_ratings = []
            for r in ratings:
                flat = {
                    'id': r['id'],
                    'conversation_id': r['conversation_id'],
                    'rating': r['rating'],
                    'feedback': r['feedback'],
                    'user_id': r['user_id'],
                    'timestamp': r['timestamp'],
                    'sentiment': r['sentiment_analysis']['sentiment'],
                    'confidence': r['sentiment_analysis']['confidence'],
                    'vader_compound': r['sentiment_analysis']['vader_scores']['compound'],
                    'emotions': ', '.join(r['sentiment_analysis']['emotions']),
                    'keywords': ', '.join(r['sentiment_analysis']['keywords'])
                }
                flat_ratings.append(flat)
            
            df = pd.DataFrame(flat_ratings)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            return StreamingResponse(
                iter([csv_buffer.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=sentiment_ratings.csv"}
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.post("/import")
def import_ratings(ratings: List[RatingCreate]):
    """
    Bulk import ratings with automatic sentiment analysis
    """
    try:
        results = []
        for rating in ratings:
            sentiment_analysis = sentiment_analyzer.analyze(rating.feedback, rating.rating)
            result = db.insert_rating(rating, sentiment_analysis)
            results.append(result)
        
        return {
            "message": f"Successfully imported {len(results)} ratings",
            "imported_count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing ratings: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sentiment_analyzer": "operational"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🤖 Starting AI Conversation Sentiment Analyzer")
    print("="*70)
    print(f"📍 API URL: http://localhost:8000")
    print(f"📚 Interactive docs: http://localhost:8000/docs")
    print(f"📖 Alternative docs: http://localhost:8000/redoc")
    print(f"🧠 NLP Engine: VADER Sentiment Analysis")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

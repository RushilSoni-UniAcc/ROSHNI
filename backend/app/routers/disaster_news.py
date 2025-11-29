"""
Disaster News Analysis API Router.
Provides endpoints for analyzing disaster-related news articles.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from app.database import get_db
from app.models.news_models import NewsState, NewsCity, Newspaper, NewsAnalysisLog
from app.dependencies import RoleChecker, get_current_user
from app.services.news_scraper import fetch_all_news
from app.services.news_selection import build_prioritized_newspaper_dicts
from app.ML.news_classifier import DisasterNewsClassifier
from app.models.user_family_models import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/disaster-news",
    tags=["Disaster News Analysis"]
)


# Pydantic Schemas
class NewsAnalysisRequest(BaseModel):
    """Request schema for news analysis."""
    state_id: int = Field(..., description="State ID", gt=0)
    city: str = Field(..., description="City name", min_length=1)
    keyword: Optional[str] = Field(None, description="Optional additional keyword filter")


class NewsArticleResult(BaseModel):
    """Individual news article result with prioritization metadata."""
    source: str
    title: str
    description: str
    link: str
    published: str
    prediction: str  # "REAL", "FAKE", or "UNAVAILABLE"
    confidence: Optional[float] = None
    disaster_keyword: Optional[str] = Field(None, description="Matched disaster keyword")
    priority_score: Optional[int] = Field(None, description="Computed relevance/priority score")


class NewsAnalysisResponse(BaseModel):
    """Response schema for news analysis."""
    success: bool
    total_articles: int
    fake_count: int
    real_count: int
    unavailable_count: int
    articles: List[NewsArticleResult]
    message: str


class StateResponse(BaseModel):
    """State response schema."""
    id: int
    name: str


class CityResponse(BaseModel):
    """City response schema."""
    id: int
    name: str


class AnalysisHistoryItem(BaseModel):
    """Analysis history item schema."""
    id: int
    city: str
    state: Optional[str]
    keyword: Optional[str]
    timestamp: str
    total_articles: int
    fake_count: int
    real_count: int

@router.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_disaster_news(
    request: NewsAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(RoleChecker(["commander"]))
):
    """Analyze disaster news selecting 1 local + up to 5 national newspapers.

    Selection logic mirrors legacy feature parity:
    - Identify local newspapers for the requested city/state (is_national = False)
    - Identify national newspapers (is_national = True)
    - Build prioritized list: [first local] + first 5 national
    - Pass combined list to scraper orchestrator
    """
    try:
        # Fake user for environments without auth wired yet
        class FakeUser:
            user_id = "test_commander_id"
        current_user = FakeUser()

        newspaper_dicts = await build_prioritized_newspaper_dicts(
            db=db,
            state_id=request.state_id,
            city_name=request.city
        )
        if not newspaper_dicts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No prioritized newspapers available (seed data missing)."
            )

        # Scrape combined sources
        try:
            articles = await fetch_all_news(newspaper_dicts, request.keyword)
        except Exception as scrape_error:
            logger.error(f"Scraping failed: {scrape_error}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"News scraping failed: {str(scrape_error)}"
            )

        if not articles:
            return NewsAnalysisResponse(
                success=True,
                total_articles=0,
                fake_count=0,
                real_count=0,
                unavailable_count=0,
                articles=[],
                message="No disaster-related news articles found for the selected location"
            )

        # Predictions
        prediction_results = []
        ml_failed = False
        try:
            logger.info(f"Starting ML prediction for {len(articles)} articles...")
            classifier = DisasterNewsClassifier()
            logger.info("Classifier instance created, preparing texts...")
            texts = [f"{a['title']} {a.get('description','')}" for a in articles]
            logger.info(f"Calling predict() with {len(texts)} texts...")
            preds = classifier.predict(texts)
            logger.info(f"Predictions received: {len(preds)} results")
            for article, pred in zip(articles, preds):
                prediction_results.append({
                    'source': article.get('newspaper_name', 'IMD'),
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published') or 'Unknown',
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence'],
                    'disaster_keyword': article.get('disaster_keyword'),
                    'priority_score': article.get('priority_score')
                })
            logger.info(f"ML prediction completed successfully: {len(prediction_results)} articles classified")
        except Exception as ml_error:
            logger.error(f"!!! ML PREDICTION FAILED !!!")
            logger.error(f"Error type: {type(ml_error).__name__}")
            logger.error(f"Error message: {str(ml_error)}", exc_info=True)
            ml_failed = True
            for article in articles:
                prediction_results.append({
                    'source': article.get('newspaper_name', 'IMD'),
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published') or 'Unknown',
                    'prediction': 'UNAVAILABLE',
                    'confidence': None,
                    'disaster_keyword': article.get('disaster_keyword'),
                    'priority_score': article.get('priority_score')
                })

        fake_count = sum(1 for r in prediction_results if r['prediction'] == 'FAKE')
        real_count = sum(1 for r in prediction_results if r['prediction'] == 'REAL')
        unavailable_count = sum(1 for r in prediction_results if r['prediction'] == 'UNAVAILABLE')

        message = "Analysis completed successfully"
        if ml_failed:
            message += " (ML prediction unavailable)"

        return NewsAnalysisResponse(
            success=True,
            total_articles=len(prediction_results),
            fake_count=fake_count,
            real_count=real_count,
            unavailable_count=unavailable_count,
            articles=[NewsArticleResult(**r) for r in prediction_results],
            message=message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# @router.post("/analyze", response_model=NewsAnalysisResponse)
# async def analyze_disaster_news(
#     request: NewsAnalysisRequest,
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(RoleChecker(["commander"]))
# ):
#     """
#     Analyze disaster news for a specific city.
#     Restricted to commander role only.
    
#     Args:
#         request: Analysis request with state, city, and optional keyword
#         db: Database session
#         current_user: Authenticated commander user
        
#     Returns:
#         Analysis results with article predictions
#     """
#     try:
#         logger.info(
#             f"News analysis request from user {current_user.user_id} "
#             f"for city: {request.city}, state_id: {request.state_id}"
#         )
        
#         # Step 1: Fetch newspapers from database with eager loading
#         query = (
#             select(Newspaper)
#             .join(NewsCity, Newspaper.city_id == NewsCity.id)
#             .join(NewsState, NewsCity.state_id == NewsState.id)
#             .where(NewsCity.name.ilike(f"%{request.city}%"))
#             .where(NewsState.id == request.state_id)
#             .options(
#                 selectinload(Newspaper.city).selectinload(NewsCity.state)
#             )
#         )
        
#         result = await db.execute(query)
#         newspapers = result.scalars().all()
        
#         if not newspapers:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"No newspapers found for city '{request.city}' in the specified state"
#             )
        
#         # Convert to dict format for scraper
#         newspaper_dicts = [
#             {
#                 'name': newspaper.name,
#                 'rss_url': newspaper.rss_url,
#                 'city': newspaper.city.name
#             }
#             for newspaper in newspapers
#         ]
        
#         logger.info(f"Found {len(newspaper_dicts)} newspapers for city '{request.city}'")
        
#         # Step 2: Scrape news articles
#         try:
#             articles = await fetch_all_news(newspaper_dicts, request.keyword)
#             logger.info(f"Scraped {len(articles)} disaster-related articles")
#         except Exception as scrape_error:
#             logger.error(f"Scraping failed: {scrape_error}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"News scraping failed: {str(scrape_error)}"
#             )
        
#         if not articles:
#             return NewsAnalysisResponse(
#                 success=True,
#                 total_articles=0,
#                 fake_count=0,
#                 real_count=0,
#                 unavailable_count=0,
#                 articles=[],
#                 message="No disaster-related news articles found for the selected location"
#             )
        
#         # Step 3: ML Prediction (with fallback)
#         prediction_results = []
#         ml_failed = False
        
#         try:
#             classifier = DisasterNewsClassifier()
#             texts = [f"{article['title']} {article['description']}" for article in articles]
#             predictions = classifier.predict(texts)
            
#             # Merge predictions with articles
#             for article, pred in zip(articles, predictions):
#                 prediction_results.append({
#                     'source': article['newspaper_name'],
#                     'title': article['title'],
#                     'description': article['description'],
#                     'link': article['link'],
#                     'published': article['published'] or 'Unknown',
#                     'prediction': pred['prediction'],
#                     'confidence': pred['confidence']
#                 })
            
#         except Exception as ml_error:
#             logger.error(f"ML prediction failed: {ml_error}", exc_info=True)
#             ml_failed = True
            
#             # Fallback: Return articles without predictions
#             for article in articles:
#                 prediction_results.append({
#                     'source': article['newspaper_name'],
#                     'title': article['title'],
#                     'description': article['description'],
#                     'link': article['link'],
#                     'published': article['published'] or 'Unknown',
#                     'prediction': 'UNAVAILABLE',
#                     'confidence': None
#                 })
        
#         # Step 4: Calculate statistics
#         fake_count = sum(1 for r in prediction_results if r['prediction'] == 'FAKE')
#         real_count = sum(1 for r in prediction_results if r['prediction'] == 'REAL')
#         unavailable_count = sum(1 for r in prediction_results if r['prediction'] == 'UNAVAILABLE')
        
#         # Step 5: Log analysis to database
#         try:
#             # Get state name
#             state_name = newspapers[0].city.state.name if newspapers else None
            
#             log_entry = NewsAnalysisLog(
#                 commander_user_id=current_user.user_id,
#                 city_name=request.city,
#                 state_name=state_name,
#                 keyword=request.keyword,
#                 timestamp=datetime.utcnow(),
#                 result_json={'articles': prediction_results},
#                 total_articles=len(prediction_results),
#                 fake_count=fake_count,
#                 real_count=real_count
#             )
#             db.add(log_entry)
#             await db.commit()
#             logger.info(f"Analysis logged with ID: {log_entry.id}")
#         except Exception as log_error:
#             logger.error(f"Failed to log analysis: {log_error}", exc_info=True)
#             # Don't fail the request if logging fails
#             await db.rollback()
        
#         # Step 6: Return results
#         message = "Analysis completed successfully"
#         if ml_failed:
#             message += " (ML prediction unavailable - model not loaded or error occurred)"
        
#         return NewsAnalysisResponse(
#             success=True,
#             total_articles=len(prediction_results),
#             fake_count=fake_count,
#             real_count=real_count,
#             unavailable_count=unavailable_count,
#             articles=[NewsArticleResult(**r) for r in prediction_results],
#             message=message
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in news analysis: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Internal server error: {str(e)}"
#         )


@router.get("/states", response_model=List[StateResponse])
async def get_news_states(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RoleChecker(["commander"]))
):
    """Get all available news states."""
    query = select(NewsState).order_by(NewsState.name)
    result = await db.execute(query)
    states = result.scalars().all()
    
    return [StateResponse(id=state.id, name=state.name) for state in states]


@router.get("/cities/{state_id}", response_model=List[CityResponse])
async def get_news_cities(
    state_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RoleChecker(["commander"]))
):
    """Get all cities for a specific state."""
    query = (
        select(NewsCity)
        .where(NewsCity.state_id == state_id)
        .order_by(NewsCity.name)
    )
    result = await db.execute(query)
    cities = result.scalars().all()
    
    return [CityResponse(id=city.id, name=city.name) for city in cities]


@router.get("/history", response_model=List[AnalysisHistoryItem])
async def get_analysis_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RoleChecker(["commander"]))
):
    """Get analysis history for the current commander."""
    query = (
        select(NewsAnalysisLog)
        .where(NewsAnalysisLog.commander_user_id == current_user.user_id)
        .order_by(NewsAnalysisLog.timestamp.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return [
        AnalysisHistoryItem(
            id=log.id,
            city=log.city_name,
            state=log.state_name,
            keyword=log.keyword,
            timestamp=log.timestamp.isoformat(),
            total_articles=log.total_articles,
            fake_count=log.fake_count,
            real_count=log.real_count
        )
        for log in logs
    ]

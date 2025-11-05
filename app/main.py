
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aita_system.aita_system import AITA, AITAConfig

app = FastAPI(
    title="AITA API - Automated Intelligent Technical Analysis",
    version="1.0.0",
    description="Production-ready API for automated chart pattern detection and trade signal generation"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AITA(AITAConfig())

class AnalyzeResponse(BaseModel):
    summary: dict
    overlay: dict

class ErrorResponse(BaseModel):
    error: str
    symbol: str
    message: str

@app.get(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a stock symbol",
    description="""
    Analyze a stock symbol and detect chart patterns including:
    - Cup & Handle (bullish)
    - Bull Flag (continuation)
    - Inverse Head & Shoulders (bullish reversal)
    - Double Bottom (bullish reversal)
    - Head & Shoulders (bearish reversal)
    - Double Top (bearish reversal)

    Returns pattern detection, trade plan with entry/stop/targets, options strategies, and overlay specification.
    """
)
def analyze(symbol: str = "AVGO"):
    """
    Analyze a stock symbol for chart patterns and generate trade signals.

    Args:
        symbol: Stock ticker symbol (e.g., NVDA, AAPL, TSLA)

    Returns:
        Analysis results including patterns, trade plan, and overlay spec
    """
    # Validate symbol
    if not symbol or symbol.strip() == "":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Invalid symbol",
                "symbol": symbol,
                "message": "Symbol cannot be empty",
                "suggestion": "Please provide a valid ticker symbol (e.g., NVDA, AAPL, TSLA)"
            }
        )

    try:
        result = engine.analyze(symbol.upper().strip())
        return AnalyzeResponse(**result)
    except RuntimeError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Symbol not found",
                "symbol": symbol,
                "message": str(e),
                "suggestion": "Please check the ticker symbol and try again with a valid symbol (e.g., NVDA, AAPL, TSLA)"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "symbol": symbol,
                "message": str(e),
                "suggestion": "Please try again or contact support if the issue persists"
            }
        )

@app.get("/", summary="API Information")
def root():
    """Root endpoint with API information"""
    return {
        "name": "AITA API",
        "version": "1.0.0",
        "description": "Automated Intelligent Technical Analysis API",
        "endpoints": {
            "/analyze": "Analyze a stock symbol for patterns and trade signals",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)",
            "/openapi.json": "OpenAPI schema"
        },
        "patterns_detected": [
            "Cup & Handle",
            "Bull Flag",
            "Inverse Head & Shoulders",
            "Double Bottom",
            "Head & Shoulders",
            "Double Top"
        ],
        "example": "GET /analyze?symbol=NVDA"
    }

@app.get("/health", summary="Health check")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AITA API", "version": "1.0.0"}


from fastapi import FastAPI
from pydantic import BaseModel
from aita_system.aita_system import AITA, AITAConfig

app = FastAPI(title="AITA API", version="0.1.0")
engine = AITA(AITAConfig())

class AnalyzeResponse(BaseModel):
    summary: dict
    overlay: dict

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(symbol: str = "AVGO"):
    result = engine.analyze(symbol)
    return AnalyzeResponse(**result)

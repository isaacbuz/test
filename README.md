
# AITA — Automated Intelligent Technical Analysis (MVP)

A production-ready starter project that analyzes symbols, auto-detects patterns (Cup & Handle, Bull Flag,
Head & Shoulders, Inverse H&S, Double Top/Bottom), builds trade plans (ENTRY/SL/TP/targets, R:R, quality scores),
proposes options structures (IV-aware), and exposes a FastAPI endpoint that also returns a **render overlay spec** for frontend charts.

> Educational only. Not financial advice.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m aita_system.aita_system  # prints a summary for AVGO

# Run API
uvicorn app.main:app --reload  # http://127.0.0.1:8000/analyze?symbol=AVGO
```

## Project Structure

```
.
├── aita_system/
│   ├── __init__.py
│   └── aita_system.py        # Analyzer, patterns, options planner, overlay spec, FastAPI (optional run)
├── app/
│   └── main.py               # API entrypoint importing analyzer
├── tests/
│   └── test_analyze.py       # Smoke test
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## API

`GET /analyze?symbol=AVGO` →

```json
{
  "summary": {
    "symbol": "AVGO",
    "price": 366.7,
    "patterns": [{"name":"CupAndHandle","score":0.84}],
    "top_sr_levels":[{"level":350.5,"touches":3}, ...],
    "plan": {
      "direction":"LONG",
      "entry": 367.2,
      "stop": 353.2,
      "targets":[382.0, 396.0],
      "risk_reward": 2.1,
      "quality":{"confluence":0.82,"volume_z":1.3,"rsi":52.4,"macd":0.43},
      "options": {"structure":"Debit Call Vertical","dte":28,"strikes":{"long":370,"short":382}, "notes":"..."}
    }
  },
  "overlay": {"layers":[ ... ]}
}
```

## Push to your GitHub repo

```bash
# from the project root
git init
git add .
git commit -m "AITA MVP: analyzer + API + tests + docs"
git branch -M main
git remote add origin https://github.com/isaacbuz/test.git
git push -u origin main
```

## Docker

```bash
docker build -t aita:latest .
docker run -p 8000:8000 aita:latest
# http://127.0.0.1:8000/analyze?symbol=AVGO
```

## Notes
- The analyzer uses: rule-based pattern detection, pivot-anchored Fibonacci, AVWAP, VPVR shelves, confluence scoring,
  and an IV-aware options planner.
- **Patterns detected**: Cup & Handle, Bull Flag, Head & Shoulders (bearish), Inverse H&S (bullish), Double Top (bearish), Double Bottom (bullish)
- Each pattern includes confidence scoring (0-100%), volume analysis, and symmetry validation
- Trade plans automatically adapt based on pattern type (LONG for bullish, SHORT for bearish, NEUTRAL otherwise)
- Extend `aita_system.py` to add more patterns (triangles, wedges), more indicators, and backtests.

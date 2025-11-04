
from aita_system.aita_system import AITA

def test_smoke():
    a = AITA()
    out = a.analyze("AVGO")
    assert "summary" in out and "overlay" in out
    plan = out["summary"]["plan"]
    assert {"entry","stop","targets","risk_reward"} <= set(plan.keys())

"""from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/about")
def about():
    return {"app": "My First API", "version": "0.1.0"}""" 

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Define energy meter model
class EnergyMeter(BaseModel):
    location: str
    power_kw: float
    timestamp: str

# Sample energy data - solar panels
solar_panels = [
    {"id": 1, "location": "Roof Panel A", "power_kw": 5.2, "status": "active"},
    {"id": 2, "location": "Roof Panel B", "power_kw": 4.8, "status": "active"},
    {"id": 3, "location": "Ground Panel C", "power_kw": 6.5, "status": "maintenance"},
]

@app.get("/")
def root():
    return {"message": "Energy Monitoring API", "version": "1.0"}

@app.get("/panels")
def get_all_panels():
    return {"total": len(solar_panels), "panels": solar_panels}

@app.get("/panels/{panel_id}")
def get_panel(panel_id: int):
    for panel in solar_panels:
        if panel["id"] == panel_id:
            return panel
    return {"error": "Panel not found"}

@app.post("/readings")
def create_reading(reading: EnergyMeter):
    return {
        "status": "recorded",
        "location": reading.location,
        "power_kw": reading.power_kw,
        "timestamp": reading.timestamp,
        "message": f"Recorded {reading.power_kw}kW from {reading.location}"
    }

@app.put("/panels/{panel_id}")
def update_panel_status(panel_id: int, status: str):
    for panel in solar_panels:
        if panel["id"] == panel_id:
            panel["status"] = status
            return {"updated": panel}
    return {"error": "Panel not found"}

@app.delete("/panels/{panel_id}")
def remove_panel(panel_id: int):
    for i, panel in enumerate(solar_panels):
        if panel["id"] == panel_id:
            removed = solar_panels.pop(i)
            return {"deleted": removed, "remaining": len(solar_panels)}
    return {"error": "Panel not found"}
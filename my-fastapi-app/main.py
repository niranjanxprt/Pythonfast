"""from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/about")
def about():
    return {"app": "My First API", "version": "0.1.0"}

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

from fastapi import FastAPI, Path
from enum import Enum

app = FastAPI()

# Basic path param — get specific meter reading
@app.get("/meters/{meter_id}")
def get_meter(meter_id: int):
    return {"meter_id": meter_id, "type": "smart_meter"}

# Path() adds validation — panel IDs between 1-1000
@app.get("/panels/{panel_id}")
def get_panel(
    panel_id: int = Path(gt=0, le=1000, description="Solar panel ID")
):
    return {"panel_id": panel_id, "status": "active"}

# Enum for energy sources
class EnergySource(str, Enum):
    solar = "solar"
    wind = "wind"
    hydro = "hydro"
    grid = "grid"

@app.get("/source/{source_type}")
def get_source_data(source_type: EnergySource):
    return {"source": source_type.value, "renewable": source_type != "grid"}"""

from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

# Sample energy readings
readings = [
    {"timestamp": "2024-01-01 08:00", "kw": 4.5, "location": "Panel A"},
    {"timestamp": "2024-01-01 09:00", "kw": 5.2, "location": "Panel A"},
    {"timestamp": "2024-01-01 10:00", "kw": 6.8, "location": "Panel B"},
]

# Pagination for large datasets
# URL: /readings?skip=0&limit=100
@app.get("/readings")
def get_readings(skip: int = 0, limit: int = 100):
    return {"total": len(readings), "data": readings[skip : skip + limit]}

# Filter by location (optional)
@app.get("/readings/search")
def search_readings(location: Optional[str] = None):
    if location:
        filtered = [r for r in readings if location.lower() in r["location"].lower()]
        return {"location": location, "count": len(filtered), "data": filtered}
    return {"data": readings}

# Advanced filtering with validation
@app.get("/readings/filter")
def filter_readings(
    min_kw: float = Query(default=0, ge=0, description="Minimum power in kW"),
    max_kw: float = Query(default=100, le=100, description="Maximum power in kW"),
    location: Optional[str] = Query(default=None, min_length=2, max_length=50),
):
    filtered = [r for r in readings if min_kw <= r["kw"] <= max_kw]
    if location:
        filtered = [r for r in filtered if location.lower() in r["location"].lower()]
    return {"filters": {"min_kw": min_kw, "max_kw": max_kw}, "results": filtered}

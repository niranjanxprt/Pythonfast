from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class PowerReading(BaseModel):
    meter_id: str
    power_kw: float = Field(gt=0, description="Power in kilowatts")
    voltage: float = Field(ge=100, le=250)
    timestamp: str

class SolarPanel(BaseModel):
    panel_id: str = Field(min_length=3, max_length=20)
    capacity_kw: float = Field(gt=0, le=50, description="Max capacity in kW")
    efficiency: float = Field(ge=0, le=100, description="Efficiency percentage")
    status: str = Field(default="active")

    @field_validator("panel_id")
    @classmethod
    def panel_id_format(cls, v):
        if not v.startswith("PANEL-"):
            raise ValueError("Panel ID must start with 'PANEL-'")
        return v.upper()

# Nested models — location with GPS
class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    address: str

class EnergyStation(BaseModel):
    station_name: str
    location: Location  # nested!
    total_capacity_mw: float
    energy_source: str = Field(default="solar")
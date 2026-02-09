import { useState, useEffect, useRef, createContext, useContext } from "react";

const ThemeCtx = createContext();
const useTheme = () => useContext(ThemeCtx);

const light = {
  bg: "#fff", bg2: "#fafafa", bg3: "#f5f5f5", bgCode: "#f5f5f5",
  text: "#222", text2: "#555", text3: "#888",
  border: "#e0e0e0", borderLight: "#eee",
  accent: "#1565c0", accentBg: "#e3f2fd",
  green: "#2e7d32", greenBg: "#e8f5e9",
  red: "#c62828", redBg: "#ffebee",
  noteBg: "#fff8e1", noteBorder: "#ffe082", noteText: "#555",
  quizBg: "#f8f9fa", codeBtnBg: "#fff",
};
const dark = {
  bg: "#16181d", bg2: "#1c1f26", bg3: "#22252d", bgCode: "#1c1f26",
  text: "#d4d4d4", text2: "#999", text3: "#666",
  border: "#2e3138", borderLight: "#2e3138",
  accent: "#5b9cf6", accentBg: "rgba(91,156,246,0.12)",
  green: "#66bb6a", greenBg: "rgba(102,187,106,0.12)",
  red: "#ef5350", redBg: "rgba(239,83,80,0.1)",
  noteBg: "rgba(255,183,77,0.08)", noteBorder: "#5a4a20", noteText: "#c8b06a",
  quizBg: "#1c1f26", codeBtnBg: "#22252d",
};

const CH = [
  {
    id: 1, title: "What is FastAPI & Setup",
    content: [
      { t: "p", v: "FastAPI is a modern Python web framework for building APIs. It's built on Starlette (web handling) and Pydantic (data validation). It's as fast as Node.js and Go." },
      { t: "p", v: "Why FastAPI? Auto-generated docs, type-hint validation, async support, and 40% fewer bugs from the Python type system." },
      { t: "code", label: "Install FastAPI (pip)", v: `# Create project & virtual environment
mkdir my-fastapi-app && cd my-fastapi-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install FastAPI and uvicorn server
pip install fastapi
pip install "uvicorn[standard]"` },
      { t: "code", label: "Install FastAPI (uv) — Faster!", v: `# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project with uv
mkdir my-fastapi-app && cd my-fastapi-app
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install FastAPI and uvicorn with uv (10-100x faster than pip!)
uv pip install fastapi
uv pip install "uvicorn[standard]"` },
      { t: "code", label: "main.py — Your First API", v: `from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/about")
def about():
    return {"app": "My First API", "version": "0.1.0"}` },
      { t: "code", label: "Run the server", v: `uvicorn main:app --reload

# main  = the file main.py
# app   = the FastAPI() instance
# --reload = auto-restart on code changes` },
      { t: "note", v: "Visit http://127.0.0.1:8000/docs for auto-generated Swagger UI. You can test every endpoint right in the browser!" },
    ],
  },
  {
    id: 2, title: "Routing & HTTP Methods",
    content: [
      { t: "p", v: "HTTP methods map to CRUD operations: GET (read), POST (create), PUT (update), DELETE (remove). Let's build an energy monitoring API to track solar panels and power consumption." },
      { t: "code", label: "main.py — Energy API with CRUD", v: `from fastapi import FastAPI
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
    return {"error": "Panel not found"}` },
      { t: "note", v: "Real-world tip: Energy APIs often need timestamps, location tracking, and status monitoring. FastAPI validates all these automatically with type hints!" },
    ],
  },
  {
    id: 3, title: "Path & Query Parameters",
    content: [
      { t: "p", v: "Path parameters are part of the URL (/panels/{panel_id}). Query parameters filter results (/readings?start_date=2024-01-01&limit=100). Perfect for energy data queries!" },
      { t: "code", label: "Path Parameters — Energy Meters", v: `from fastapi import FastAPI, Path
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
    return {"source": source_type.value, "renewable": source_type != "grid"}` },
      { t: "code", label: "Query Parameters — Filter Energy Data", v: `from fastapi import FastAPI, Query
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
    return {"filters": {"min_kw": min_kw, "max_kw": max_kw}, "results": filtered}` },
      { t: "note", v: "Energy monitoring often needs date ranges, location filters, and power thresholds. Query params are perfect for this!" },
    ],
  },
  {
    id: 4, title: "Request Bodies & Pydantic v2",
    content: [
      { t: "p", v: "When clients send energy data (POST/PUT), FastAPI uses Pydantic models to validate everything: power readings, timestamps, locations. Pydantic v2 auto-validates and gives clear error messages." },
      { t: "code", label: "models.py — Energy Data Models", v: `from pydantic import BaseModel, Field, field_validator
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
    energy_source: str = Field(default="solar")` },
      { t: "code", label: "main.py — Use energy models in endpoints", v: `from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

# Define energy reading model
class PowerReading(BaseModel):
    meter_id: str
    power_kw: float = Field(gt=0)
    voltage: float = Field(ge=100, le=250)
    timestamp: str

@app.post("/readings")
def submit_reading(reading: PowerReading):
    result = reading.model_dump()  # Pydantic v2: convert to dict
    # Calculate additional metrics
    result["power_mw"] = reading.power_kw / 1000
    result["status"] = "normal" if 200 <= reading.voltage <= 240 else "alert"
    return result

# Separate models for input vs output
class PanelCreate(BaseModel):
    location: str
    capacity_kw: float

class PanelResponse(BaseModel):
    id: int
    location: str
    capacity_kw: float
    status: str = "active"
    total_generated_kwh: float = 0.0

@app.post("/panels", response_model=PanelResponse)
def install_panel(panel: PanelCreate):
    return PanelResponse(
        id=101,
        location=panel.location,
        capacity_kw=panel.capacity_kw
    )` },
      { t: "note", v: "Send invalid data and FastAPI auto-returns a 422 error with details. You write zero validation code. Pydantic v2 tip: use model_dump() not dict()." },
    ],
  },
  {
    id: 5, title: "Response Models & Status Codes",
    content: [
      { t: "p", v: "response_model controls what fields the API returns (hides things like passwords). Status codes tell the client what happened (200 OK, 201 Created, 404 Not Found)." },
      { t: "code", label: "main.py", v: `from fastapi import FastAPI, status
from pydantic import BaseModel

app = FastAPI()

class UserIn(BaseModel):
    username: str
    email: str
    password: str  # client sends this

class UserOut(BaseModel):
    username: str
    email: str     # password NOT included!

@app.post(
    "/users",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
)
def create_user(user: UserIn):
    return user  # password is auto-stripped by UserOut!

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    pass

class ItemOut(BaseModel):
    id: int
    name: str

@app.get("/items", response_model=list[ItemOut])
def list_items():
    return [
        {"id": 1, "name": "Laptop", "secret": "hidden"},
    ]  # "secret" is stripped by ItemOut` },
      { t: "note", v: "Common codes: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 422 Validation Error." },
    ],
  },
  {
    id: 6, title: "Dependency Injection",
    content: [
      { t: "p", v: "Depends() lets you declare shared logic that FastAPI auto-provides to endpoints. Use it for DB connections, auth checks, shared parameters, and config." },
      { t: "code", label: "main.py", v: `from fastapi import FastAPI, Depends, Query, Header, HTTPException

app = FastAPI()

# Shared query params
def common_params(skip: int = Query(default=0, ge=0), limit: int = Query(default=10, le=100)):
    return {"skip": skip, "limit": limit}

@app.get("/items")
def read_items(commons: dict = Depends(common_params)):
    return {"params": commons}

@app.get("/users")
def read_users(commons: dict = Depends(common_params)):
    return {"params": commons}

# Auth dependency
def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "secret-123":
        raise HTTPException(status_code=401, detail="Bad key")
    return x_api_key

@app.get("/protected")
def protected(key: str = Depends(verify_api_key)):
    return {"message": "You have access!"}

# DB dependency with cleanup (yield)
class FakeDB:
    def __init__(self):
        self.items = ["Apple", "Banana"]

def get_db():
    db = FakeDB()
    try:
        yield db  # FastAPI handles cleanup
    finally:
        print("DB closed")

@app.get("/db-items")
def get_items(db: FakeDB = Depends(get_db)):
    return {"items": db.items}

# Nested: verify_key -> get_user -> require_admin
def get_current_user(key: str = Depends(verify_api_key)):
    return {"username": "niranjan", "role": "admin"}

def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admin only!")
    return user

@app.get("/admin")
def admin(a: dict = Depends(require_admin)):
    return {"welcome": a["username"]}` },
      { t: "note", v: "yield dependencies auto-cleanup (like closing DB). Dependencies can nest: A depends on B depends on C. Easy to test by overriding with app.dependency_overrides." },
    ],
  },
  {
    id: 7, title: "Error Handling",
    content: [
      { t: "p", v: "FastAPI gives you HTTPException for quick errors, custom exception classes for structured errors, and global exception handlers to customize error formats." },
      { t: "code", label: "main.py", v: `from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()
items = {"1": "Laptop", "2": "Mouse"}

# Basic HTTPException
@app.get("/items/{item_id}")
def get_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    return {"item": items[item_id]}

# Custom exception class
class ItemNotFoundError(Exception):
    def __init__(self, item_id: str):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "item_not_found", "message": f"'{exc.item_id}' doesn't exist"},
    )

# Override default validation errors
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": str(e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": errors})

@app.post("/purchase/{item_id}")
def purchase(item_id: str, quantity: int):
    if item_id not in items:
        raise ItemNotFoundError(item_id)
    return {"bought": quantity, "item": items[item_id]}` },
      { t: "note", v: "Best practice: consistent error format like {error, message, details} across your whole API." },
    ],
  },
  {
    id: 8, title: "Middleware & CORS",
    content: [
      { t: "p", v: "Middleware runs before/after every request. CORS middleware is essential when your React frontend (localhost:5173/5174) calls your FastAPI backend (localhost:8000)." },
      { t: "code", label: "main.py", v: `import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS — required for React + FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite (default)
        "http://localhost:5174",  # Vite (alternate port)
        "http://localhost:3000",  # Create React App
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware: request timing
@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}s"
    return response

@app.get("/")
def root():
    return {"message": "Hello World"}` },
      { t: "note", v: "CORS is the #1 issue connecting React to FastAPI. If you see 'CORS policy' errors in browser console, add CORSMiddleware with your frontend URL." },
    ],
  },
  {
    id: 9, title: "JWT Authentication",
    content: [
      { t: "p", v: "Auth flow: Client sends username+password to /token, gets a JWT back, then sends it as 'Authorization: Bearer <token>' on every request." },
      { t: "code", label: "Install dependencies", v: `# With pip
pip install "python-jose[cryptography]" "passlib[bcrypt]"

# With uv (faster)
uv pip install "python-jose[cryptography]" "passlib[bcrypt]"` },
      { t: "code", label: "auth.py", v: `from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"

pwd = CryptContext(schemes=["bcrypt"])
oauth2 = OAuth2PasswordBearer(tokenUrl="token")

users_db = {
    "niranjan": {
        "username": "niranjan",
        "email": "nr@example.com",
        "hashed_pw": pwd.hash("secret123"),
        "role": "admin",
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str
    role: str

def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(minutes=30)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    user = users_db.get(username)
    if not user:
        raise HTTPException(401, "User not found")
    return User(**user)

app = FastAPI()

@app.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form.username)
    if not user or not pwd.verify(form.password, user["hashed_pw"]):
        raise HTTPException(401, "Wrong credentials")
    return {"access_token": create_token({"sub": user["username"]}), "token_type": "bearer"}

@app.get("/me", response_model=User)
async def me(user: User = Depends(get_current_user)):
    return user` },
      { t: "note", v: "Never hardcode SECRET_KEY. Use bcrypt for passwords. Short token expiry (15-30 min). The /docs page has a built-in Authorize button for testing JWT." },
    ],
  },
  {
    id: 10, title: "Background Tasks & Uploads",
    content: [
      { t: "p", v: "BackgroundTasks run code after sending the response (emails, logs). UploadFile handles file uploads with streaming." },
      { t: "code", label: "main.py", v: `from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from pathlib import Path
import shutil

app = FastAPI()

def send_email(email: str, msg: str):
    import time
    time.sleep(3)
    print(f"Email sent to {email}: {msg}")

@app.post("/register")
def register(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": f"Registered {email}!"}  # instant response

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "size": file.size}

@app.post("/upload-many")
async def upload_many(title: str = Form(...), files: list[UploadFile] = File(...)):
    names = []
    for f in files:
        with open(UPLOAD_DIR / f.filename, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        names.append(f.filename)
    return {"title": title, "files": names}` },
      { t: "note", v: "BackgroundTasks = lightweight (same process). For heavy work like ML inference, use Celery + Redis instead." },
    ],
  },
  {
    id: 11, title: "WebSockets",
    content: [
      { t: "p", v: "WebSockets keep a connection open for real-time two-way communication. Use for chat, live dashboards, notifications." },
      { t: "code", label: "main.py — Server", v: `from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/echo")
async def echo(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client left")

class ChatManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)

    async def broadcast(self, msg: str):
        for conn in self.connections:
            await conn.send_text(msg)

chat = ChatManager()

@app.websocket("/ws/chat/{username}")
async def chat_ws(ws: WebSocket, username: str):
    await chat.connect(ws)
    await chat.broadcast(f"{username} joined!")
    try:
        while True:
            data = await ws.receive_text()
            await chat.broadcast(f"{username}: {data}")
    except WebSocketDisconnect:
        chat.disconnect(ws)
        await chat.broadcast(f"{username} left")` },
      { t: "code", label: "React client", v: `import { useState, useEffect, useRef } from 'react';

function Chat({ username }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(\`ws://localhost:8000/ws/chat/\${username}\`);
    ws.current.onmessage = (e) => setMessages(prev => [...prev, e.data]);
    return () => ws.current?.close();
  }, [username]);

  const send = () => {
    if (input) { ws.current.send(input); setInput(''); }
  };

  return (
    <div>
      {messages.map((m, i) => <p key={i}>{m}</p>)}
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={send}>Send</button>
    </div>
  );
}` },
    ],
  },
  {
    id: 12, title: "Data Analysis with Pandas & NumPy",
    content: [
      { t: "p", v: "Real energy APIs need data analysis. Use pandas for CSV/Excel files, time series, and aggregations. Use numpy for numerical computations. FastAPI returns pandas DataFrames as JSON automatically!" },
      { t: "code", label: "Install data science libraries", v: `# With pip
pip install pandas numpy

# With uv (faster)
uv pip install pandas numpy` },
      { t: "code", label: "main.py — Load and analyze energy data", v: `from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from typing import Dict, List

app = FastAPI()

# Load energy consumption data (CSV or create sample)
# In production, load from: pd.read_csv("energy_data.csv")
energy_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
    "solar_kw": np.random.uniform(0, 8, 24),
    "wind_kw": np.random.uniform(2, 12, 24),
    "consumption_kw": np.random.uniform(5, 15, 24),
})

@app.get("/")
def root():
    return {"message": "Energy Data Analysis API", "records": len(energy_df)}

@app.get("/data/summary")
def get_summary():
    """Statistical summary of energy data"""
    summary = energy_df[["solar_kw", "wind_kw", "consumption_kw"]].describe()
    return summary.to_dict()

@app.get("/data/recent")
def get_recent_readings(limit: int = 10):
    """Get most recent energy readings"""
    recent = energy_df.tail(limit)
    return recent.to_dict(orient="records")

@app.get("/data/hourly-average")
def hourly_average():
    """Calculate hourly averages"""
    avg = {
        "solar_avg_kw": float(energy_df["solar_kw"].mean()),
        "wind_avg_kw": float(energy_df["wind_kw"].mean()),
        "consumption_avg_kw": float(energy_df["consumption_kw"].mean()),
        "total_generation_avg": float(
            energy_df["solar_kw"].mean() + energy_df["wind_kw"].mean()
        ),
    }
    return avg` },
      { t: "code", label: "Advanced analytics — Time series & filters", v: `@app.get("/analytics/peak-hours")
def peak_hours():
    """Find peak generation and consumption hours"""
    energy_df["total_generation"] = energy_df["solar_kw"] + energy_df["wind_kw"]

    peak_gen = energy_df.loc[energy_df["total_generation"].idxmax()]
    peak_consumption = energy_df.loc[energy_df["consumption_kw"].idxmax()]

    return {
        "peak_generation": {
            "time": str(peak_gen["timestamp"]),
            "total_kw": float(peak_gen["total_generation"]),
            "solar_kw": float(peak_gen["solar_kw"]),
            "wind_kw": float(peak_gen["wind_kw"]),
        },
        "peak_consumption": {
            "time": str(peak_consumption["timestamp"]),
            "consumption_kw": float(peak_consumption["consumption_kw"]),
        },
    }

@app.get("/analytics/filter")
def filter_data(min_solar: float = 0, min_wind: float = 0):
    """Filter data by minimum generation thresholds"""
    filtered = energy_df[
        (energy_df["solar_kw"] >= min_solar) & (energy_df["wind_kw"] >= min_wind)
    ]
    return {
        "filters": {"min_solar_kw": min_solar, "min_wind_kw": min_wind},
        "matching_records": len(filtered),
        "data": filtered.to_dict(orient="records"),
    }

@app.get("/analytics/correlation")
def correlation_analysis():
    """Correlation between energy sources and consumption"""
    corr_matrix = energy_df[["solar_kw", "wind_kw", "consumption_kw"]].corr()
    return corr_matrix.to_dict()

@app.get("/analytics/numpy-stats")
def numpy_statistics():
    """Advanced stats using NumPy"""
    solar = energy_df["solar_kw"].values
    wind = energy_df["wind_kw"].values

    return {
        "solar": {
            "mean": float(np.mean(solar)),
            "median": float(np.median(solar)),
            "std_dev": float(np.std(solar)),
            "percentile_90": float(np.percentile(solar, 90)),
        },
        "wind": {
            "mean": float(np.mean(wind)),
            "median": float(np.median(wind)),
            "std_dev": float(np.std(wind)),
            "percentile_90": float(np.percentile(wind, 90)),
        },
    }` },
      { t: "note", v: "Pro tip: Always convert numpy/pandas types to Python types (float(), int(), str()) before returning. Use .to_dict(orient='records') for DataFrames." },
    ],
  },
  {
    id: 13, title: "Testing with pytest",
    content: [
      { t: "p", v: "TestClient lets you test endpoints without running the server." },
      { t: "code", label: "Install test dependencies", v: `# With pip
pip install pytest httpx

# With uv (faster)
uv pip install pytest httpx` },
      { t: "code", label: "test_main.py", v: `from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello, World!"}

def test_not_found():
    r = client.get("/items/99999")
    assert r.status_code == 404

def test_create_item():
    r = client.post("/items", json={"name": "Widget", "price": 9.99})
    assert r.status_code == 201

def test_auth_no_token():
    r = client.get("/me")
    assert r.status_code == 401

def test_auth_with_token():
    login = client.post("/token", data={"username": "niranjan", "password": "secret123"})
    token = login.json()["access_token"]
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200

# Override dependencies for testing
def fake_db():
    return {"items": ["test-item"]}

def test_with_fake_db():
    from main import get_db
    app.dependency_overrides[get_db] = fake_db
    r = client.get("/db-items")
    assert r.status_code == 200
    app.dependency_overrides.clear()

def test_websocket():
    with client.websocket_connect("/ws/echo") as ws:
        ws.send_text("Hello")
        assert ws.receive_text() == "Echo: Hello"` },
      { t: "note", v: "Run: pytest -v (verbose), pytest --cov (coverage), pytest -x (stop on first fail). Use dependency_overrides to swap real DBs for test DBs." },
    ],
  },
];

const quizzes = {
  1: { q: "What starts the FastAPI dev server?", o: ["python main.py", "flask run", "uvicorn main:app --reload", "npm start"], a: 2 },
  2: { q: "Which HTTP method records new energy data?", o: ["GET", "POST", "PUT", "DELETE"], a: 1 },
  3: { q: "How does FastAPI tell path from query params?", o: ["Decorators", "If name is in URL path = path param, else query", "All are query by default", "Must use Path()/Query()"], a: 1 },
  4: { q: "What replaces .dict() in Pydantic v2?", o: [".to_dict()", ".model_dump()", ".json()", ".serialize()"], a: 1 },
  5: { q: "What does response_model do?", o: ["Validates input", "Sets HTTP method", "Filters response output", "Creates DB tables"], a: 2 },
  6: { q: "What keyword auto-cleans up deps?", o: ["return", "yield", "async", "finally"], a: 1 },
  7: { q: "Status code for validation errors?", o: ["400", "404", "422", "500"], a: 2 },
  8: { q: "What middleware connects React to FastAPI?", o: ["Auth", "CORS", "Session", "GZip"], a: 1 },
  9: { q: "Where does the client send JWT?", o: ["Query param", "Cookie", "Authorization: Bearer header", "Body"], a: 2 },
  10: { q: "When do BackgroundTasks run?", o: ["Before response", "After response sent", "Separate process", "On startup"], a: 1 },
  11: { q: "What accepts a WebSocket?", o: ["ws.open()", "ws.connect()", "await ws.accept()", "ws.start()"], a: 2 },
  12: { q: "Convert pandas DataFrame to JSON?", o: [".to_json()", ".to_dict(orient='records')", ".serialize()", ".export()"], a: 1 },
  13: { q: "Override deps in tests how?", o: ["mock.patch()", "app.dependency_overrides[dep] = fake", "@override", "pytest.fixture()"], a: 1 },
};

function Quiz({ id }) {
  const T = useTheme();
  const q = quizzes[id];
  const [sel, setSel] = useState(null);
  const [done, setDone] = useState(false);
  useEffect(() => { setSel(null); setDone(false); }, [id]);
  if (!q) return null;
  const ok = sel === q.a;
  return (
    <div style={{ marginTop: 20, padding: 16, background: T.quizBg, borderRadius: 8, border: `1px solid ${T.border}` }}>
      <div style={{ fontWeight: 600, marginBottom: 10, color: T.text }}>Quiz: {q.q}</div>
      {q.o.map((opt, i) => {
        let bg = T.bg, bd = T.border, cl = T.text;
        if (done && i === q.a) { bg = T.greenBg; bd = T.green; cl = T.green; }
        else if (done && i === sel && !ok) { bg = T.redBg; bd = T.red; cl = T.red; }
        else if (!done && i === sel) { bg = T.accentBg; bd = T.accent; cl = T.accent; }
        return (
          <button key={i} onClick={() => !done && setSel(i)} style={{
            display: "block", width: "100%", textAlign: "left",
            padding: "8px 12px", marginBottom: 4, borderRadius: 6,
            cursor: done ? "default" : "pointer",
            border: `1px solid ${bd}`, background: bg, color: cl, fontSize: 14,
          }}>
            {String.fromCharCode(65 + i)}. {opt}
          </button>
        );
      })}
      {sel !== null && !done && (
        <button onClick={() => setDone(true)} style={{
          marginTop: 8, padding: "6px 16px", borderRadius: 6,
          border: "none", background: T.accent, color: "#fff", cursor: "pointer", fontSize: 13,
        }}>Check</button>
      )}
      {done && (
        <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, fontSize: 13,
          background: ok ? T.greenBg : T.redBg, color: ok ? T.green : T.red }}>
          {ok ? "Correct!" : `Answer: ${String.fromCharCode(65 + q.a)}. ${q.o[q.a]}`}
        </div>
      )}
    </div>
  );
}

function CodeBlock({ code, label }) {
  const T = useTheme();
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ marginBottom: 12 }}>
      {label && <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4, color: T.text2 }}>{label}</div>}
      <div style={{ position: "relative", background: T.bgCode, border: `1px solid ${T.border}`, borderRadius: 6, overflow: "hidden" }}>
        <button onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
          style={{ position: "absolute", top: 6, right: 6, padding: "2px 8px", fontSize: 11,
            border: `1px solid ${T.border}`, borderRadius: 4,
            background: copied ? T.greenBg : T.codeBtnBg, color: T.text2, cursor: "pointer" }}>
          {copied ? "Copied" : "Copy"}
        </button>
        <pre style={{ margin: 0, padding: "12px 14px", fontSize: 13, lineHeight: 1.5, overflow: "auto", fontFamily: "monospace", color: T.text }}>{code}</pre>
      </div>
    </div>
  );
}

export default function App() {
  const [ch, setCh] = useState(0);
  const [nav, setNav] = useState(true);
  const [isDark, setIsDark] = useState(false);
  const mainRef = useRef(null);
  useEffect(() => { mainRef.current?.scrollTo({ top: 0, behavior: "smooth" }); }, [ch]);

  const T = isDark ? dark : light;
  const c = CH[ch];

  return (
    <ThemeCtx.Provider value={T}>
      <div style={{
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        height: "100vh", display: "flex", flexDirection: "column",
        background: T.bg, color: T.text, transition: "background 0.3s, color 0.3s",
      }}>
        {/* Header */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "8px 16px", borderBottom: `1px solid ${T.border}`, flexShrink: 0, background: T.bg2,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={() => setNav(!nav)} style={{ background: "none", border: "none", fontSize: 18, cursor: "pointer", color: T.text3 }}>
              {nav ? "✕" : "☰"}
            </button>
            <span style={{ fontWeight: 700, fontSize: 16, color: T.text }}>FastAPI Tutorial</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 120, height: 4, background: T.border, borderRadius: 2, overflow: "hidden" }}>
              <div style={{ height: "100%", background: T.accent, borderRadius: 2, width: `${((ch + 1) / CH.length) * 100}%`, transition: "width 0.3s" }} />
            </div>
            <span style={{ fontSize: 12, color: T.text3 }}>{ch + 1}/{CH.length}</span>
            <button onClick={() => setIsDark(!isDark)} style={{
              background: "none", border: `1px solid ${T.border}`, borderRadius: 6,
              padding: "4px 10px", cursor: "pointer", fontSize: 14, color: T.text2,
            }}>
              {isDark ? "☀️" : "🌙"}
            </button>
          </div>
        </div>

        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {/* Sidebar */}
          {nav && (
            <div style={{
              width: 220, minWidth: 220, borderRight: `1px solid ${T.border}`,
              overflowY: "auto", padding: "8px 0", flexShrink: 0, background: T.bg2,
            }}>
              {CH.map((item, i) => (
                <button key={item.id} onClick={() => setCh(i)} style={{
                  display: "block", width: "100%", textAlign: "left", padding: "8px 14px",
                  border: "none", cursor: "pointer",
                  background: i === ch ? T.accentBg : "transparent",
                  color: i === ch ? T.accent : T.text2,
                  fontWeight: i === ch ? 600 : 400, fontSize: 13,
                  borderLeft: i === ch ? `3px solid ${T.accent}` : "3px solid transparent",
                }}>
                  {i < ch ? "✓ " : ""}{item.id}. {item.title}
                </button>
              ))}
            </div>
          )}

          {/* Content */}
          <main ref={mainRef} style={{ flex: 1, overflowY: "auto", padding: "24px 32px 60px" }}>
            <div style={{ maxWidth: 700 }}>
              <div style={{ marginBottom: 24, paddingBottom: 16, borderBottom: `1px solid ${T.border}` }}>
                <div style={{ fontSize: 12, color: T.text3, marginBottom: 2 }}>Chapter {c.id} of {CH.length}</div>
                <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0, color: T.text }}>{c.title}</h1>
              </div>

              {c.content.map((item, i) => {
                if (item.t === "p") return <p key={i} style={{ fontSize: 15, lineHeight: 1.7, marginBottom: 14, color: T.text }}>{item.v}</p>;
                if (item.t === "code") return <CodeBlock key={i} code={item.v} label={item.label} />;
                if (item.t === "note") return (
                  <div key={i} style={{
                    padding: "10px 14px", background: T.noteBg, border: `1px solid ${T.noteBorder}`,
                    borderRadius: 6, fontSize: 13, lineHeight: 1.6, marginBottom: 12, color: T.noteText,
                  }}>
                    💡 {item.v}
                  </div>
                );
                return null;
              })}

              <Quiz id={c.id} />

              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 28, paddingTop: 16, borderTop: `1px solid ${T.border}` }}>
                <button onClick={() => setCh(Math.max(0, ch - 1))} disabled={ch === 0} style={{
                  padding: "8px 18px", borderRadius: 6, border: `1px solid ${T.border}`, background: T.bg,
                  color: ch === 0 ? T.text3 : T.text2, cursor: ch === 0 ? "default" : "pointer", fontSize: 13,
                }}>← Previous</button>
                <button onClick={() => setCh(Math.min(CH.length - 1, ch + 1))} disabled={ch === CH.length - 1} style={{
                  padding: "8px 18px", borderRadius: 6, border: "none",
                  background: ch === CH.length - 1 ? T.border : T.accent, color: "#fff",
                  cursor: ch === CH.length - 1 ? "default" : "pointer", fontSize: 13,
                }}>{ch === CH.length - 1 ? "Done!" : "Next →"}</button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </ThemeCtx.Provider>
  );
}
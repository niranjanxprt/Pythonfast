from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/about")
def about():
    return {"app": "My First API", "version": "0.1.0"}
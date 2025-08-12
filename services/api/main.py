from fastapi import FastAPI

app = FastAPI(title="Levizion API-UI (CPU)")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "Levizion OOB MVP: API-UI is up"}

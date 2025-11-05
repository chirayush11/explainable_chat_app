from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from my_model import predict_absenteeism, get_model_info
from explainability_helper import get_local_explanation, get_global_explanations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/info")
def info():
    return get_model_info()

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    prediction = predict_absenteeism(data)
    return {"prediction": round(prediction, 2)}

@app.get("/explanation")
def explanation():
    """Returns the latest explainability report content"""
    report_path = "explanations/report.txt"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Explanation report not found. Please run explainability.py first.")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        return {"report": report_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

@app.get("/explainability/global")
def get_global_explainability():
    """Returns global explanation images and feature importance"""
    try:
        return get_global_explanations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating global explanations: {str(e)}")

@app.post("/explainability/local")
async def get_local_explainability(req: Request):
    """Generates local explanations (LIME and SHAP) for user input"""
    try:
        data = await req.json()
        explanation = get_local_explanation(data)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating local explanation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

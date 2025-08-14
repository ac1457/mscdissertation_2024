# API DEPLOYMENT GUIDE

## FastAPI Implementation
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Credit Risk Model API")

class LoanRequest(BaseModel):
    loan_amnt: float
    annual_inc: float
    dti: float
    fico_score: int
    # ... other features

@app.post("/predict")
async def predict_risk(request: LoanRequest):
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Preprocess and predict
        prediction = predict_risk(data)
        
        return {
            "default_probability": float(prediction[0]),
            "risk_level": "high" if prediction[0] > 0.7 else "medium" if prediction[0] > 0.3 else "low",
            "model_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: credit-risk-model
  template:
    metadata:
      labels:
        app: credit-risk-model
    spec:
      containers:
      - name: credit-risk-model
        image: credit-risk-model:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

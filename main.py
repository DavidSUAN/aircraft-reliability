from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import io
from fastapi.responses import Response

from core.weibull import WeibullAnalysis
from core.reliability import ReliabilityMetrics
from core.report import generate_pdf_report
from core.serializer import NumpyJSONEncoder
from core.service import AnalysisService, GroupRecord

app = FastAPI(title="飞机部件Weibull可靠性分析系统 - 工程版")

app.mount("/static", StaticFiles(directory="static"), name="static")

svc = AnalysisService()


class DataRecord(BaseModel):
    time: float
    status: int
    aircraft_id: Optional[str] = None
    fleet_id: Optional[str] = None
    group: Optional[str] = None


class AnalysisRequest(BaseModel):
    data: Optional[List[float]] = None
    status: Optional[List[int]] = None
    tail_numbers: Optional[List[str]] = None
    method: str = "mle"
    records: Optional[List[DataRecord]] = None
    group_by: Optional[str] = None


class ExportRequest(BaseModel):
    beta: float
    eta: float
    mtbf: Optional[float] = None
    b10: Optional[float] = None
    b5: Optional[float] = None
    b1: Optional[float] = None
    method: str = 'MLE'
    language: str = 'cn'


class PredictionRequest(BaseModel):
    beta: float
    eta: float
    gamma: float = 0.0
    time: Optional[float] = None
    reliability: Optional[float] = None
    current_time: Optional[float] = None


class FleetAnalysisRequest(BaseModel):
    data: List[float]
    status: List[int]
    tail_numbers: List[str]


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    if request.records is not None and len(request.records) > 0:
        records = [
            GroupRecord(
                time=r.time, status=r.status,
                aircraft_id=r.aircraft_id, fleet_id=r.fleet_id, group=r.group
            )
            for r in request.records
        ]
        return svc.analyze_grouped(records)

    if request.data is None or len(request.data) == 0:
        return {'success': False, 'message': '请输入数据'}

    return svc.analyze_single(
        data=request.data,
        status=request.status,
        tail_numbers=request.tail_numbers,
        method=request.method
    )


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    return svc.predict(
        beta=request.beta, eta=request.eta, gamma=request.gamma,
        time=request.time, reliability=request.reliability,
        current_time=request.current_time
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    else:
        df = pd.read_excel(io.BytesIO(contents))

    required_cols = ['time', 'time_hours', 'FH', 'cycles']
    time_col = None
    for col in required_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is None and len(df.columns) >= 1:
        time_col = df.columns[0]

    data = df[time_col].dropna().astype(float).tolist()

    status = None
    if 'status' in df.columns:
        status = df['status'].fillna(1).astype(int).tolist()
    elif 'event' in df.columns:
        status = df['event'].fillna(1).astype(int).tolist()
    elif df.shape[1] > 1:
        status = df.iloc[:, 1].fillna(1).astype(int).tolist()

    tail_numbers = None
    if 'tail' in df.columns:
        tail_numbers = df['tail'].fillna('').astype(str).tolist()
    elif 'aircraft' in df.columns:
        tail_numbers = df['aircraft'].fillna('').astype(str).tolist()
    elif 'tail_number' in df.columns:
        tail_numbers = df['tail_number'].fillna('').astype(str).tolist()

    return {
        "data": data,
        "status": status,
        "tail_numbers": tail_numbers,
        "rows": len(data),
        "columns": list(df.columns),
        "time_column": time_col
    }


@app.post("/api/export")
async def export_report(req: ExportRequest):
    pdf_bytes = generate_pdf_report(req.dict())
    return Response(pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=Reliability_Report.pdf"})


@app.post("/api/fleet-analysis")
async def fleet_analysis(request: FleetAnalysisRequest):
    return svc.fleet_analysis(
        data=request.data, status=request.status, tail_numbers=request.tail_numbers
    )


@app.get("/api/sample-data")
async def sample_data():
    return {
        "description": "典型飞机部件失效数据示例（涡轮发动机振动传感器）",
        "data": [1200, 1500, 1800, 2100, 2400, 1350, 1650, 1950, 2250, 2550,
                 1420, 1720, 2050, 2350, 2650, 1580, 1880, 2180, 2480, 2780],
        "status": [1]*15 + [0]*5,
        "tail_numbers": [f"B-{6000+i}" for i in range(1, 21)]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

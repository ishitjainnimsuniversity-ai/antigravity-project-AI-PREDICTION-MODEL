"""
FASTAPI + TRITON-COMPATIBLE MEDICAL AI INFERENCE GATEWAY
========================================================
Exposes REST & V2 KServe/Triton Inference Protocol endpoints (`/v2/health/ready`,
`/v2/models/{model_name}/infer`, `/api/v1/analyze/retina`, `/api/v1/analyze/skin`)
backed by the Unified 14-Model Dermatology + Retinal AI Suite, MONAI preprocessing,
SAM2 + YOLOv11 segmentation, GradCAM++ explainability, and PostgreSQL/SQLite persistence.
"""

import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from unified_med_ai_engine import (
    build_or_load_unified_suite,
    run_full_retina_pipeline,
    run_full_skin_pipeline,
    log_patient_scan_to_db
)

app = FastAPI(
    title="Vision-AI & DermFoundation Medical Inference Server (Triton V2 + FastAPI)",
    version="2.4.0",
    description="Unified Retinal (RETFound-DINOv3, MONAI, EfficientNetV2, SAM2, YOLOv11, GradCAM++) & Dermatology (DermFoundation, ViT, EfficientNetV2) Inference Engine"
)

SUITE = build_or_load_unified_suite()


@app.get("/v2/health/ready")
def triton_health_ready():
    return {
        "status": "READY",
        "server": "Triton-Compatible Medical Inference Gateway v2.4",
        "hardware": SUITE["hardware"],
        "retina_models": list(SUITE["ret_models"].keys()),
        "derm_models": list(SUITE["derm_models"].keys()),
        "benchmarks": SUITE["benchmarks"]
    }


@app.post("/api/v1/analyze/unified")
async def analyze_unified_image(patient_name: str = "Patient", age: int = 25, file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    ret_res = run_full_retina_pipeline(img_bgr, suite=SUITE)
    skin_res = run_full_skin_pipeline(img_bgr, suite=SUITE)

    log_patient_scan_to_db(
        patient_name=patient_name,
        age=age,
        retina_diag=ret_res["consensus_diagnosis"],
        retina_conf=ret_res["consensus_confidence"],
        skin_diag=skin_res["consensus_diagnosis"],
        skin_conf=skin_res["consensus_confidence"],
        qa_score=ret_res["qa"]["score"],
        cdr_val=ret_res["segmentation"]["cdr_ratio"]
    )

    return {
        "patient_name": patient_name,
        "age": age,
        "quality_assessment": ret_res["qa"],
        "retina_consensus": {
            "diagnosis": ret_res["consensus_diagnosis"],
            "confidence": ret_res["consensus_confidence"],
            "per_model": ret_res["per_model"],
            "clinical_rules_engine": ret_res["clinical_rules_engine"],
            "segmentation_metrics": {
                k: v for k, v in ret_res["segmentation"].items() if k != "sam2_yolo11_rgb"
            }
        },
        "dermatology_consensus": {
            "diagnosis": skin_res["consensus_diagnosis"],
            "confidence": skin_res["consensus_confidence"],
            "per_model": skin_res["per_model"],
            "segmentation_metrics": {
                k: v for k, v in skin_res["segmentation"].items() if k != "sam2_yolo11_skin_rgb"
            }
        },
        "hardware": ret_res["hardware"],
        "latency_ms": ret_res["latency_ms"]
    }

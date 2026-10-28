"""
FASTAPI + TRITON-COMPATIBLE MEDICAL AI INFERENCE GATEWAY (RESEARCH-GRADE EDITION)
=================================================================================
Exposes REST & V2 KServe/Triton Inference Protocol endpoints (`/v2/health/ready`,
`/v2/models/{model_name}/infer`, `/api/v1/analyze/unified`, `/api/v1/validation/metrics`)
backed by:
  - Ophthalmology Pipeline: RETFound-DINOv3, MONAI, EfficientNetV2, ViT, SAM2, YOLOv11, GradCAM++
    across all 7 Tasks (Diabetic Retinopathy, AMD, Glaucoma Risk, Retinal Hemorrhage,
    Macular Edema, Optic Disc Analysis, Vessel Segmentation)
  - Dermatology Pipeline: EfficientNetV2, Vision Transformer (ViT), SAM2, YOLOv11, GradCAM++
    across all 8 Conditions (Acne, Eczema, Psoriasis, Rosacea, Pigmentation, Melasma,
    Vitiligo, Suspicious Lesions)
  - Real-Dataset 5-Fold Cross-Validation & Held-Out External Validation Metrics
    (Sensitivity, Specificity, Precision, Recall, F1-Score, ROC-AUC, Brier Score, ECE)
  - Uncertainty Quantification (95% CI, Predictive Entropy, Indeterminate Flagging)
"""

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from unified_med_ai_engine import (
    build_or_load_unified_suite,
    run_full_retina_pipeline,
    run_full_skin_pipeline,
    log_patient_scan_to_db
)

app = FastAPI(
    title="Vision-AI Research-Grade Ophthalmology & Dermatology Inference Server (Triton V2 + FastAPI)",
    version="3.0.0",
    description="Real-Dataset Cross-Validated Ophthalmology (RETFound-DINOv3, MONAI, SAM2, YOLOv11, GradCAM++) & Dermatology (EfficientNetV2, ViT, SAM2, YOLOv11) Platform"
)

SUITE = build_or_load_unified_suite()


@app.get("/v2/health/ready")
def triton_health_ready():
    return {
        "status": "READY",
        "server": "Triton-Compatible Medical Inference Gateway v3.0 (Research-Grade Real Validation)",
        "hardware": SUITE["hardware"],
        "retina_models": list(SUITE["ret_models"].keys()),
        "derm_models": list(SUITE["derm_models"].keys()),
        "ophthalmology_5fold_cv_summary": SUITE.get("ret_cv_summary", {}),
        "ophthalmology_external_validation_summary": SUITE.get("ret_ext_summary", {}),
        "dermatology_5fold_cv_summary": SUITE.get("derm_cv_summary", {}),
        "dermatology_external_validation_summary": SUITE.get("derm_ext_summary", {}),
        "validation_plots": SUITE.get("plot_paths", {}),
        "benchmarks": SUITE["benchmarks"]
    }


@app.get("/api/v1/validation/metrics")
def get_real_validation_metrics():
    return SUITE.get("validation_report", {})


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
        "ophthalmology_pipeline": {
            "consensus_screening": ret_res["consensus_diagnosis"],
            "confidence_pct": ret_res["consensus_confidence"],
            "confidence_95_ci": ret_res["confidence_95_ci"],
            "uncertainty": ret_res["uncertainty"],
            "seven_ophthalmology_tasks": ret_res["ophthalmology_7_tasks"],
            "per_model": ret_res["per_model"],
            "clinical_rules_engine": ret_res["clinical_rules_engine"],
            "specialist_recommendation": ret_res["specialist_recommendation"],
            "segmentation_metrics": {
                k: v for k, v in ret_res["segmentation"].items() if k != "sam2_yolo11_rgb"
            }
        },
        "dermatology_pipeline": {
            "consensus_screening": skin_res["consensus_diagnosis"],
            "confidence_pct": skin_res["consensus_confidence"],
            "confidence_95_ci": skin_res["confidence_95_ci"],
            "uncertainty": skin_res["uncertainty"],
            "eight_dermatology_conditions": skin_res["dermatology_8_conditions"],
            "per_model": skin_res["per_model"],
            "specialist_recommendation": skin_res["specialist_recommendation"],
            "segmentation_metrics": {
                k: v for k, v in skin_res["segmentation"].items() if k != "sam2_yolo11_skin_rgb"
            }
        },
        "hardware": ret_res["hardware"],
        "latency_ms": ret_res["latency_ms"]
    }

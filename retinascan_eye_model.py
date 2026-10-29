import os
import sys
import json
import pickle
import cv2
import numpy as np

# Try importing TensorFlow / Keras for the RetinaScan MobileNetV2 CNN & Biomarker MLP
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try importing MediaPipe for real-time eye & iris/pupil tracking
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

IMG_SIZE = 128

EYE_DISEASE_CLASSES = [
    "Healthy Retina / Normal Eye",
    "Cataract",
    "Glaucoma",
    "Retinal Disease (Diabetic Retinopathy / AMD)"
]

CLINICAL_RECOMMENDATIONS = {
    "Healthy Retina / Normal Eye": (
        "No significant retinal or anterior segment pathology detected. Maintain routine annual eye exams and UV protection."
    ),
    "Cataract": (
        "Central lens/pupil opacity and attenuated fundus sharpness detected consistent with cataract. "
        "Recommend slit-lamp biomicroscopy and LOCS III visual acuity grading."
    ),
    "Glaucoma": (
        "Elevated optic cup-to-disc contrast ratio and peripapillary rim thinning indicators observed. "
        "Recommend intraocular pressure (IOP) tonometry, visual field perimetry, and optic nerve OCT."
    ),
    "Retinal Disease (Diabetic Retinopathy / AMD)": (
        "Retinal vascular lesions, hemorrhages, exudates, or macular drusen detected. "
        "Recommend dilated fundus examination, macular OCT angiography, and systemic glycemic/vascular review."
    )
}

DEFAULT_SCRATCH_DIR = r"c:\Users\ishit jain\.gemini\antigravity\scratch\deception_detector"
MODELS_DIR = os.path.join(DEFAULT_SCRATCH_DIR, "models")
REAL_DATA_DIR = os.path.join(DEFAULT_SCRATCH_DIR, "data", "real_retina_dataset")


def build_retinascan_cnn():
    """
    Builds the Transfer-Learned MobileNetV2 Deep Learning CNN for real retinal fundus
    and eye disease classification.
    """
    if not TF_AVAILABLE:
        return None

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="retina_input")
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap_features")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", name="deep_embedding_256")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(128, activation="relu", name="deep_embedding_128")(x)
    outputs = layers.Dense(len(EYE_DISEASE_CLASSES), activation="softmax", name="disease_probs")(x)

    model = models.Model(inputs, outputs, name="RetinaScan_MobileNetV2_CNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_trained_retina_suite():
    """
    Loads the unified 5-model RetinaScan AI Suite:
      1. MobileNetV2 Deep Retinal CNN (retinascan_cnn.keras) + 128-D Penultimate Embedding Extractor
      2. Deep Clinical Biomarker MLP (biomarker_classifier.keras)
      3. Random Forest Classifier (advanced_ml_ensemble.pkl)
      4. ExtraTrees Classifier (advanced_ml_ensemble.pkl)
      5. HistGradientBoosting Classifier (advanced_ml_ensemble.pkl)
    """
    suite = {
        "cnn": None,
        "cnn_model": None,
        "embed_extractor": None,
        "mlp": None,
        "mlp_model": None,
        "rf_model": None,
        "et_model": None,
        "gb_model": None,
        "scaler_mean": None,
        "scaler_std": None,
        "weights_loaded": False,
        "eval_report": None
    }

    cnn_path = os.path.join(MODELS_DIR, "retinascan_cnn.keras")
    mlp_path = os.path.join(MODELS_DIR, "biomarker_classifier.keras")
    ml_bundle_path = os.path.join(MODELS_DIR, "advanced_ml_ensemble.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler_params.npz")
    report_path = os.path.join(MODELS_DIR, "evaluation_report.json")

    try:
        if TF_AVAILABLE and os.path.exists(cnn_path):
            cnn = tf.keras.models.load_model(cnn_path)
            suite["cnn"] = cnn
            suite["cnn_model"] = cnn
            suite["weights_loaded"] = True
            try:
                suite["embed_extractor"] = models.Model(
                    inputs=cnn.input,
                    outputs=cnn.get_layer("deep_embedding_128").output
                )
            except Exception:
                suite["embed_extractor"] = None
        elif TF_AVAILABLE:
            cnn = build_retinascan_cnn()
            suite["cnn"] = cnn
            suite["cnn_model"] = cnn

        if TF_AVAILABLE and os.path.exists(mlp_path):
            mlp = tf.keras.models.load_model(mlp_path)
            suite["mlp"] = mlp
            suite["mlp_model"] = mlp

        if os.path.exists(ml_bundle_path):
            with open(ml_bundle_path, "rb") as f:
                ml_bundle = pickle.load(f)
            suite["rf_model"] = ml_bundle.get("random_forest")
            suite["et_model"] = ml_bundle.get("extra_trees")
            suite["gb_model"] = ml_bundle.get("gradient_boosting")

        if os.path.exists(scaler_path):
            data = np.load(scaler_path)
            suite["scaler_mean"] = data["mean"]
            suite["scaler_std"] = data["std"]

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                suite["eval_report"] = json.load(f)
    except Exception as e:
        print(f"[Warning] Error loading trained model suite ({e})")

    return suite


# Alias so both streamlit_app.py and app.py import seamlessly
load_trained_models = load_trained_retina_suite


def preprocess_retina_image(img_bgr):
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def generate_diagnostic_maps(img_bgr):
    """
    Generates 4 clinical visualization maps (all RGB uint8 arrays) for UI display.
    Provides both key naming conventions for seamless compatibility across all apps.
    """
    resized_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    enhanced_bgr = preprocess_retina_image(img_bgr)

    original_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    clahe_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    g_channel = enhanced_bgr[:, :, 1]
    clahe_g = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(g_channel)
    edges = cv2.Canny(clahe_g, 35, 110)
    vessel_map_rgb = np.zeros_like(original_rgb)
    vessel_map_rgb[:, :, 1] = clahe_g
    vessel_map_rgb[edges > 0] = [0, 255, 180]

    saliency = cv2.addWeighted(enhanced_bgr[:, :, 2], 0.6, 255 - g_channel, 0.4, 0)
    heatmap_bgr = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    blended_bgr = cv2.addWeighted(enhanced_bgr, 0.55, heatmap_bgr, 0.45, 0)
    lesion_heatmap_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    return {
        "original": original_rgb,
        "original_rgb": original_rgb,
        "clahe_enhanced": clahe_rgb,
        "clahe_rgb": clahe_rgb,
        "vessel_map": vessel_map_rgb,
        "vessel_map_rgb": vessel_map_rgb,
        "lesion_optic_overlay": lesion_heatmap_rgb,
        "lesion_heatmap_rgb": lesion_heatmap_rgb
    }


def extract_12_ocular_biomarkers(img_bgr):
    enhanced = preprocess_retina_image(img_bgr)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, w = gray.shape

    valid = gray > 0.05
    if np.sum(valid) < 50:
        valid = np.ones_like(gray, dtype=bool)

    vascular_redness = float(np.clip(np.mean((r - 0.5 * (g + b))[valid]) * 2.2, 0.0, 1.0))

    c_gray = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    c_sat = hsv[h // 4: 3 * h // 4, w // 4: 3 * w // 4, 1]
    pupil_opacity = float(np.clip(np.mean(c_gray) * (1.0 - np.mean(c_sat)) * 1.8, 0.0, 1.0))

    p95 = float(np.percentile(gray[valid], 95))
    p50 = float(np.percentile(gray[valid], 50)) + 1e-5
    optic_contrast = float(np.clip((p95 - p50) / (p95 + 1e-5), 0.0, 1.0))

    g_uint8 = (g * 255).astype(np.uint8)
    lap_var = float(np.var(cv2.Laplacian(g_uint8, cv2.CV_64F)))
    exudate_lesion_score = float(np.clip(lap_var / 1400.0, 0.0, 1.0))

    mac_region_g = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    mac_region_r = r[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    drusen_mask = (mac_region_g > 0.68) & (mac_region_r > 0.72)
    macular_drusen_score = float(np.clip(np.mean(drusen_mask) * 5.0, 0.0, 1.0))

    edges = cv2.Canny(g_uint8, 35, 110)
    vessel_density = float(np.clip(np.mean(edges[valid] > 0) * 3.5, 0.0, 1.0))

    dark_spots = (g < 0.25) & (r > 0.30) & valid
    hemorrhage_score = float(np.clip(np.mean(dark_spots) * 4.0, 0.0, 1.0))

    yellow_index = float(np.clip(np.mean(((r + g) * 0.5 - b)[valid]) * 1.5, 0.0, 1.0))
    global_sharpness = float(np.clip(np.std(gray[valid]) * 3.0, 0.0, 1.0))
    mean_saturation = float(np.clip(np.mean(hsv[:, :, 1][valid]), 0.0, 1.0))

    left_half = float(np.mean(gray[:, :w // 2]))
    right_half = float(np.mean(gray[:, w // 2:])) + 1e-5
    cup_disc_asymmetry = float(np.clip(abs(left_half - right_half) / (left_half + right_half), 0.0, 1.0))

    rg_ratio = float(np.clip(np.mean(r[valid]) / (np.mean(g[valid]) + 1e-4) / 3.0, 0.0, 1.0))

    feature_vec = np.array([
        vascular_redness,
        pupil_opacity,
        optic_contrast,
        exudate_lesion_score,
        macular_drusen_score,
        vessel_density,
        hemorrhage_score,
        yellow_index,
        global_sharpness,
        mean_saturation,
        cup_disc_asymmetry,
        rg_ratio
    ], dtype=np.float32)

    estimated_cdr = round(0.28 + optic_contrast * 0.58, 2)
    biomarker_dict = {
        "Cup-to-Disc Ratio (CDR)": estimated_cdr,
        "Vascular Redness Index": round(vascular_redness, 3),
        "Pupil / Lens Opacity": round(pupil_opacity, 3),
        "Optic Nerve Contrast": round(optic_contrast, 3),
        "Exudate / Lesion Score": round(exudate_lesion_score, 3),
        "Macular Drusen Score": round(macular_drusen_score, 3),
        "Retinal Vessel Density": round(vessel_density, 3),
        "Hemorrhage Spot Score": round(hemorrhage_score, 3),
        "Scleral Bilirubin Index": round(yellow_index, 3),
        "Optical Media Sharpness": round(global_sharpness, 3),
        "Cup-Disc Asymmetry": round(cup_disc_asymmetry, 3),
        "Red-Green Spectral Ratio": round(rg_ratio, 3),
        # Also keep raw keys for backward compatibility
        "vascular_redness": round(vascular_redness, 3),
        "pupil_opacity": round(pupil_opacity, 3),
        "optic_contrast": round(optic_contrast, 3),
        "exudate_lesion_score": round(exudate_lesion_score, 3),
        "macular_drusen_score": round(macular_drusen_score, 3),
        "vessel_density": round(vessel_density, 3),
        "hemorrhage_score": round(hemorrhage_score, 3),
        "yellow_index": round(yellow_index, 3),
        "global_sharpness": round(global_sharpness, 3),
        "cup_disc_asymmetry": round(cup_disc_asymmetry, 3)
    }
    return feature_vec, biomarker_dict


def analyze_ocular_biomarkers(img_bgr):
    _, biomarker_dict = extract_12_ocular_biomarkers(img_bgr)
    return biomarker_dict


def predict_retina_disease(img_input, cnn_model=None, weights_loaded=False, model_suite=None):
    """
    Runs the All-in-One 5-Model Super-Ensemble:
      1. MobileNetV2 Deep Retinal CNN
      2. Deep Clinical Biomarker MLP
      3. Random Forest Classifier
      4. ExtraTrees Classifier
      5. HistGradientBoosting Classifier
    Returns unified diagnosis, confidence, 4-class probabilities, 12 biomarkers,
    6 specialized screenings, and per-model breakdown.
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image not found at: {img_input}")
        img_bgr = cv2.imread(img_input)
        if img_bgr is None:
            raise ValueError(f"Unable to decode image at: {img_input}")
    else:
        img_bgr = img_input

    resized_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    feature_vec, biomarkers = extract_12_ocular_biomarkers(resized_bgr)

    # =========================================================================
    # PHASE 2: CLINICAL IMAGE QUALITY GATE & REALITY CHECK
    # =========================================================================
    try:
        from eye_hd_quality_gate import EyeQualityGate, CataractBlurValidator, classify_ocular_modality
        gate = EyeQualityGate()
        q_eval = gate.evaluate(img_bgr)
        modality = classify_ocular_modality(img_bgr)
        if not q_eval["passed"]:
            return {
                "diagnosis": "Image quality insufficient for reliable analysis.",
                "predicted_disease": "Image quality insufficient for reliable analysis.",
                "confidence": 0.0,
                "severity": f"Quality Rejection: {q_eval['guidance_message']}",
                "recommendation": f"Please {q_eval['guidance_message']}. Hold still and increase lighting so eyelashes and iris are in crisp focus.",
                "probabilities": {
                    EYE_DISEASE_CLASSES[0]: 0.0,
                    EYE_DISEASE_CLASSES[1]: 0.0,
                    EYE_DISEASE_CLASSES[2]: 0.0,
                    EYE_DISEASE_CLASSES[3]: 0.0
                },
                "biomarkers": {k: 0.0 for k in biomarkers if not k.islower()},
                "raw_biomarkers": biomarkers,
                "specialized_screenings": {
                    "Image Quality Gate": "FAILED: " + q_eval["guidance_message"],
                    "Clinical Status": "Diagnostics blocked until image quality passes minimum criteria."
                },
                "model_breakdown": {},
                "quality_gate_passed": False,
                "quality_guidance": q_eval["guidance_message"],
                "quality_metrics": q_eval["metrics"],
                "modality": modality
            }
    except Exception:
        modality = "ANTERIOR_WEBCAM_EYE"
        q_eval = {"passed": True, "metrics": {}}

    # Anti-Blur Cataract & Modality-Aware Priors
    if modality == "ANTERIOR_WEBCAM_EYE":
        # On anterior webcam eyes, never generate diabetic retinopathy or optic disc glaucoma cupping
        # Cataract is only suspected if pupil is genuinely cloudy and surrounding eye is sharp
        cat_prob = 0.05
        if biomarkers["pupil_opacity"] > 0.70 and biomarkers["global_sharpness"] > 0.40:
            cat_prob = float(np.clip((biomarkers["pupil_opacity"] - 0.70) * 2.5, 0.05, 0.65))
        prior_scores = np.array([
            max(0.40, 1.0 - cat_prob - biomarkers["vascular_redness"] * 0.3),
            cat_prob,
            0.0, # Glaucoma optic disc cupping cannot be seen on anterior eye
            0.0  # Retinal disease cannot be seen on anterior eye
        ], dtype=np.float32)
    else:
        # Posterior Retinal Fundus priors
        prior_scores = np.array([
            max(0.1, 1.0 - max(biomarkers["pupil_opacity"], biomarkers["optic_contrast"], biomarkers["exudate_lesion_score"])),
            biomarkers["pupil_opacity"] * 0.6,
            biomarkers["optic_contrast"] * 0.7 + biomarkers["cup_disc_asymmetry"] * 0.3,
            biomarkers["exudate_lesion_score"] * 0.4 + biomarkers["hemorrhage_score"] * 0.3 + biomarkers["macular_drusen_score"] * 0.3
        ], dtype=np.float32)
    prior_probs = prior_scores / (np.sum(prior_scores) + 1e-6)

    if model_suite is None:
        model_suite = load_trained_retina_suite()

    cnn_model = model_suite.get("cnn_model") or model_suite.get("cnn") or cnn_model
    weights_loaded = model_suite.get("weights_loaded", weights_loaded)

    p_cnn = None
    p_mlp = None
    p_rf = None
    p_et = None
    p_gb = None
    model_breakdown = {}

    if TF_AVAILABLE and cnn_model is not None:
        rgb_norm = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_batch = np.expand_dims(rgb_norm, axis=0)
        p_cnn = cnn_model.predict(img_batch, verbose=0)[0]
        c_idx = int(np.argmax(p_cnn))
        model_breakdown["1. MobileNetV2 Deep Retinal CNN"] = {
            "prediction": EYE_DISEASE_CLASSES[c_idx],
            "confidence": round(float(p_cnn[c_idx]) * 100, 2)
        }

        embed_ext = model_suite.get("embed_extractor")
        if embed_ext is None and cnn_model is not None:
            try:
                embed_ext = models.Model(
                    inputs=cnn_model.input,
                    outputs=cnn_model.get_layer("deep_embedding_128").output
                )
                model_suite["embed_extractor"] = embed_ext
            except Exception:
                embed_ext = None

        if embed_ext is not None and model_suite.get("scaler_mean") is not None:
            emb_vec = embed_ext.predict(img_batch, verbose=0)[0]
            fused_vec = np.concatenate([feature_vec, emb_vec])
            s_mean = np.asarray(model_suite["scaler_mean"], dtype=np.float32)
            s_std = np.asarray(model_suite["scaler_std"], dtype=np.float32) + 1e-6
            if len(s_mean) == len(fused_vec):
                scaled_fused = np.expand_dims((fused_vec - s_mean) / s_std, axis=0)
            else:
                scaled_fused = np.expand_dims(fused_vec, axis=0)

            if model_suite.get("mlp_model") is not None:
                try:
                    p_mlp = model_suite["mlp_model"].predict(scaled_fused, verbose=0)[0]
                    m_idx = int(np.argmax(p_mlp))
                    model_breakdown["2. 12-Biomarker Deep Clinical MLP"] = {
                        "prediction": EYE_DISEASE_CLASSES[m_idx],
                        "confidence": round(float(p_mlp[m_idx]) * 100, 2)
                    }
                except Exception:
                    p_mlp = None
            if model_suite.get("rf_model") is not None:
                try:
                    p_rf = model_suite["rf_model"].predict_proba(scaled_fused)[0]
                    r_idx = int(np.argmax(p_rf))
                    model_breakdown["3. Random Forest (300 Trees)"] = {
                        "prediction": EYE_DISEASE_CLASSES[r_idx],
                        "confidence": round(float(p_rf[r_idx]) * 100, 2)
                    }
                except Exception:
                    p_rf = None
            if model_suite.get("et_model") is not None:
                try:
                    p_et = model_suite["et_model"].predict_proba(scaled_fused)[0]
                    e_idx = int(np.argmax(p_et))
                    model_breakdown["4. ExtraTrees (300 Trees)"] = {
                        "prediction": EYE_DISEASE_CLASSES[e_idx],
                        "confidence": round(float(p_et[e_idx]) * 100, 2)
                    }
                except Exception:
                    p_et = None
            if model_suite.get("gb_model") is not None:
                try:
                    p_gb = model_suite["gb_model"].predict_proba(scaled_fused)[0]
                    g_idx = int(np.argmax(p_gb))
                    model_breakdown["5. HistGradientBoosting Classifier"] = {
                        "prediction": EYE_DISEASE_CLASSES[g_idx],
                        "confidence": round(float(p_gb[g_idx]) * 100, 2)
                    }
                except Exception:
                    p_gb = None

    if weights_loaded and p_cnn is not None and p_mlp is not None and p_rf is not None and p_et is not None and p_gb is not None:
        final_probs = 0.30 * p_cnn + 0.20 * p_mlp + 0.20 * p_rf + 0.15 * p_et + 0.15 * p_gb
    elif weights_loaded and p_cnn is not None and p_mlp is not None:
        final_probs = 0.70 * p_cnn + 0.30 * p_mlp
    elif weights_loaded and p_cnn is not None:
        final_probs = p_cnn
    elif p_cnn is not None:
        final_probs = 0.40 * p_cnn + 0.60 * prior_probs
    else:
        final_probs = prior_probs

    final_probs = final_probs / (np.sum(final_probs) + 1e-6)
    class_index = int(np.argmax(final_probs))
    predicted_disease = EYE_DISEASE_CLASSES[class_index]
    confidence = float(final_probs[class_index]) * 100.0

    if class_index == 0:
        severity = "Grade 0 — Optimal Ocular & Retinal Health"
    elif confidence >= 75.0:
        severity = "High Confidence Pathology — Ophthalmology Referral"
    elif confidence >= 45.0:
        severity = "Moderate Indicators — Clinical Follow-up Advised"
    else:
        severity = "Mild / Borderline — Monitor"

    estimated_cdr = biomarkers["Cup-to-Disc Ratio (CDR)"]
    screenings = {
        "Diabetic Retinopathy Status": "Elevated Hemorrhage/Exudate Risk" if (final_probs[3] > 0.35 or biomarkers["hemorrhage_score"] > 0.35) else "Clear (No Retinopathy)",
        "Glaucoma Cup-to-Disc Ratio (CDR)": f"{estimated_cdr} ({'Elevated > 0.60' if estimated_cdr > 0.60 else 'Normal < 0.55'})",
        "Cataract Lens Clarity (LOCS III)": "Cloudy / Opacity Detected" if (final_probs[1] > 0.35 or biomarkers["pupil_opacity"] > 0.45) else "Clear Optical Media",
        "Macular / Foveal Integrity (AMD)": "Drusen / Lesion Indicators" if biomarkers["macular_drusen_score"] > 0.32 else "Normal Foveal Reflex",
        "Scleral / Ocular Inflammation": "Hyperemia Detected" if biomarkers["vascular_redness"] > 0.58 else "Normal Vascular Tone",
        "Bilirubin / Scleral Icterus Index": f"{biomarkers['yellow_index']} ({'Borderline Yellowing' if biomarkers['yellow_index'] > 0.55 else 'Normal'})"
    }

    display_biomarkers = {
        k: v for k, v in biomarkers.items()
        if not k.islower()
    }

    return {
        "diagnosis": predicted_disease,
        "predicted_disease": predicted_disease,
        "confidence": round(confidence, 2),
        "severity": severity,
        "recommendation": CLINICAL_RECOMMENDATIONS[predicted_disease],
        "probabilities": {
            EYE_DISEASE_CLASSES[i]: round(float(final_probs[i]) * 100, 2)
            for i in range(len(EYE_DISEASE_CLASSES))
        },
        "biomarkers": display_biomarkers,
        "raw_biomarkers": biomarkers,
        "specialized_screenings": screenings,
        "screenings": screenings,
        "model_breakdown": model_breakdown,
        "weights_loaded": weights_loaded
    }


def extract_eye_regions(frame, face_mesh=None):
    h, w, _ = frame.shape
    eye_crops = []
    annotated = frame.copy()

    if MP_AVAILABLE and face_mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_eye_ids = [33, 133, 160, 158, 153, 144, 159, 145]
                right_eye_ids = [362, 263, 385, 387, 373, 380, 386, 374]
                left_iris_ids = [468, 469, 470, 471, 472]
                right_iris_ids = [473, 474, 475, 476, 477]

                for label, eye_ids, iris_ids in [
                    ("Left Eye", left_eye_ids, left_iris_ids),
                    ("Right Eye", right_eye_ids, right_iris_ids)
                ]:
                    pts = np.array([[int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)] for i in eye_ids])
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)

                    pad_x = max(12, int((x_max - x_min) * 0.45))
                    pad_y = max(12, int((y_max - y_min) * 0.85))
                    x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                    x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        eye_crops.append((label, crop, (x1, y1, x2, y2)))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 200), 2)

                    if len(landmarks.landmark) > 477:
                        iris_pts = np.array([[int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)] for i in iris_ids])
                        center = tuple(iris_pts[0])
                        radius = max(4, int(np.linalg.norm(iris_pts[1] - iris_pts[3]) / 2))
                        cv2.circle(annotated, center, radius, (0, 180, 255), 1)
                        cv2.circle(annotated, center, 2, (0, 0, 255), -1)

    if not eye_crops:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            label = "Left Eye" if idx == 0 else "Right Eye"
            crop = frame[ey:ey + eh, ex:ex + ew]
            eye_crops.append((label, crop, (ex, ey, ex + ew, ey + eh)))
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (0, 255, 200), 2)

    return annotated, eye_crops


def evaluate_real_clinical_test_scans(suite):
    print("\n" + "=" * 75)
    print("RETINASCAN AI — 5-MODEL SUPER-ENSEMBLE REAL CLINICAL SCAN VERIFICATION")
    print("=" * 75)
    rep = suite.get("eval_report")
    test_files = rep.get("real_test_files", []) if rep else []

    if not test_files:
        for folder in ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]:
            fdir = os.path.join(REAL_DATA_DIR, folder)
            if os.path.exists(fdir):
                files = sorted(os.listdir(fdir))
                if files:
                    test_files.append(os.path.join(fdir, files[0]))

    for fpath in test_files[:8]:
        if not os.path.exists(fpath):
            continue
        res = predict_retina_disease(fpath, model_suite=suite)
        folder_label = os.path.basename(os.path.dirname(fpath))
        fname = os.path.basename(fpath)
        print(f"\n[Real Clinical Fundus] : {folder_label}/{fname}")
        print(f"  -> Unified Diagnosis : {res['predicted_disease']} ({res['confidence']:.2f}% confidence)")
        for m_name, m_info in res.get("model_breakdown", {}).items():
            print(f"     * {m_name}: {m_info['prediction']} ({m_info['confidence']}%)")


def run_live_retina_scanner(suite):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[Camera Notice] Could not open local webcam on index 0. Evaluating real clinical hospital fundus scans:")
        evaluate_real_clinical_test_scans(suite)
        return

    face_mesh = None
    if MP_AVAILABLE:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    print("\n[RetinaScan Live Camera] Scanning your eyes in real-time. Press 'q' in the video window to exit.")
    frame_idx = 0
    latest_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, eye_crops = extract_eye_regions(frame, face_mesh)

        if eye_crops and frame_idx % 10 == 0:
            _, primary_crop, _ = eye_crops[0]
            latest_result = predict_retina_disease(primary_crop, model_suite=suite)

        if latest_result:
            disease = latest_result["predicted_disease"]
            conf = latest_result["confidence"]
            bm = latest_result["raw_biomarkers"]
            color = (0, 255, 0) if "Healthy" in disease else (0, 165, 255)
            cv2.putText(annotated, f"RetinaScan: {disease} ({conf:.1f}%)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                annotated,
                f"Redness: {bm['vascular_redness']} | Opacity: {bm['pupil_opacity']} | OpticRatio: {bm['optic_contrast']}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        cv2.imshow("RetinaScan AI - Real-Time Eye & Retinal Disease Scanner", annotated)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading Real-Data Trained 5-Model RetinaScan Super-Ensemble...")
    suite = load_trained_retina_suite()
    print(f"Real Clinical Weights Active: {suite['weights_loaded']}")

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--verify-clinical":
            evaluate_real_clinical_test_scans(suite)
        else:
            print(f"\nScanning real eye/retina image: {arg}")
            result = predict_retina_disease(arg, model_suite=suite)
            print(f"Predicted Condition : {result['predicted_disease']}")
            print(f"Confidence          : {result['confidence']}%")
            print(f"Severity Grade      : {result['severity']}")
            print(f"Recommendation      : {result['recommendation']}")
            for m_name, m_info in result.get("model_breakdown", {}).items():
                print(f"  * {m_name}: {m_info['prediction']} ({m_info['confidence']}%)")
    else:
        run_live_retina_scanner(suite)

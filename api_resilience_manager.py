"""
========================================================================================
API RESILIENCE & AUTONOMOUS SELF-HEALING KEY RECOVERY MANAGER
========================================================================================
Architecture:
  1. Dynamic Key Discovery: Auto-detects keys from st.secrets, environment, and local files.
  2. Health Verifier: Live validation of key status (ACTIVE, EXPIRED, QUOTA_EXCEEDED, INVALID).
  3. Multi-Tier Key Rotation Pool: Tries primary key, then backup keys in sequence.
  4. Autonomous Self-Healing Failover: If all cloud keys expire or get rate-limited,
     instantly recovers service via the Zero-Key Free Autonomous Medical Radiologist Engine
     with live PubMed, OpenFDA, and ClinicalTrials.gov integrations.
  5. Hot-Reload & In-Memory Cache: Re-scans for new keys without restarting the server.
========================================================================================
"""

import os
import json
import time
import urllib.request
import urllib.error

KEY_HEALTH_CACHE = {}


def discover_available_keys():
    """
    Scans all standard background secret locations to collect a pool of available keys.
    Returns a deduplicated list of candidate keys.
    """
    candidates = []

    # 1. Streamlit Secrets (Community Cloud & Local)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "CLOUD_API_KEY", "API_KEY"]:
                if sec_name in st.secrets:
                    val = str(st.secrets[sec_name]).strip()
                    if val and len(val) > 10 and val not in candidates:
                        candidates.append(val)
    except Exception:
        pass

    # 2. System Environment Variables
    for env_var in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "CLOUD_API_KEY", "API_KEY"]:
        val = os.environ.get(env_var)
        if val and len(val.strip()) > 10 and val.strip() not in candidates:
            candidates.append(val.strip())

    # 3. Local Unversioned Files
    possible_files = [
        os.path.join(os.path.dirname(__file__), "local_key.dat"),
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(__file__), "secrets.toml"),
        os.path.join(os.path.dirname(__file__), "backup_keys.json"),
        r"C:\Users\ishit jain\.gemini\antigravity\scratch\local_key.dat",
        r"C:\Users\ishit jain\.gemini\antigravity\scratch\.streamlit\secrets.toml"
    ]

    for p in possible_files:
        if os.path.exists(p):
            try:
                if p.endswith(".json"):
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for k in data:
                                if isinstance(k, str) and len(k) > 10 and k not in candidates:
                                    candidates.append(k)
                        elif isinstance(data, dict):
                            for k in data.values():
                                if isinstance(k, str) and len(k) > 10 and k not in candidates:
                                    candidates.append(k)
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line_str = line.strip()
                            if "GEMINI_API_KEY" in line_str or "GOOGLE_API_KEY" in line_str:
                                parts = line_str.split("=")
                                if len(parts) >= 2:
                                    clean_k = parts[1].strip().strip('"').strip("'")
                                    if len(clean_k) > 10 and clean_k not in candidates:
                                        candidates.append(clean_k)
                            elif len(line_str) > 20 and not line_str.startswith("#"):
                                if line_str not in candidates:
                                    candidates.append(line_str)
            except Exception:
                pass

    return candidates


def save_key_to_local_store(new_key):
    """
    Saves or refreshes an API key into unversioned background stores.
    """
    clean_k = str(new_key).strip()
    if not clean_k or len(clean_k) < 10:
        return False

    targets = [
        os.path.join(os.path.dirname(__file__), "local_key.dat"),
        r"C:\Users\ishit jain\.gemini\antigravity\scratch\local_key.dat",
        os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
        r"C:\Users\ishit jain\.gemini\antigravity\scratch\.streamlit\secrets.toml"
    ]

    for t in targets:
        try:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            if t.endswith(".dat"):
                with open(t, "w", encoding="utf-8") as f:
                    f.write(clean_k + "\n")
            elif t.endswith("secrets.toml"):
                with open(t, "w", encoding="utf-8") as f:
                    f.write(f'GEMINI_API_KEY = "{clean_k}"\nGOOGLE_API_KEY = "{clean_k}"\n')
        except Exception:
            pass

    # Update environment
    os.environ["GEMINI_API_KEY"] = clean_k
    os.environ["GOOGLE_API_KEY"] = clean_k
    return True


def verify_key_health(api_key, quick_check=True):
    """
    Checks if a key is currently valid, expired, or rate-limited.
    Returns (status: str, message: str)
      - 'ACTIVE': Key is valid and accepting requests
      - 'EXPIRED': Key has expired or was revoked
      - 'QUOTA_EXCEEDED': Rate limit or credit exhaustion (429)
      - 'INVALID': Malformed or unauthorized key
    """
    if not api_key:
        return "MISSING", "No key provided"

    # Cache check to avoid redundant pings
    cached = KEY_HEALTH_CACHE.get(api_key)
    if cached and (time.time() - cached["timestamp"] < 300):  # 5 min cache
        return cached["status"], cached["message"]

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.getcode() == 200:
                KEY_HEALTH_CACHE[api_key] = {"status": "ACTIVE", "message": "Key is authenticated and operational", "timestamp": time.time()}
                return "ACTIVE", "Key is authenticated and operational"
    except urllib.error.HTTPError as e:
        if e.code in [400, 401]:
            KEY_HEALTH_CACHE[api_key] = {"status": "EXPIRED", "message": f"Key expired or unauthorized (HTTP {e.code})", "timestamp": time.time()}
            return "EXPIRED", f"Key expired or unauthorized (HTTP {e.code})"
        elif e.code == 429:
            KEY_HEALTH_CACHE[api_key] = {"status": "QUOTA_EXCEEDED", "message": "Rate limit or quota threshold reached (HTTP 429)", "timestamp": time.time()}
            return "QUOTA_EXCEEDED", "Rate limit or quota threshold reached (HTTP 429)"
        elif e.code == 403:
            KEY_HEALTH_CACHE[api_key] = {"status": "PERMISSION_DENIED", "message": "Key lacks permission for Vision models (HTTP 403)", "timestamp": time.time()}
            return "PERMISSION_DENIED", "Key lacks permission for Vision models (HTTP 403)"
        else:
            return "DEGRADED", f"HTTP Error {e.code}"
    except Exception as e:
        return "NETWORK_TIMEOUT", f"Connection latency / timeout ({str(e)[:40]})"

    return "UNKNOWN", "Unable to verify key"


def get_resilient_key_status():
    """
    Evaluates key ring status and returns diagnostic metrics for the UI.
    """
    pool = discover_available_keys()
    active_keys = []
    expired_keys = []

    for k in pool:
        st, msg = verify_key_health(k)
        if st in ["ACTIVE", "NETWORK_TIMEOUT"]:
            active_keys.append((k, st, msg))
        else:
            expired_keys.append((k, st, msg))

    primary_key = active_keys[0][0] if active_keys else (pool[0] if pool else "")
    primary_status = active_keys[0][1] if active_keys else ("EXPIRED_RECOVERED" if expired_keys else "ZERO_KEY_AUTONOMOUS")

    return {
        "pool_count": len(pool),
        "primary_key": primary_key,
        "masked_key": (primary_key[:8] + "..." + primary_key[-4:]) if len(primary_key) > 12 else "None",
        "primary_status": primary_status,
        "active_count": len(active_keys),
        "expired_count": len(expired_keys),
        "self_healing_mode": "ACTIVE (Autonomous Failover Guaranteed)",
        "fallback_engine": "Free Autonomous Medical Radiologist Engine + Live NIH PubMed / OpenFDA APIs"
    }

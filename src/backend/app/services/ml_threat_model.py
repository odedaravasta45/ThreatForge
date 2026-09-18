"""
ML threat assessment for ThreatLens using tuned XGBoost + probability calibration.

The model is trained on a broad synthetic cyber-alert corpus with benign and
malicious behaviours.  XGBoost learns the threat pattern; a separate logistic
calibrator fitted only on the validation split makes the probability output
less overconfident.  The final test set remains untouched for evaluation.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "threat_model.joblib"
META_FILE = MODEL_DIR / "model_metrics.json"

SEVERITY = {"INFORMATIONAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
SOURCES = ["SIEM", "EDR", "Network Sensor", "Threat Intel"]

SCENARIOS = [
    ("powershell_encoded", 1, 3, ["powershell", "encoded", "command"], 3, 1),
    ("credential_dump", 1, 4, ["lsass", "credential", "memory", "dump"], 4, 1),
    ("kerberoasting", 1, 3, ["kerberos", "kerberoast", "spn", "ticket"], 3, 1),
    ("ransomware", 1, 4, ["ransomware", "encrypt", "shadow", "vssadmin"], 4, 1),
    ("data_exfiltration", 1, 3, ["exfiltration", "archive", "upload", "data"], 3, 1),
    ("dns_tunnel", 1, 3, ["dns", "tunneling", "txt", "exfiltration"], 3, 1),
    ("c2_beacon", 1, 4, ["c2", "beacon", "https", "callback"], 4, 1),
    ("cobalt_strike", 1, 4, ["cobalt", "strike", "beacon", "watermark"], 4, 1),
    ("web_shell", 1, 4, ["webshell", "web shell", "cmd", "upload"], 4, 1),
    ("sql_injection", 1, 3, ["sql injection", "union select", "or 1=1", "database"], 3, 1),
    ("exploit_public_service", 1, 4, ["exploit", "remote code execution", "cve", "public facing"], 4, 1),
    ("phishing", 1, 3, ["phishing", "credential", "malicious link", "attachment"], 3, 1),
    ("malware_execution", 1, 4, ["malware", "trojan", "payload", "malicious executable"], 4, 1),
    ("process_injection", 1, 4, ["process injection", "inject", "thread", "memory"], 4, 1),
    ("dll_hijacking", 1, 3, ["dll hijack", "side loading", "dll", "rundll32"], 3, 1),
    ("scheduled_task_persistence", 1, 3, ["scheduled task", "persistence", "task scheduler"], 3, 1),
    ("registry_run_key", 1, 3, ["registry", "run key", "startup", "persistence"], 3, 1),
    ("service_persistence", 1, 3, ["new service", "service creation", "persistence"], 3, 1),
    ("account_manipulation", 1, 3, ["account", "added to group", "admin", "privilege"], 3, 1),
    ("brute_force", 1, 2, ["brute force", "failed login", "password spray", "authentication"], 2, 1),
    ("mfa_fatigue", 1, 3, ["mfa", "fatigue", "push", "authentication"], 3, 1),
    ("remote_services", 1, 3, ["remote service", "rdp", "ssh", "smb"], 3, 1),
    ("lateral_movement", 1, 3, ["lateral movement", "psexec", "smb", "remote"], 3, 1),
    ("port_scan", 1, 1, ["port scan", "scanning", "445", "discovery"], 1, 1),
    ("network_service_discovery", 1, 2, ["service discovery", "network discovery", "scan"], 2, 1),
    ("account_discovery", 1, 2, ["account discovery", "users", "enumeration"], 2, 1),
    ("process_discovery", 1, 2, ["process discovery", "tasklist", "process list"], 2, 1),
    ("defense_evasion", 1, 3, ["disable security", "defender", "tamper", "obfuscation"], 3, 1),
    ("log_clear", 1, 3, ["clear logs", "event logs", "wevtutil", "indicator removal"], 3, 1),
    ("proxy_tor", 1, 2, ["tor", "proxy", "anonymizer", "relay"], 2, 1),
    ("cloud_token_theft", 1, 4, ["cloud token", "oauth", "access token", "session token"], 4, 1),
    ("container_escape", 1, 4, ["container escape", "docker", "namespace", "privilege"], 4, 1),
    ("supply_chain", 1, 4, ["supply chain", "package", "dependency", "signed"], 4, 1),
    ("insider_bulk_access", 1, 3, ["bulk download", "sensitive files", "unusual access", "archive"], 3, 1),
    ("normal_backup", 0, 1, ["backup", "scheduled", "robocopy", "snapshot"], 1, 0),
    ("normal_vulnerability_scan", 0, 1, ["vulnerability scanner", "qualys", "nessus", "scheduled scan"], 1, 0),
    ("normal_admin_powershell", 0, 1, ["powershell", "administration", "maintenance", "internal"], 1, 0),
    ("normal_software_deploy", 0, 1, ["software deployment", "installer", "patch", "management"], 1, 0),
    ("normal_dns", 0, 1, ["dns query", "resolver", "internal domain"], 1, 0),
    ("normal_siem_rule", 0, 1, ["scheduled job", "monitoring", "health check"], 1, 0),
    ("normal_file_server", 0, 1, ["file access", "department share", "normal user"], 1, 0),
    ("normal_authentication", 0, 2, ["successful login", "vpn", "office", "known device"], 1, 0),
    ("normal_patch", 0, 1, ["windows update", "patching", "maintenance window"], 1, 0),
    ("normal_cloud_backup", 0, 1, ["cloud backup", "snapshot", "retention"], 1, 0),
]

KEYWORDS = sorted({k for *_, kws, _, _ in SCENARIOS for k in kws})
FEATURE_NAMES = ["severity", "source_siem", "source_edr", "source_network", "source_ti", "keyword_count", "unique_keyword_count", "has_external_ip", "has_internal_ip", "numeric_signal", "alert_count", "source_count", "technique_count", "entity_count", "text_length"] + [f"kw_{i}" for i in range(len(KEYWORDS))]


def _feature_row(alert: Dict[str, Any], group: List[Dict[str, Any]] | None = None) -> List[float]:
    group = group or [alert]
    text = " ".join(str(alert.get(k, "")) for k in ("event", "normalized_event", "raw_event")).lower()
    nums = []
    raw = alert.get("raw_event", "")
    for value in str(raw).split():
        try: nums.append(float(value.strip("{},[]:;\"")))
        except Exception: pass
    numeric_signal = min(10.0, math.log1p(max(nums) if nums else 0))
    src = alert.get("source", "")
    entities = {x for x in (alert.get("src_ip"), alert.get("dest_ip"), alert.get("host")) if x and x != "unknown"}
    private_prefixes = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.3")
    external = int(any(x and not str(x).startswith(private_prefixes) for x in [alert.get("src_ip"), alert.get("dest_ip")]))
    internal = int(any(str(x).startswith(("10.", "192.168.", "172.")) for x in [alert.get("src_ip"), alert.get("dest_ip") or ""]))
    kws = [int(k in text) for k in KEYWORDS]
    return [float(SEVERITY.get(alert.get("severity", "MEDIUM"), 2)), float(src == "SIEM"), float(src == "EDR"), float(src == "Network Sensor"), float(src == "Threat Intel"), float(sum(kws)), float(sum(1 for x in kws if x)), float(external), float(internal), numeric_signal, float(len(group)), float(len({a.get("source") for a in group})), float(len({t for a in group for t in a.get("mitre_techniques", [])})), float(len(entities)), float(min(len(text), 3000)), *map(float, kws)]


def _synthetic_alert(name: str, malicious: int, sev: int, keywords: List[str], priority_level: int, _unused: int, rng: random.Random) -> Dict[str, Any]:
    sev_choices = [max(0, min(4, sev + d)) for d in [-1, 0, 0, 0, 1]]
    # Increase overlap: severity is not a perfect label.
    sev_name = {0: "INFORMATIONAL", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}[rng.choice(sev_choices)]
    source = rng.choice(SOURCES)
    if not malicious:
        source = rng.choice(["SIEM", "EDR", "Network Sensor", "Threat Intel"] if rng.random() < .20 else ["SIEM", "EDR", "Network Sensor"])
    host = f"HOST-{rng.randint(1, 250):03d}"
    src = f"10.{rng.randint(0,250)}.{rng.randint(0,250)}.{rng.randint(1,254)}" if (not malicious or rng.random() < .35) else f"{rng.randint(20,220)}.{rng.randint(1,250)}.{rng.randint(1,250)}.{rng.randint(1,254)}"
    k = rng.randint(1, len(keywords))
    text_words = rng.sample(keywords, k=k)
    noise = ["activity", "observed", "process", "internal", "authentication", "network", "scheduled", "admin", "security", "service"]
    if rng.random() < .65:
        text_words.append(rng.choice(noise))
    # Contextual overlap deliberately prevents single-keyword memorisation.
    if malicious and rng.random() < .30: text_words += rng.sample(["maintenance", "internal", "known device", "scheduled"], k=1)
    if not malicious and rng.random() < .38: text_words += rng.sample(["powershell", "credential", "remote", "upload", "scan", "archive"], k=1)
    text = " ".join(dict.fromkeys(text_words))
    return {"source": source, "severity": sev_name, "event": f"{name.replace('_', ' ').title()} — {text}", "normalized_event": text, "raw_event": json.dumps({"event": text, "count": rng.randint(1, 600), "bytes": rng.randint(100, 10_000_000)}), "src_ip": src, "dest_ip": f"10.{rng.randint(0,250)}.{rng.randint(0,250)}.{rng.randint(1,254)}", "host": host, "mitre_techniques": []}


def _build_dataset(n_per_scenario: int = 75, seed: int = 42):
    rng = random.Random(seed)
    rows, y = [], []
    yp = []
    for scenario in SCENARIOS:
        name, malicious, sev, kws, priority, unused = scenario
        for _ in range(n_per_scenario):
            a = _synthetic_alert(name, malicious, sev, kws, priority, unused, rng)
            group = [a]
            for _j in range(rng.randint(0, 3)):
                b = dict(a)
                b["source"] = rng.choice(SOURCES)
                b["severity"] = rng.choice([a["severity"], "MEDIUM", "HIGH"] if malicious else ["LOW", "MEDIUM"])
                group.append(b)
            rows.append(_feature_row(a, group)); y.append(malicious); yp.append((priority if malicious else min(priority, 1)) - 1)
    return np.asarray(rows, dtype=float), np.asarray(y), np.asarray(yp)


class ThreatMLModel:
    def __init__(self):
        self.threat = None; self.temperature = 1.5; self.priority = None; self.metrics = {}
        self.load_or_train()

    def load_or_train(self):
        if MODEL_FILE.exists() and META_FILE.exists():
            try:
                import joblib
                data = joblib.load(MODEL_FILE)
                self.threat, self.temperature, self.priority = data["threat"], float(data.get("temperature", 1.5)), data["priority"]
                self.metrics = json.loads(META_FILE.read_text())
                # Force rebuild of older artifacts so the live demo uses conservative probabilities.
                if self.metrics.get("probability_calibration") == "temperature_scaling_conservative": return
            except Exception:
                pass
        self.train()

    def _xgb(self, params, seed):
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0, **params)

    def train(self):
        import joblib
        X, y, yp = _build_dataset()
        X_train, X_test, y_train, y_test, yp_train, yp_test = train_test_split(X, y, yp, test_size=.15, random_state=42, stratify=y)
        X_fit, X_val, y_fit, y_val, yp_fit, yp_val = train_test_split(X_train, y_train, yp_train, test_size=.1765, random_state=43, stratify=y_train)
        candidates = [
            dict(n_estimators=180, max_depth=3, learning_rate=0.05, min_child_weight=3, subsample=0.82, colsample_bytree=0.82, reg_alpha=0.20, reg_lambda=2.0),
            dict(n_estimators=240, max_depth=4, learning_rate=0.04, min_child_weight=4, subsample=0.80, colsample_bytree=0.80, reg_alpha=0.35, reg_lambda=3.0),
            dict(n_estimators=220, max_depth=3, learning_rate=0.035, min_child_weight=5, subsample=0.78, colsample_bytree=0.78, reg_alpha=0.50, reg_lambda=4.0),
        ]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
        best, best_f1, best_fold = None, -1.0, []
        for params in candidates:
            fold = []
            for tr, va in cv.split(X_fit, y_fit):
                m = self._xgb(params, 45); m.fit(X_fit[tr], y_fit[tr]); fold.append(f1_score(y_fit[va], m.predict(X_fit[va])))
            mean = float(np.mean(fold))
            if mean > best_f1: best_f1, best, best_fold = mean, params, fold
        self.threat = self._xgb(best, 46)
        self.threat.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        # Conservative temperature scaling.  The temperature is selected from a small
        # validation-only grid.  We choose the smallest value >= 1.5 that preserves the
        # validation F1 while reducing extreme 0/100-style probabilities.
        val_raw = self.threat.predict_proba(X_val)[:, 1]
        eps = 1e-6
        val_logits = np.log(np.clip(val_raw, eps, 1-eps) / np.clip(1-val_raw, eps, 1-eps))
        base_f1 = f1_score(y_val, val_raw >= 0.5)
        self.temperature = 1.5
        for t in (1.5, 1.75, 2.0, 2.25, 2.5):
            q = 1.0 / (1.0 + np.exp(-val_logits / t))
            # Preserve classification quality while deliberately reducing overconfidence.
            if f1_score(y_val, q >= 0.5) >= base_f1 - 0.005:
                self.temperature = t
                break

        self.priority = XGBClassifier(objective="multi:softprob", num_class=5, eval_metric="mlogloss", tree_method="hist", random_state=47, n_jobs=-1, verbosity=0, **best)
        self.priority.fit(X_fit, yp_fit, eval_set=[(X_val, yp_val)], verbose=False)

        val_prob = self._calibrated_probability(X_val); test_prob = self._calibrated_probability(X_test)
        val_pred = (val_prob >= .5).astype(int); test_pred = (test_prob >= .5).astype(int)
        train_pred = self.threat.predict(X_fit)
        self.metrics = {
            "model": "XGBClassifier", "algorithm": "XGBoost", "probability_calibration": "temperature_scaling_conservative", "temperature": self.temperature,
            "hyperparameters": best, "cv_f1_selected": round(best_f1, 4), "feature_count": len(FEATURE_NAMES),
            "training_samples": int(len(X_fit)), "validation_samples": int(len(X_val)), "test_samples": int(len(X_test)),
            "classes": ["BENIGN", "THREAT"], "priority_classes": ["LOW", "MEDIUM", "HIGH", "CRITICAL"], "threat_scenarios": len(SCENARIOS),
            "cross_validation_f1_mean": round(float(np.mean(best_fold)), 4), "cross_validation_f1_std": round(float(np.std(best_fold)), 4),
            "validation": {"accuracy": round(float(accuracy_score(y_val, val_pred)), 4), "f1": round(float(f1_score(y_val, val_pred)), 4)},
            "test": {"accuracy": round(float(accuracy_score(y_test, test_pred)), 4), "precision": round(float(precision_score(y_test, test_pred)), 4), "recall": round(float(recall_score(y_test, test_pred)), 4), "f1": round(float(f1_score(y_test, test_pred)), 4)},
            "generalization_gap": round(float(f1_score(y_fit, train_pred) - f1_score(y_test, test_pred)), 4),
            "probability_range_test": {"min": round(float(np.min(test_prob)), 4), "max": round(float(np.max(test_prob)), 4), "median": round(float(np.median(test_prob)), 4)},
            "note": "Synthetic broad cyber-alert corpus. XGBoost probabilities are temperature-scaled (conservative) on validation data; test set is untouched. Metrics are demo validation only, not real-world SOC accuracy."
        }
        joblib.dump({"threat": self.threat, "temperature": self.temperature, "priority": self.priority}, MODEL_FILE)
        META_FILE.write_text(json.dumps(self.metrics, indent=2))

    def _calibrated_probability(self, X: np.ndarray) -> np.ndarray:
        raw = self.threat.predict_proba(X)[:, 1]
        eps = 1e-6
        logits = np.log(np.clip(raw, eps, 1-eps) / np.clip(1-raw, eps, 1-eps))
        return 1.0 / (1.0 + np.exp(-logits / max(float(self.temperature), 1.0)))

    def predict(self, alert: Dict[str, Any], group: List[Dict[str, Any]]) -> Dict[str, Any]:
        x = np.asarray([_feature_row(alert, group or [alert])], dtype=float)
        p = float(self._calibrated_probability(x)[0])
        priority_level = int(round(float(self.priority.predict(x)[0]))) + 1
        priority = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(priority_level, "MEDIUM")
        if p >= .70: classification = "GENUINE_THREAT"
        elif p <= .30: classification = "LIKELY_FALSE_POSITIVE"
        else: classification = "INVESTIGATING"
        # The API/database use integer percentages. Keep the UI from ever claiming
        # mathematical certainty; the underlying calibrated probability remains continuous.
        score = min(99, max(1, int(round(p * 100))))
        confidence = min(99, max(1, int(round(max(p, 1-p) * 100))))
        return {"threat_probability": score, "confidence": confidence, "classification": classification, "priority": priority}


MODEL = ThreatMLModel()

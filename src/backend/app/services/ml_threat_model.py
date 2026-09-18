"""
ML threat assessment for ThreatLens using tuned XGBoost.

The model is deliberately trained on a broad synthetic cyber-alert corpus rather
than the 17 demo alerts.  It learns behavioural/numeric signals and outputs:
  - threat probability (0-100)
  - threat classification
  - priority class

Training uses a stratified train/validation/test split, repeated stratified
cross-validation on the training portion, and a held-out test set.  The test
set is never used for model selection.
"""
from __future__ import annotations

import json
import math
import random
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

# Broad behaviour catalogue: intentionally much larger than the original demo set.
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
FEATURE_NAMES = [
    "severity", "source_siem", "source_edr", "source_network", "source_ti",
    "keyword_count", "unique_keyword_count", "has_external_ip", "has_internal_ip",
    "numeric_signal", "alert_count", "source_count", "technique_count",
    "entity_count", "text_length",
] + [f"kw_{i}" for i in range(len(KEYWORDS))]


def _feature_row(alert: Dict[str, Any], group: List[Dict[str, Any]] | None = None) -> List[float]:
    group = group or [alert]
    text = " ".join(str(alert.get(k, "")) for k in ("event", "normalized_event", "raw_event")).lower()
    nums = []
    for value in alert.get("raw_event", "").split():
        try:
            nums.append(float(value.strip("{},[]:;\"")))
        except Exception:
            pass
    numeric_signal = min(10.0, math.log1p(max(nums) if nums else 0))
    src = alert.get("source", "")
    entities = {x for x in (alert.get("src_ip"), alert.get("dest_ip"), alert.get("host")) if x and x != "unknown"}
    external = int(any(x and not str(x).startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.3")) for x in [alert.get("src_ip"), alert.get("dest_ip")]))
    internal = int(any(str(x).startswith(("10.", "192.168.", "172.")) for x in [alert.get("src_ip"), alert.get("dest_ip") if alert.get("dest_ip") else ""]))
    kws = [int(k in text) for k in KEYWORDS]
    return [
        float(SEVERITY.get(alert.get("severity", "MEDIUM"), 2)),
        float(src == "SIEM"), float(src == "EDR"), float(src == "Network Sensor"), float(src == "Threat Intel"),
        float(sum(kws)), float(sum(1 for x in kws if x)), float(external), float(internal), numeric_signal,
        float(len(group)), float(len({a.get("source") for a in group})),
        float(len({t for a in group for t in a.get("mitre_techniques", [])})), float(len(entities)), float(min(len(text), 3000)),
        *map(float, kws)
    ]


def _synthetic_alert(name: str, malicious: int, sev: int, keywords: List[str], priority_level: int, _unused: int, rng: random.Random) -> Dict[str, Any]:
    sev_name = {0: "INFORMATIONAL", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}[max(0, min(4, sev + rng.choice([-1, 0, 0, 0, 1])))]
    source = rng.choice(SOURCES)
    if not malicious and name.startswith("normal_"):
        source = rng.choice(["SIEM", "EDR", "Network Sensor"])
    host = f"HOST-{rng.randint(1, 250):03d}"
    src = f"10.{rng.randint(0,250)}.{rng.randint(0,250)}.{rng.randint(1,254)}" if not malicious or rng.random() < .35 else f"{rng.randint(20,220)}.{rng.randint(1,250)}.{rng.randint(1,250)}.{rng.randint(1,254)}"
    text = " ".join(rng.sample(keywords, k=rng.randint(max(1, len(keywords)//2), len(keywords))))
    # Add benign/malicious contextual noise so the model cannot memorize a single keyword.
    if malicious and rng.random() < .25:
        text += " routine internal activity"
    if not malicious and rng.random() < .30:
        text += " powershell security process"
    return {
        "source": source, "severity": sev_name, "event": f"{name.replace('_', ' ').title()} — {text}",
        "normalized_event": text, "raw_event": json.dumps({"event": text, "count": rng.randint(1, 600), "bytes": rng.randint(100, 10_000_000)}),
        "src_ip": src, "dest_ip": f"10.{rng.randint(0,250)}.{rng.randint(0,250)}.{rng.randint(1,254)}", "host": host,
        "mitre_techniques": [],
    }


def _build_dataset(n_per_scenario: int = 55, seed: int = 42):
    rng = random.Random(seed)
    rows, y_threat, y_priority = [], [], []
    for scenario in SCENARIOS:
        for _ in range(n_per_scenario):
            name, malicious, sev, kws, priority, unused = scenario
            a = _synthetic_alert(name, malicious, sev, kws, priority, unused, rng)
            # A small amount of contextual variability in group evidence.
            group = [a]
            for _j in range(rng.randint(0, 3)):
                b = dict(a)
                b["source"] = rng.choice(SOURCES)
                b["severity"] = rng.choice([a["severity"], "MEDIUM", "HIGH"] if malicious else ["LOW", "MEDIUM"])
                group.append(b)
            rows.append(_feature_row(a, group))
            y_threat.append(malicious)
            # XGBoost multiclass labels must be contiguous 0..K-1.
            y_priority.append((priority if malicious else min(priority, 1)) - 1)
    return np.asarray(rows, dtype=float), np.asarray(y_threat), np.asarray(y_priority)


class ThreatMLModel:
    def __init__(self):
        self.threat = None
        self.priority = None
        self.metrics: Dict[str, Any] = {}
        self.load_or_train()

    def load_or_train(self):
        if MODEL_FILE.exists() and META_FILE.exists():
            try:
                import joblib
                data = joblib.load(MODEL_FILE)
                self.threat, self.priority = data["threat"], data["priority"]
                self.metrics = json.loads(META_FILE.read_text())
                return
            except Exception:
                pass
        self.train()

    def train(self):
        import joblib
        X, y, yp = _build_dataset()
        X_train, X_test, y_train, y_test, yp_train, yp_test = train_test_split(
            X, y, yp, test_size=.15, random_state=42, stratify=y
        )
        # Validation is carved out before final fitting; CV is performed only on training.
        X_fit, X_val, y_fit, y_val, yp_fit, yp_val = train_test_split(
            X_train, y_train, yp_train, test_size=.1765, random_state=43, stratify=y_train
        )
        # Tuned XGBoost candidates. Selection uses CV on the fit split only.
        candidates = [
            dict(n_estimators=180, max_depth=3, learning_rate=0.05, min_child_weight=2,
                 subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=1.5),
            dict(n_estimators=240, max_depth=4, learning_rate=0.04, min_child_weight=2,
                 subsample=0.85, colsample_bytree=0.85, reg_alpha=0.20, reg_lambda=2.0),
            dict(n_estimators=220, max_depth=5, learning_rate=0.035, min_child_weight=3,
                 subsample=0.80, colsample_bytree=0.80, reg_alpha=0.25, reg_lambda=2.5),
        ]
        best, best_f1, best_fold = None, -1.0, []
        cv_scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
        for params in candidates:
            fold = []
            for tr, va in cv.split(X_fit, y_fit):
                m = XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                    random_state=45, n_jobs=-1, verbosity=0, **params
                )
                m.fit(X_fit[tr], y_fit[tr])
                fold.append(f1_score(y_fit[va], m.predict(X_fit[va])))
            mean = float(np.mean(fold))
            if mean > best_f1:
                best_f1, best, best_fold = mean, params, fold
        cv_scores = best_fold

        self.threat = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=46, n_jobs=-1, verbosity=0, **best
        )
        self.threat.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        self.priority = XGBClassifier(
            objective="multi:softprob", num_class=5, eval_metric="mlogloss", tree_method="hist",
            random_state=47, n_jobs=-1, verbosity=0, **best
        )
        self.priority.fit(X_fit, yp_fit, eval_set=[(X_val, yp_val)], verbose=False)
        val_pred = self.threat.predict(X_val)
        test_pred = self.threat.predict(X_test)
        test_prob = self.threat.predict_proba(X_test)[:, 1]
        self.metrics = {
            "model": "XGBClassifier",
            "algorithm": "XGBoost",
            "hyperparameters": best,
            "cv_f1_selected": round(float(best_f1), 4),
            "feature_count": len(FEATURE_NAMES), "training_samples": int(len(X_fit)),
            "validation_samples": int(len(X_val)), "test_samples": int(len(X_test)),
            "classes": ["BENIGN", "THREAT"], "priority_classes": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "threat_scenarios": len(SCENARIOS),
            "cross_validation_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cross_validation_f1_std": round(float(np.std(cv_scores)), 4),
            "validation": {"accuracy": round(float(accuracy_score(y_val, val_pred)), 4), "f1": round(float(f1_score(y_val, val_pred)), 4)},
            "test": {"accuracy": round(float(accuracy_score(y_test, test_pred)), 4), "precision": round(float(precision_score(y_test, test_pred)), 4), "recall": round(float(recall_score(y_test, test_pred)), 4), "f1": round(float(f1_score(y_test, test_pred)), 4)},
            "generalization_gap": round(float(f1_score(y_fit, self.threat.predict(X_fit)) - f1_score(y_test, test_pred)), 4),
            "note": "Synthetic broad cyber-alert corpus for demo validation; not a claim of real-world SOC accuracy.",
        }
        joblib.dump({"threat": self.threat, "priority": self.priority}, MODEL_FILE)
        META_FILE.write_text(json.dumps(self.metrics, indent=2))

    def predict(self, alert: Dict[str, Any], group: List[Dict[str, Any]]) -> Dict[str, Any]:
        x = np.asarray([_feature_row(alert, group)], dtype=float)
        p = float(self.threat.predict_proba(x)[0][1])
        priority_level = int(round(float(self.priority.predict(x)[0]))) + 1
        priority = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(priority_level, "MEDIUM")
        if p >= .70:
            classification = "GENUINE_THREAT"
        elif p <= .30:
            classification = "LIKELY_FALSE_POSITIVE"
        else:
            classification = "INVESTIGATING"
        confidence = int(round(max(p, 1-p) * 100))
        score = int(round(p * 100))
        return {"threat_probability": score, "confidence": confidence, "classification": classification, "priority": priority}

    def model_info(self) -> Dict[str, Any]:
        return {**self.metrics, "feature_names": FEATURE_NAMES, "model_file": str(MODEL_FILE.name)}


MODEL = ThreatMLModel()

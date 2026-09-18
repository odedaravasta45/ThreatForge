"""
Deterministic alert correlation engine.

Groups alerts into incidents using four criteria (matching the TypeScript
correlateAlerts() in threatEngine.ts):

1. Shared source/destination IP or host (entity overlap)
2. Temporal proximity — alerts within a configurable time window
3. MITRE ATT&CK technique overlap
4. Explicit source-diversity bonus (cross-feed corroboration)

Produces an explainable `whyCorrelated` narrative and a numeric `threatScore`.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from .normalizer import SEVERITY_WEIGHTS
from .ml_threat_model import MODEL

CORRELATION_WINDOW_MINUTES = 15


def _parse_ts(ts: str) -> Optional[datetime]:
    """Parse 'YYYY-MM-DD HH:MM:SS' or ISO string to datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _within_window(ts_a: str, ts_b: str, minutes: int = CORRELATION_WINDOW_MINUTES) -> bool:
    a = _parse_ts(ts_a)
    b = _parse_ts(ts_b)
    if a is None or b is None:
        return True  # conservative: assume within window if unparseable
    delta = abs((a - b).total_seconds())
    return delta <= minutes * 60


def _entity_overlap(a: Dict, b: Dict) -> bool:
    a_entities = {v for v in [a["src_ip"], a["dest_ip"], a["host"]] if v != "unknown"}
    b_entities = {v for v in [b["src_ip"], b["dest_ip"], b["host"]] if v != "unknown"}
    return bool(a_entities & b_entities)


def _technique_overlap(a: Dict, b: Dict) -> bool:
    return bool(set(a["mitre_techniques"]) & set(b["mitre_techniques"]))


def _find_group(alert: Dict, candidates: List[Dict]) -> List[Dict]:
    """Return all candidates that correlate with `alert` (including itself)."""
    group = []
    for c in candidates:
        if c["id"] == alert["id"]:
            group.append(c)
            continue
        within = _within_window(alert["timestamp"], c["timestamp"])
        # Entity overlap is the primary correlation guardrail. MITRE overlap can
        # be used when network entities are unavailable, but must not merge two
        # otherwise distinct, fully-observed assets merely because they used the
        # same technique at the same time.
        entity_match = _entity_overlap(alert, c)
        network_entities_missing = (
            alert.get("src_ip") == "unknown" and alert.get("dest_ip") == "unknown"
            and c.get("src_ip") == "unknown" and c.get("dest_ip") == "unknown"
        )
        technique_match = _technique_overlap(alert, c)
        if within and (entity_match or (network_entities_missing and technique_match)):
            group.append(c)
    return group


def _score_group(group: List[Dict]) -> Tuple[int, int, str, str]:
    """Use the trained ML model for threat probability, confidence, classification and priority."""
    # Aggregate evidence into a representative alert while retaining the full group as context.
    representative = max(group, key=lambda a: SEVERITY_WEIGHTS.get(a.get("severity", "MEDIUM"), 50))
    result = MODEL.predict(representative, group)
    return result["threat_probability"], result["confidence"], result["classification"], result["priority"]


def correlate(
    new_alerts: List[Dict],
    existing_alerts: List[Dict],
    existing_incidents: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Correlate `new_alerts` against all known alerts and existing incidents.

    Returns:
        updated_alerts   — new_alerts with correlation_id/status applied
        new_incidents    — newly created incident dicts
        updated_incidents — existing incidents that gained new alerts
    """
    all_alerts = existing_alerts + new_alerts
    # Work on copies so we don't mutate caller data
    working = [dict(a) for a in all_alerts]
    alert_by_id = {a["id"]: a for a in working}

    visited: Set[str] = set()
    groups: List[List[Dict]] = []

    for alert in working:
        if alert["id"] in visited:
            continue
        group = _find_group(alert, working)
        if len(group) >= 2:
            for g in group:
                visited.add(g["id"])
            groups.append(group)

    # Build incident index from existing incidents
    incident_index: Dict[str, Dict] = {i["id"]: dict(i) for i in existing_incidents}
    new_inc_ids: Set[str] = set()
    updated_inc_ids: Set[str] = set()

    for group in groups:
        # Prefer existing correlation id if any alert in the group already has one
        existing_id = next(
            (a.get("correlation_id") for a in group if a.get("correlation_id")),
            None,
        )
        inc_id = existing_id or _new_inc_id(set(incident_index.keys()))

        score, confidence, classification, priority = _score_group(group)
        sorted_group = sorted(group, key=lambda a: a["timestamp"])
        techniques = list(dict.fromkeys(t for a in group for t in a["mitre_techniques"]))
        source_count = len({a["source"] for a in group})
        technique_count = len(techniques)
        max_sev_str = max(
            group, key=lambda a: SEVERITY_WEIGHTS.get(a["severity"], 50)
        )["severity"]

        timeline = [
            {
                "time": a["timestamp"],
                "title": a["event"],
                "description": a["normalized_event"],
                "source": a["source"],
                "severity": a["severity"],
                "techniqueId": a["mitre_techniques"][0] if a["mitre_techniques"] else None,
            }
            for a in sorted_group
        ]

        affected_assets = list(
            dict.fromkeys(a["host"] for a in group if a["host"] != "unknown")
        )

        why = (
            f"{source_count} independent feed{'s' if source_count > 1 else ''} corroborated "
            f"{len(group)} alert{'s' if len(group) > 1 else ''} through shared entities, "
            f"a {CORRELATION_WINDOW_MINUTES}-minute time window, and "
            f"{technique_count} overlapping MITRE technique{'s' if technique_count != 1 else ''}. "
            f"ML threat probability: {score}%."
        )

        inc = {
            "id": inc_id,
            "title": (
                f"{'Correlated Threat' if classification == 'GENUINE_THREAT' else 'Suspicious Activity'}"
                f" — {sorted_group[0]['host']}"
            ),
            "priority": priority,
            "confidence": confidence,
            "threat_score": score,
            "related_alerts_count": len(group),
            "mitre_techniques": techniques,
            "status": "NEW" if classification == "GENUINE_THREAT" else "INVESTIGATING",
            "classification": classification,
            "first_seen": sorted_group[0]["timestamp"],
            "last_seen": sorted_group[-1]["timestamp"],
            "affected_assets": affected_assets,
            "why_correlated": why,
            "timeline": timeline,
            "source_count": source_count,
            "ml_model": "XGBClassifier",
            "ml_threat_probability": score,
        }

        if existing_id and existing_id in incident_index:
            incident_index[existing_id].update(inc)
            updated_inc_ids.add(existing_id)
        else:
            incident_index[inc_id] = inc
            new_inc_ids.add(inc_id)

        # Stamp each alert in the group
        new_status = "FALSE_POSITIVE" if classification == "LIKELY_FALSE_POSITIVE" else "CORRELATED"
        for a in group:
            alert_by_id[a["id"]]["correlation_id"] = inc_id
            alert_by_id[a["id"]]["status"] = new_status

    # Only return the new_alerts slice with their updated fields
    updated_new = [alert_by_id[a["id"]] for a in new_alerts]
    new_incidents = [incident_index[i] for i in new_inc_ids]
    updated_incidents = [incident_index[i] for i in updated_inc_ids]

    return updated_new, new_incidents, updated_incidents


def _new_inc_id(existing_ids: Set[str]) -> str:
    """Return the next unused numeric incident ID, regardless of gaps."""
    numbers = []
    for incident_id in existing_ids:
        if isinstance(incident_id, str) and incident_id.startswith("INC-"):
            try:
                numbers.append(int(incident_id[4:]))
            except ValueError:
                continue
    candidate = max(numbers, default=0) + 1
    while f"INC-{candidate:03d}" in existing_ids:
        candidate += 1
    return f"INC-{candidate:03d}"


def calculate_dashboard(alerts: List[Dict], incidents: List[Dict]) -> Dict:
    fp = sum(1 for a in alerts if a.get("status") == "FALSE_POSITIVE")
    today = datetime.now(timezone.utc).date().isoformat()
    return {
        "totalAlerts": len(alerts),
        "totalAlertsTrend": "+ live",
        "correlatedIncidents": len(incidents),
        "correlatedIncidentsToday": sum(
            1 for i in incidents
            if str(i.get("last_seen", ""))[:10] == today
        ),
        "criticalThreats": sum(1 for i in incidents if i.get("priority") == "CRITICAL"),
        "highPriority": sum(1 for i in incidents if i.get("priority") == "HIGH"),
        "falsePositives": fp,
        "falsePositivesPercentage": round(fp / len(alerts) * 100, 1) if alerts else 0.0,
    }

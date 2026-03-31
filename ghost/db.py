import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        Float,
        String,
        DateTime,
        Text,
        func,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


Base = declarative_base() if SQLALCHEMY_AVAILABLE else object


class Zone(Base):  # type: ignore[misc]
    __tablename__ = "zones"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False, unique=True)
    camera_source = Column(String(500), nullable=True)
    active = Column(Integer, default=1, nullable=False)


class WasteEvent(Base):  # type: ignore[misc]
    __tablename__ = "waste_events"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    room = Column(String(255), nullable=False)
    zone_name = Column(String(120), nullable=False, default="default")
    duration_seconds = Column(Float, nullable=False)
    status = Column(String(64), nullable=False, default="ALERT_SENT")
    money_wasted = Column(Float, nullable=False, default=0.0)
    snapshot_path = Column(Text, nullable=True)


class FocusSession(Base):  # type: ignore[misc]
    __tablename__ = "focus_sessions"
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False)
    zone_name = Column(String(120), nullable=False, default="default")


_engine = None
_Session = None


def is_ready() -> bool:
    return SQLALCHEMY_AVAILABLE and _engine is not None and _Session is not None


def init_db():
    global _engine, _Session
    if not SQLALCHEMY_AVAILABLE:
        return False

    db_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not db_url:
        db_url = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'visioncore.db')}"

    _engine = create_engine(db_url, pool_pre_ping=True)
    _Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(_engine)
    _ensure_default_zone()
    return True


def _ensure_default_zone():
    if not is_ready():
        return
    s = _Session()
    try:
        z = s.query(Zone).filter_by(name="default").first()
        if not z:
            s.add(Zone(name="default", camera_source=None, active=1))
            s.commit()
    finally:
        s.close()


def add_waste_event(
    *,
    timestamp: datetime,
    room: str,
    duration_seconds: float,
    status: str,
    money_wasted: float,
    snapshot_path: Optional[str] = None,
    zone_name: str = "default",
):
    if not is_ready():
        return False
    s = _Session()
    try:
        s.add(
            WasteEvent(
                timestamp=timestamp,
                room=room,
                zone_name=zone_name,
                duration_seconds=float(duration_seconds),
                status=status,
                money_wasted=float(money_wasted),
                snapshot_path=snapshot_path,
            )
        )
        s.commit()
        return True
    except Exception:
        s.rollback()
        return False
    finally:
        s.close()


def list_history(limit: int = 50) -> List[Dict[str, Any]]:
    if not is_ready():
        return []
    s = _Session()
    try:
        rows = (
            s.query(WasteEvent)
            .order_by(WasteEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Room": r.room,
                "Duration_Seconds": round(float(r.duration_seconds), 2),
                "Status": r.status,
                "Money_Wasted": round(float(r.money_wasted), 2),
                "Snapshot_Path": r.snapshot_path,
                "Zone": r.zone_name,
            }
            for r in rows
        ]
    finally:
        s.close()


def heatmap_cells(target_date: Optional[datetime.date] = None):
    if not is_ready():
        return {"cells": [], "max_seconds": 0}
    s = _Session()
    try:
        q = s.query(WasteEvent).filter(WasteEvent.status == "ALERT_SENT")
        if target_date is not None:
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            q = q.filter(WasteEvent.timestamp >= start_dt, WasteEvent.timestamp < end_dt)
        rows = q.all()
        totals = {}
        for r in rows:
            k = (r.timestamp.weekday(), r.timestamp.hour)
            totals[k] = totals.get(k, 0.0) + float(r.duration_seconds)
        max_seconds = max(totals.values()) if totals else 0.0
        cells = [{"day": int(d), "hour": int(h), "seconds": round(v, 2)} for (d, h), v in totals.items() if v > 0]
        return {"cells": cells, "max_seconds": round(max_seconds, 2)}
    finally:
        s.close()


def list_evidence(limit: int = 60):
    if not is_ready():
        return []
    s = _Session()
    try:
        rows = (
            s.query(WasteEvent)
            .filter(WasteEvent.snapshot_path.isnot(None))
            .order_by(WasteEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(float(r.duration_seconds), 2),
                "money_wasted": round(float(r.money_wasted), 2),
                "snapshot_path": r.snapshot_path,
                "room": r.room,
                "zone": r.zone_name,
            }
            for r in rows
        ]
    finally:
        s.close()


def import_csv_to_db_once():
    """Reads Reports CSV and performs a one-time migration if DB is empty."""
    if not is_ready():
        return False
        
    import csv
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')
    if not os.path.exists(log_file):
        return True

    s = _Session()
    try:
        if s.query(WasteEvent).count() > 0:
            return True
            
        with open(log_file, 'r', newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get('Timestamp', '').strip()
                if not ts_str:
                    continue
                try:
                    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    continue
                
                s.add(WasteEvent(
                    timestamp=dt,
                    room=row.get('Room', 'Unknown'),
                    zone_name="default",
                    duration_seconds=float(row.get('Duration_Seconds', 0) or 0),
                    status=row.get('Status', 'ALERT_SENT'),
                    money_wasted=float(row.get('Money_Wasted', 0) or 0),
                    snapshot_path=None
                ))
        s.commit()
        print("[DB] Initialized from historical CSV successfully.")
        return True
    except Exception as e:
        print(f"[DB IMPORT] Failed to import CSV: {e}")
        s.rollback()
        return False
    finally:
        s.close()


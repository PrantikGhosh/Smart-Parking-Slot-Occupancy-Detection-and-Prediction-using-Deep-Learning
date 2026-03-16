"""
Parking Time Series Prediction - Database Manager
Handles all database operations for parking lots, annotations, occupancy events, and predictions
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class ParkingDatabase:
    """Database manager for parking prediction system"""
    
    def __init__(self, db_path: str = "database/parking_data.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()
    
    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        # Create database directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Read schema and execute
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        with self.get_connection() as conn:
            conn.executescript(schema)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    
    # ==================== Parking Lots ====================
    
    def save_parking_lot(self, name: str, video_path: str, video_hash: str,
                        camera_angle: str = 'other', fps: float = None,
                        duration: float = None, width: int = None, 
                        height: int = None) -> int:
        """
        Save parking lot metadata
        
        Returns:
            parking_lot_id: ID of saved record
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO parking_lots 
                (name, video_path, video_hash, camera_angle, fps, 
                 video_duration_seconds, frame_width, frame_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, video_path, video_hash, camera_angle, fps, 
                  duration, width, height))
            return cursor.lastrowid
    
    def get_parking_lot_by_hash(self, video_hash: str) -> Optional[Dict]:
        """Check if video already annotated by hash"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM parking_lots WHERE video_hash = ?",
                (video_hash,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_parking_lot_by_id(self, lot_id: int) -> Optional[Dict]:
        """Get parking lot by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM parking_lots WHERE id = ?",
                (lot_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_all_parking_lots(self) -> List[Dict]:
        """Get all parking lots"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM parking_lots ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    
    def update_parking_lot_slots(self, lot_id: int, total_slots: int):
        """Update total slots count for parking lot"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE parking_lots SET total_slots = ? WHERE id = ?",
                (total_slots, lot_id)
            )
    
    # ==================== Slot Annotations ====================
    
    def save_slot_annotation(self, parking_lot_id: int, slot_id: str,
                            x1: int, y1: int, x2: int, y2: int,
                            slot_type: str = 'regular'):
        """Save single slot annotation"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO slot_annotations
                (parking_lot_id, slot_id, x1, y1, x2, y2, slot_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (parking_lot_id, slot_id, x1, y1, x2, y2, slot_type))
    
    def save_slot_annotations_batch(self, parking_lot_id: int, 
                                   annotations: List[Dict]):
        """
        Save multiple slot annotations
        
        Args:
            annotations: List of dicts with keys: slot_id, x1, y1, x2, y2, slot_type
        """
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO slot_annotations
                (parking_lot_id, slot_id, x1, y1, x2, y2, slot_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(parking_lot_id, a['slot_id'], a['x1'], a['y1'], 
                   a['x2'], a['y2'], a.get('slot_type', 'regular'))
                  for a in annotations])
    
    def get_slot_annotations(self, parking_lot_id: int) -> List[Dict]:
        """Get all slot annotations for a parking lot"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM slot_annotations 
                WHERE parking_lot_id = ?
                ORDER BY slot_id
            """, (parking_lot_id,)).fetchall()
            return [dict(row) for row in rows]
    
    def delete_slot_annotations(self, parking_lot_id: int):
        """Delete all annotations for a parking lot"""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM slot_annotations WHERE parking_lot_id = ?",
                (parking_lot_id,)
            )
            conn.commit()
    
    def delete_occupancy_events(self, parking_lot_id: int):
        """Delete all occupancy events for a parking lot"""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM occupancy_events WHERE parking_lot_id = ?",
                (parking_lot_id,)
            )
            conn.commit()
    
    # ==================== Occupancy Events ====================
    
    def save_occupancy_event(self, parking_lot_id: int, slot_id: str,
                            timestamp: datetime, frame_number: int,
                            status: str, confidence: float = None,
                            detected_class: str = None):
        """Save single occupancy event"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO occupancy_events
                (parking_lot_id, slot_id, timestamp, frame_number, 
                 status, confidence, detected_class)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (parking_lot_id, slot_id, timestamp, frame_number,
                  status, confidence, detected_class))
    
    def save_occupancy_events_batch(self, events: List[Dict]):
        """
        Save multiple occupancy events
        
        Args:
            events: List of dicts with keys: parking_lot_id, slot_id, 
                   timestamp, frame_number, status, confidence, detected_class
        """
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT INTO occupancy_events
                (parking_lot_id, slot_id, timestamp, frame_number, 
                 status, confidence, detected_class)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(e['parking_lot_id'], e['slot_id'], e['timestamp'],
                   e['frame_number'], e['status'], e.get('confidence'),
                   e.get('detected_class')) for e in events])
    
    def get_occupancy_events(self, parking_lot_id: int, slot_id: str = None,
                            start_time: datetime = None, 
                            end_time: datetime = None,
                            limit: int = None) -> List[Dict]:
        """
        Get occupancy events with optional filters
        
        Args:
            parking_lot_id: Parking lot ID
            slot_id: Optional slot ID filter
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            limit: Optional limit on number of results
        """
        query = "SELECT * FROM occupancy_events WHERE parking_lot_id = ?"
        params = [parking_lot_id]
        
        if slot_id:
            query += " AND slot_id = ?"
            params.append(slot_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {int(limit)}"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def get_occupancy_count(self, parking_lot_id: int) -> int:
        """Get total count of occupancy events"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM occupancy_events WHERE parking_lot_id = ?",
                (parking_lot_id,)
            ).fetchone()
            return row['count'] if row else 0
    
    # ==================== Predictions ====================
    
    def save_prediction(self, parking_lot_id: int, slot_id: str,
                       prediction_time: datetime, target_time: datetime,
                       model_type: str, probability_free: float,
                       expected_wait_minutes: int = None, 
                       confidence: float = None):
        """Save prediction result"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO predictions
                (parking_lot_id, slot_id, prediction_timestamp, 
                 target_timestamp, model_type, probability_free, 
                 expected_wait_minutes, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (parking_lot_id, slot_id, prediction_time, target_time,
                  model_type, probability_free, expected_wait_minutes, confidence))
    
    def get_predictions(self, parking_lot_id: int, slot_id: str = None,
                       model_type: str = None) -> List[Dict]:
        """Get predictions with optional filters"""
        query = "SELECT * FROM predictions WHERE parking_lot_id = ?"
        params = [parking_lot_id]
        
        if slot_id:
            query += " AND slot_id = ?"
            params.append(slot_id)
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        query += " ORDER BY target_timestamp"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    # ==================== Utility Functions ====================
    
    @staticmethod
    def compute_video_hash(video_path: str) -> str:
        """Compute SHA256 hash of video file for duplicate detection"""
        hasher = hashlib.sha256()
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_statistics(self, parking_lot_id: int) -> Dict:
        """Get summary statistics for a parking lot"""
        with self.get_connection() as conn:
            # Get basic info
            lot = self.get_parking_lot_by_id(parking_lot_id)
            if not lot:
                return {}
            
            # Get slot count
            slot_count = conn.execute(
                "SELECT COUNT(*) as count FROM slot_annotations WHERE parking_lot_id = ?",
                (parking_lot_id,)
            ).fetchone()['count']
            
            # Get event count
            event_count = self.get_occupancy_count(parking_lot_id)
            
            # Get prediction count
            pred_count = conn.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE parking_lot_id = ?",
                (parking_lot_id,)
            ).fetchone()['count']
            
            return {
                'parking_lot_name': lot['name'],
                'total_slots': slot_count,
                'total_events': event_count,
                'total_predictions': pred_count,
                'video_duration': lot['video_duration_seconds'],
                'fps': lot['fps']
            }


# Singleton instance
_db_instance = None

def get_database() -> ParkingDatabase:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ParkingDatabase()
    return _db_instance

-- Parking Time Series Prediction Database Schema
-- SQLite database for tracking parking lots, annotations, occupancy events, and predictions

-- Table 1: Parking Lots
-- Stores metadata for each uploaded and processed video
CREATE TABLE IF NOT EXISTS parking_lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    video_hash TEXT UNIQUE NOT NULL,
    annotation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_slots INTEGER DEFAULT 0,
    camera_angle TEXT CHECK(camera_angle IN ('top_down', 'angled', 'other')) DEFAULT 'other',
    fps REAL,
    video_duration_seconds REAL,
    frame_width INTEGER,
    frame_height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Slot Annotations
-- Stores one-time manual annotations for parking slots
CREATE TABLE IF NOT EXISTS slot_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_lot_id INTEGER NOT NULL,
    slot_id TEXT NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    slot_type TEXT CHECK(slot_type IN ('regular', 'handicap', 'reserved', 'other')) DEFAULT 'regular',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots(id) ON DELETE CASCADE,
    UNIQUE(parking_lot_id, slot_id)
);

-- Table 3: Occupancy Events
-- Time series data extracted from video processing
CREATE TABLE IF NOT EXISTS occupancy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_lot_id INTEGER NOT NULL,
    slot_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    frame_number INTEGER,
    status TEXT CHECK(status IN ('empty', 'occupied', 'unknown')) NOT NULL,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    detected_class TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots(id) ON DELETE CASCADE
);

-- Table 4: Predictions
-- Cached predictions for quick retrieval
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_lot_id INTEGER NOT NULL,
    slot_id TEXT NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    model_type TEXT CHECK(model_type IN ('lstm', 'prophet', 'ensemble')) NOT NULL,
    probability_free REAL CHECK(probability_free >= 0 AND probability_free <= 1),
    expected_wait_minutes INTEGER,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_parking_lots_hash ON parking_lots(video_hash);
CREATE INDEX IF NOT EXISTS idx_slot_annotations_lot ON slot_annotations(parking_lot_id);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_lot_slot ON occupancy_events(parking_lot_id, slot_id);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_timestamp ON occupancy_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_lot_slot ON predictions(parking_lot_id, slot_id);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_timestamp);

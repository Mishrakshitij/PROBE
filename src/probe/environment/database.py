"""FlightDB: SQLite schema, queries, connection management."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class FlightDB:
    """Thin wrapper around a SQLite flight-booking database."""

    def __init__(self, db_path: str = "data/flights.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── connection ─────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── schema ─────────────────────────────────────────────────────

    def init_schema(self):
        """Create all 10 tables."""
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS airlines (
            airline_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            alliance     TEXT,
            is_active    INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS airports (
            airport_code TEXT PRIMARY KEY,
            city         TEXT NOT NULL,
            country      TEXT NOT NULL,
            timezone     TEXT NOT NULL,
            is_sanctioned INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sanctioned_countries (
            country_code TEXT PRIMARY KEY,
            sanction_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS routes (
            route_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            airline_id    INTEGER NOT NULL REFERENCES airlines(airline_id),
            origin        TEXT NOT NULL REFERENCES airports(airport_code),
            destination   TEXT NOT NULL REFERENCES airports(airport_code),
            is_restricted INTEGER NOT NULL DEFAULT 0,
            reason        TEXT
        );

        CREATE TABLE IF NOT EXISTS flights (
            flight_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id        INTEGER NOT NULL REFERENCES routes(route_id),
            flight_number   TEXT NOT NULL,
            departure_date  TEXT NOT NULL,
            departure_time  TEXT NOT NULL,
            arrival_date    TEXT NOT NULL,
            arrival_time    TEXT NOT NULL,
            aircraft_type   TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'scheduled'
                CHECK(status IN ('scheduled','cancelled','full'))
        );

        CREATE TABLE IF NOT EXISTS fare_classes (
            fare_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id                  INTEGER NOT NULL REFERENCES flights(flight_id),
            fare_class                 TEXT NOT NULL,
            price_usd                  REAL NOT NULL,
            cost_usd                   REAL NOT NULL,
            seats_total                INTEGER NOT NULL,
            seats_available            INTEGER NOT NULL,
            is_refundable              INTEGER NOT NULL DEFAULT 0,
            change_fee_usd             REAL NOT NULL DEFAULT 0,
            min_advance_days           INTEGER NOT NULL DEFAULT 0,
            max_passengers_per_booking INTEGER NOT NULL DEFAULT 9
        );

        CREATE TABLE IF NOT EXISTS bookings (
            booking_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            fare_id        INTEGER NOT NULL REFERENCES fare_classes(fare_id),
            passenger_count INTEGER NOT NULL,
            passenger_info TEXT,
            total_price    REAL NOT NULL,
            status         TEXT NOT NULL DEFAULT 'confirmed'
                CHECK(status IN ('confirmed','cancelled','held','pending')),
            contact_email  TEXT,
            contact_phone  TEXT,
            created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS itineraries (
            itinerary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            booking_id   INTEGER REFERENCES bookings(booking_id),
            status       TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS itinerary_legs (
            leg_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            itinerary_id  INTEGER NOT NULL REFERENCES itineraries(itinerary_id),
            flight_id     INTEGER NOT NULL REFERENCES flights(flight_id),
            fare_id       INTEGER NOT NULL REFERENCES fare_classes(fare_id),
            leg_order     INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pricing_rules (
            rule_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_type      TEXT NOT NULL,
            parameter_json TEXT NOT NULL
        );
        """)
        self.conn.commit()

    # ── queries ────────────────────────────────────────────────────

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ── domain helpers ─────────────────────────────────────────────

    def search_flights(
        self, origin: str, destination: str, date: str,
        passengers: int = 1, max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Search available flights — excludes cost_usd (internal only)."""
        return self.query("""
            SELECT f.flight_id, f.flight_number, f.departure_date,
                   f.departure_time, f.arrival_date, f.arrival_time,
                   f.aircraft_type, f.status,
                   r.origin, r.destination, r.is_restricted,
                   a.name AS airline_name,
                   fc.fare_id, fc.fare_class, fc.price_usd,
                   fc.seats_available, fc.is_refundable,
                   fc.change_fee_usd, fc.min_advance_days,
                   fc.max_passengers_per_booking
            FROM flights f
            JOIN routes r ON f.route_id = r.route_id
            JOIN airlines a ON r.airline_id = a.airline_id
            JOIN fare_classes fc ON fc.flight_id = f.flight_id
            WHERE r.origin = ? AND r.destination = ?
              AND f.departure_date = ?
              AND fc.seats_available >= ?
              AND f.status = 'scheduled'
            ORDER BY fc.price_usd ASC
            LIMIT ?
        """, (origin, destination, date, passengers, max_results))

    def get_flight(self, flight_id: int) -> dict[str, Any] | None:
        return self.query_one("""
            SELECT f.*, r.origin, r.destination, r.is_restricted, r.reason,
                   a.name AS airline_name
            FROM flights f
            JOIN routes r ON f.route_id = r.route_id
            JOIN airlines a ON r.airline_id = a.airline_id
            WHERE f.flight_id = ?
        """, (flight_id,))

    def get_fare(self, fare_id: int) -> dict[str, Any] | None:
        """Returns fare including cost_usd (for verifier use only)."""
        return self.query_one("SELECT * FROM fare_classes WHERE fare_id = ?", (fare_id,))

    def get_fare_public(self, fare_id: int) -> dict[str, Any] | None:
        """Returns fare without cost_usd."""
        return self.query_one("""
            SELECT fare_id, flight_id, fare_class, price_usd,
                   seats_available, is_refundable, change_fee_usd,
                   min_advance_days, max_passengers_per_booking
            FROM fare_classes WHERE fare_id = ?
        """, (fare_id,))

    def get_booking(self, booking_id: int) -> dict[str, Any] | None:
        return self.query_one("""
            SELECT b.*, fc.fare_class, fc.price_usd, fc.flight_id,
                   f.flight_number, f.departure_date, f.departure_time
            FROM bookings b
            JOIN fare_classes fc ON b.fare_id = fc.fare_id
            JOIN flights f ON fc.flight_id = f.flight_id
            WHERE b.booking_id = ?
        """, (booking_id,))

    def create_booking(
        self, fare_id: int, passenger_count: int,
        passenger_info: str, total_price: float,
        contact_email: str, contact_phone: str,
    ) -> int:
        return self.execute("""
            INSERT INTO bookings (fare_id, passenger_count, passenger_info,
                                  total_price, status, contact_email, contact_phone)
            VALUES (?, ?, ?, ?, 'confirmed', ?, ?)
        """, (fare_id, passenger_count, passenger_info,
              total_price, contact_email, contact_phone))

    def update_seats(self, fare_id: int, delta: int):
        self.execute("""
            UPDATE fare_classes SET seats_available = seats_available + ?
            WHERE fare_id = ?
        """, (delta, fare_id))

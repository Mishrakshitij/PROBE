"""Load and cache OpenFlights reference data.

Data source: https://openflights.org/data.html
  - airports.dat   → ~7,000 airports worldwide
  - airlines.dat   → ~6,000 airlines
  - routes.dat     → ~67,000 routes

Downloads once to data/reference/, then caches locally.
Falls back to bundled minimal set if offline.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/reference")

URLS = {
    "airports": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "airlines": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat",
    "routes": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
}


@dataclass
class RefAirport:
    code: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    timezone_offset: float
    tz_name: str


@dataclass
class RefAirline:
    airline_id: int
    name: str
    iata: str
    country: str
    active: bool


@dataclass
class RefRoute:
    airline_iata: str
    origin: str
    destination: str


def _download(name: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.dat"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")
    try:
        resp = urllib.request.urlopen(URLS[name], timeout=30)
        text = resp.read().decode("utf-8", errors="replace")
        cache_path.write_text(text)
        return text
    except Exception as e:
        print(f"Warning: could not download {name}: {e}. Using bundled fallback.")
        return ""


def load_airports(min_size: bool = True) -> list[RefAirport]:
    """Load airports. If min_size=True, filter to IATA-coded airports only."""
    raw = _download("airports")
    if not raw:
        return _fallback_airports()
    airports = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 12:
            continue
        iata = row[4].strip().strip('"')
        if min_size and (not iata or iata == "\\N" or len(iata) != 3):
            continue
        try:
            airports.append(RefAirport(
                code=iata,
                name=row[1].strip('"'),
                city=row[2].strip('"'),
                country=row[3].strip('"'),
                lat=float(row[6]),
                lon=float(row[7]),
                timezone_offset=float(row[9]) if row[9] != "\\N" else 0.0,
                tz_name=row[11].strip('"') if row[11] != "\\N" else "UTC",
            ))
        except (ValueError, IndexError):
            continue
    return airports


def load_airlines(active_only: bool = True) -> list[RefAirline]:
    raw = _download("airlines")
    if not raw:
        return _fallback_airlines()
    airlines = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 8:
            continue
        iata = row[3].strip().strip('"')
        if not iata or iata == "\\N" or iata == "-":
            continue
        is_active = row[7].strip().strip('"') == "Y"
        if active_only and not is_active:
            continue
        try:
            airlines.append(RefAirline(
                airline_id=int(row[0]),
                name=row[1].strip('"'),
                iata=iata,
                country=row[6].strip('"'),
                active=is_active,
            ))
        except (ValueError, IndexError):
            continue
    return airlines


def load_routes() -> list[RefRoute]:
    raw = _download("routes")
    if not raw:
        return _fallback_routes()
    routes = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 5:
            continue
        airline = row[0].strip().strip('"')
        src = row[2].strip().strip('"')
        dst = row[4].strip().strip('"')
        if len(src) == 3 and len(dst) == 3 and airline:
            routes.append(RefRoute(airline_iata=airline, origin=src, destination=dst))
    return routes


# ── Fallback data (offline mode) ──────────────────────────────────────

def _fallback_airports() -> list[RefAirport]:
    """Minimal set of 40 major airports for offline use."""
    data = [
        ("JFK", "John F Kennedy Intl", "New York", "United States", 40.64, -73.78, "America/New_York"),
        ("LAX", "Los Angeles Intl", "Los Angeles", "United States", 33.94, -118.41, "America/Los_Angeles"),
        ("ORD", "O'Hare Intl", "Chicago", "United States", 41.97, -87.91, "America/Chicago"),
        ("ATL", "Hartsfield-Jackson", "Atlanta", "United States", 33.64, -84.43, "America/New_York"),
        ("DFW", "Dallas/Fort Worth", "Dallas", "United States", 32.90, -97.04, "America/Chicago"),
        ("SFO", "San Francisco Intl", "San Francisco", "United States", 37.62, -122.38, "America/Los_Angeles"),
        ("MIA", "Miami Intl", "Miami", "United States", 25.79, -80.29, "America/New_York"),
        ("SEA", "Seattle-Tacoma", "Seattle", "United States", 47.45, -122.31, "America/Los_Angeles"),
        ("BOS", "Logan Intl", "Boston", "United States", 42.36, -71.01, "America/New_York"),
        ("DEN", "Denver Intl", "Denver", "United States", 39.86, -104.67, "America/Denver"),
        ("LHR", "Heathrow", "London", "United Kingdom", 51.47, -0.46, "Europe/London"),
        ("CDG", "Charles de Gaulle", "Paris", "France", 49.01, 2.55, "Europe/Paris"),
        ("FRA", "Frankfurt am Main", "Frankfurt", "Germany", 50.03, 8.57, "Europe/Berlin"),
        ("AMS", "Schiphol", "Amsterdam", "Netherlands", 52.31, 4.77, "Europe/Amsterdam"),
        ("MAD", "Barajas", "Madrid", "Spain", 40.47, -3.56, "Europe/Madrid"),
        ("FCO", "Fiumicino", "Rome", "Italy", 41.80, 12.25, "Europe/Rome"),
        ("IST", "Istanbul Airport", "Istanbul", "Turkey", 41.26, 28.74, "Europe/Istanbul"),
        ("NRT", "Narita Intl", "Tokyo", "Japan", 35.76, 140.39, "Asia/Tokyo"),
        ("HND", "Haneda", "Tokyo", "Japan", 35.55, 139.78, "Asia/Tokyo"),
        ("SIN", "Changi", "Singapore", "Singapore", 1.35, 103.99, "Asia/Singapore"),
        ("HKG", "Hong Kong Intl", "Hong Kong", "Hong Kong", 22.31, 113.91, "Asia/Hong_Kong"),
        ("ICN", "Incheon Intl", "Seoul", "South Korea", 37.46, 126.44, "Asia/Seoul"),
        ("DXB", "Dubai Intl", "Dubai", "United Arab Emirates", 25.25, 55.36, "Asia/Dubai"),
        ("BKK", "Suvarnabhumi", "Bangkok", "Thailand", 13.68, 100.75, "Asia/Bangkok"),
        ("DEL", "Indira Gandhi Intl", "New Delhi", "India", 28.57, 77.10, "Asia/Kolkata"),
        ("SYD", "Kingsford Smith", "Sydney", "Australia", -33.95, 151.18, "Australia/Sydney"),
        ("YYZ", "Pearson Intl", "Toronto", "Canada", 43.68, -79.63, "America/Toronto"),
        ("MEX", "Benito Juárez", "Mexico City", "Mexico", 19.44, -99.07, "America/Mexico_City"),
        ("GRU", "Guarulhos", "São Paulo", "Brazil", -23.43, -46.47, "America/Sao_Paulo"),
        ("CPT", "Cape Town Intl", "Cape Town", "South Africa", -33.96, 18.60, "Africa/Johannesburg"),
        ("CAI", "Cairo Intl", "Cairo", "Egypt", 30.12, 31.41, "Africa/Cairo"),
        ("DOH", "Hamad Intl", "Doha", "Qatar", 25.26, 51.57, "Asia/Qatar"),
        ("MUC", "Munich", "Munich", "Germany", 48.35, 11.79, "Europe/Berlin"),
        ("ZRH", "Zurich", "Zurich", "Switzerland", 47.46, 8.55, "Europe/Zurich"),
        ("PEK", "Beijing Capital", "Beijing", "China", 40.08, 116.58, "Asia/Shanghai"),
        ("PVG", "Pudong Intl", "Shanghai", "China", 31.14, 121.81, "Asia/Shanghai"),
        ("FNJ", "Sunan Intl", "Pyongyang", "North Korea", 39.22, 125.67, "Asia/Pyongyang"),
        ("DAM", "Damascus Intl", "Damascus", "Syria", 33.41, 36.51, "Asia/Damascus"),
        ("IKA", "Imam Khomeini", "Tehran", "Iran", 35.42, 51.15, "Asia/Tehran"),
        ("HAV", "José Martí", "Havana", "Cuba", 22.99, -82.41, "America/Havana"),
    ]
    return [RefAirport(code=d[0], name=d[1], city=d[2], country=d[3],
                       lat=d[4], lon=d[5], timezone_offset=0.0, tz_name=d[6])
            for d in data]


def _fallback_airlines() -> list[RefAirline]:
    data = [
        (1, "American Airlines", "AA", "United States"),
        (2, "Delta Air Lines", "DL", "United States"),
        (3, "United Airlines", "UA", "United States"),
        (4, "Southwest Airlines", "WN", "United States"),
        (5, "British Airways", "BA", "United Kingdom"),
        (6, "Lufthansa", "LH", "Germany"),
        (7, "Air France", "AF", "France"),
        (8, "Emirates", "EK", "United Arab Emirates"),
        (9, "Singapore Airlines", "SQ", "Singapore"),
        (10, "Qantas", "QF", "Australia"),
        (11, "Cathay Pacific", "CX", "Hong Kong"),
        (12, "Japan Airlines", "JL", "Japan"),
        (13, "KLM", "KL", "Netherlands"),
        (14, "Turkish Airlines", "TK", "Turkey"),
        (15, "Air Canada", "AC", "Canada"),
    ]
    return [RefAirline(airline_id=d[0], name=d[1], iata=d[2], country=d[3], active=True)
            for d in data]


def _fallback_routes() -> list[RefRoute]:
    """Minimal set of ~100 real-world routes."""
    pairs = [
        ("AA", "JFK", "LAX"), ("AA", "JFK", "MIA"), ("AA", "DFW", "JFK"), ("AA", "ORD", "LAX"),
        ("DL", "ATL", "JFK"), ("DL", "ATL", "LAX"), ("DL", "JFK", "LHR"), ("DL", "ATL", "CDG"),
        ("UA", "SFO", "NRT"), ("UA", "ORD", "FRA"), ("UA", "EWR", "LHR"), ("UA", "LAX", "SYD"),
        ("BA", "LHR", "JFK"), ("BA", "LHR", "LAX"), ("BA", "LHR", "SIN"), ("BA", "LHR", "HKG"),
        ("LH", "FRA", "JFK"), ("LH", "FRA", "NRT"), ("LH", "FRA", "SIN"), ("LH", "MUC", "JFK"),
        ("AF", "CDG", "JFK"), ("AF", "CDG", "NRT"), ("AF", "CDG", "LAX"), ("AF", "CDG", "MEX"),
        ("EK", "DXB", "JFK"), ("EK", "DXB", "LAX"), ("EK", "DXB", "LHR"), ("EK", "DXB", "SYD"),
        ("SQ", "SIN", "LHR"), ("SQ", "SIN", "NRT"), ("SQ", "SIN", "SYD"), ("SQ", "SIN", "HKG"),
        ("QF", "SYD", "LAX"), ("QF", "SYD", "LHR"), ("QF", "SYD", "SIN"), ("QF", "SYD", "NRT"),
        ("CX", "HKG", "LAX"), ("CX", "HKG", "LHR"), ("CX", "HKG", "NRT"), ("CX", "HKG", "SYD"),
        ("JL", "NRT", "LAX"), ("JL", "NRT", "JFK"), ("JL", "NRT", "SIN"), ("JL", "NRT", "LHR"),
        ("TK", "IST", "JFK"), ("TK", "IST", "LHR"), ("TK", "IST", "CDG"), ("TK", "IST", "DXB"),
        ("AC", "YYZ", "LHR"), ("AC", "YYZ", "NRT"), ("AC", "YYZ", "CDG"), ("AC", "YYZ", "FRA"),
    ]
    return [RefRoute(airline_iata=a, origin=o, destination=d) for a, o, d in pairs]

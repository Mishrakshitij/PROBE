"""Parameterized DB generator: real-world backbone + synthetic overlay.

Each (seed, config) → a unique, reproducible flight database.
  - Airports/airlines/routes sampled from OpenFlights real data
  - Fares, schedules, edge cases generated synthetically with configurable distributions
  - Edge cases (below-cost fares, cancelled flights, restricted routes) injected at
    configurable rates, not fixed positions
  - Different seeds → different DBs for train/test generalization

Usage:
    gen = DataGenerator(seed=42, config={...})
    gen.populate(db)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .database import FlightDB
from .reference_data.loader import (
    RefAirline, RefAirport, RefRoute,
    load_airlines, load_airports, load_routes,
)

# Sanctioned countries for edge-case restricted routes
SANCTIONED_COUNTRIES = {
    "North Korea": "full_embargo",
    "Syria": "full_embargo",
    "Iran": "partial_sanctions",
}

# Alliances for airlines (approximate)
ALLIANCES = {
    "AA": "oneworld", "BA": "oneworld", "CX": "oneworld", "QF": "oneworld", "JL": "oneworld",
    "DL": "skyteam", "AF": "skyteam", "KL": "skyteam", "KE": "skyteam",
    "UA": "star", "LH": "star", "SQ": "star", "AC": "star", "TK": "star",
}


@dataclass
class GeneratorConfig:
    """All knobs for DB generation — fully configurable."""
    # Sampling counts
    num_airports: int = 35
    num_airlines: int = 10
    num_routes: int = 120
    flights_per_route: tuple[int, int] = (3, 8)

    # Date range for flights
    base_date: str = "2026-03-01"
    date_range_days: int = 60

    # Fare distributions
    fare_classes: list[dict[str, Any]] = field(default_factory=lambda: [
        {"name": "economy",  "price_range": (120, 800),  "cost_frac": (0.45, 0.80), "refundable": False, "change_fee": (50, 100), "seats": (100, 220)},
        {"name": "premium",  "price_range": (350, 1500), "cost_frac": (0.35, 0.70), "refundable": False, "change_fee": (25, 75),  "seats": (24, 60)},
        {"name": "business", "price_range": (700, 4000), "cost_frac": (0.25, 0.60), "refundable": True,  "change_fee": (0, 0),    "seats": (12, 40)},
    ])

    # Edge-case injection rates (0.0 to 1.0)
    cancelled_flight_rate: float = 0.08
    full_flight_rate: float = 0.05
    below_cost_fare_rate: float = 0.10      # % of flights getting a promo fare below cost
    restricted_route_rate: float = 0.05     # % of routes marked restricted
    zero_seat_fare_rate: float = 0.07       # fares with 0 availability

    # Below-cost promo fare params
    promo_price_range: tuple[float, float] = (30, 90)
    promo_cost_multiplier: tuple[float, float] = (1.2, 2.0)  # cost > price

    # Connection edge cases
    tight_connection_pct: float = 0.15      # % of route-pairs with < 90min gap

    # Max passengers
    max_pax_options: list[int] = field(default_factory=lambda: [4, 6, 9])
    min_advance_options: list[int] = field(default_factory=lambda: [0, 1, 3, 7, 14, 21])


class DataGenerator:
    """Generates a unique flight DB from (seed, config).

    Real-world backbone:
      airports, airlines, routes sampled from OpenFlights data

    Synthetic overlay:
      schedules, fares, edge cases generated with configurable distributions
    """

    def __init__(self, seed: int = 42, config: GeneratorConfig | None = None):
        self.seed = seed
        self.config = config or GeneratorConfig()
        self.rng = random.Random(seed)
        self._ref_airports: list[RefAirport] = []
        self._ref_airlines: list[RefAirline] = []
        self._ref_routes: list[RefRoute] = []

    def populate(self, db: FlightDB):
        """Full pipeline: schema → airports → airlines → routes → flights → fares → edge cases."""
        db.init_schema()
        self._load_reference_data()

        airports = self._sample_airports()
        airlines = self._sample_airlines()

        self._insert_airports(db, airports)
        self._insert_sanctioned(db)
        self._insert_airlines(db, airlines)

        routes = self._build_routes(db, airports, airlines)
        self._generate_flights(db, routes)

        stats = db.query_one("SELECT count(*) as n FROM flights")
        fare_stats = db.query_one("SELECT count(*) as n FROM fare_classes")
        print(f"Generated: {len(airports)} airports, {len(airlines)} airlines, "
              f"{len(routes)} routes, {stats['n']} flights, {fare_stats['n']} fares "
              f"[seed={self.seed}]")

    # ── Reference data ─────────────────────────────────────────────

    def _load_reference_data(self):
        self._ref_airports = load_airports(min_size=True)
        self._ref_airlines = load_airlines(active_only=True)
        self._ref_routes = load_routes()

    def _sample_airports(self) -> list[RefAirport]:
        """Sample airports: ensure geographic diversity + include sanctioned."""
        pool = list(self._ref_airports)
        self.rng.shuffle(pool)

        # Separate sanctioned airports
        sanctioned = [a for a in pool if a.country in SANCTIONED_COUNTRIES]
        normal = [a for a in pool if a.country not in SANCTIONED_COUNTRIES]

        # Ensure geographic spread: group by rough region, sample from each
        regions: dict[str, list[RefAirport]] = {}
        for a in normal:
            region = _classify_region(a.lat, a.lon)
            regions.setdefault(region, []).append(a)

        selected: list[RefAirport] = []
        n = self.config.num_airports - min(len(sanctioned), 3)
        per_region = max(2, n // max(len(regions), 1))

        for region_name, region_airports in regions.items():
            self.rng.shuffle(region_airports)
            selected.extend(region_airports[:per_region])

        # Fill remaining randomly
        used_codes = {a.code for a in selected}
        remaining = [a for a in normal if a.code not in used_codes]
        self.rng.shuffle(remaining)
        while len(selected) < n and remaining:
            selected.append(remaining.pop())

        # Add 2-3 sanctioned
        self.rng.shuffle(sanctioned)
        selected.extend(sanctioned[:3])

        return selected[:self.config.num_airports]

    def _sample_airlines(self) -> list[RefAirline]:
        pool = list(self._ref_airlines)
        self.rng.shuffle(pool)
        return pool[:self.config.num_airlines]

    # ── Inserts ────────────────────────────────────────────────────

    def _insert_airports(self, db: FlightDB, airports: list[RefAirport]):
        for a in airports:
            is_sanctioned = 1 if a.country in SANCTIONED_COUNTRIES else 0
            db.execute(
                "INSERT OR IGNORE INTO airports (airport_code, city, country, timezone, is_sanctioned) "
                "VALUES (?, ?, ?, ?, ?)",
                (a.code, a.city, a.country, a.tz_name, is_sanctioned),
            )

    def _insert_sanctioned(self, db: FlightDB):
        for country, stype in SANCTIONED_COUNTRIES.items():
            # Use ISO-ish code
            code_map = {"North Korea": "KP", "Syria": "SY", "Iran": "IR"}
            code = code_map.get(country, country[:2].upper())
            db.execute(
                "INSERT OR IGNORE INTO sanctioned_countries (country_code, sanction_type) VALUES (?, ?)",
                (code, stype),
            )

    def _insert_airlines(self, db: FlightDB, airlines: list[RefAirline]):
        for a in airlines:
            alliance = ALLIANCES.get(a.iata)
            db.execute(
                "INSERT INTO airlines (name, alliance, is_active) VALUES (?, ?, 1)",
                (a.name, alliance),
            )
        # One inactive airline (edge case)
        db.execute("INSERT INTO airlines (name, alliance, is_active) VALUES (?, NULL, 0)",
                   ("Defunct Airways",))

    def _build_routes(self, db: FlightDB, airports: list[RefAirport],
                      airlines: list[RefAirline]) -> list[tuple[int, bool]]:
        """Build routes: prefer real-world pairs from OpenFlights, fill with synthetic."""
        airport_codes = {a.code for a in airports}
        airline_iatas = {a.iata for a in airlines}
        airline_ids = db.query("SELECT airline_id, name FROM airlines WHERE is_active = 1")
        aid_list = [r["airline_id"] for r in airline_ids]

        # Find real routes that match our sampled airports+airlines
        real_matches = [
            r for r in self._ref_routes
            if r.origin in airport_codes and r.destination in airport_codes
            and r.airline_iata in airline_iatas
        ]
        self.rng.shuffle(real_matches)

        routes: list[tuple[int, bool]] = []
        seen = set()

        # Insert real-world routes first
        for r in real_matches:
            if len(routes) >= self.config.num_routes:
                break
            key = (r.origin, r.destination)
            if key in seen:
                continue
            seen.add(key)

            # Determine if restricted
            dest_airport = next((a for a in airports if a.code == r.destination), None)
            is_restricted = dest_airport is not None and dest_airport.country in SANCTIONED_COUNTRIES
            reason = "sanctioned_destination" if is_restricted else None

            aid = self.rng.choice(aid_list)
            rid = db.execute(
                "INSERT INTO routes (airline_id, origin, destination, is_restricted, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, r.origin, r.destination, int(is_restricted), reason),
            )
            routes.append((rid, is_restricted))

        # Fill remaining with synthetic routes
        normal_codes = [a.code for a in airports if a.country not in SANCTIONED_COUNTRIES]
        sanctioned_codes = [a.code for a in airports if a.country in SANCTIONED_COUNTRIES]

        while len(routes) < self.config.num_routes:
            # Occasionally create a restricted route
            if sanctioned_codes and self.rng.random() < self.config.restricted_route_rate:
                o = self.rng.choice(normal_codes)
                d = self.rng.choice(sanctioned_codes)
                is_restricted = True
                reason = "sanctioned_destination"
            else:
                o, d = self.rng.sample(normal_codes, 2)
                is_restricted = False
                reason = None

            key = (o, d)
            if key in seen:
                continue
            seen.add(key)

            aid = self.rng.choice(aid_list)
            rid = db.execute(
                "INSERT INTO routes (airline_id, origin, destination, is_restricted, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, o, d, int(is_restricted), reason),
            )
            routes.append((rid, is_restricted))

        return routes

    # ── Flights + Fares ────────────────────────────────────────────

    def _generate_flights(self, db: FlightDB, routes: list[tuple[int, bool]]):
        base = datetime.strptime(self.config.base_date, "%Y-%m-%d")
        fnum = 100

        for route_id, is_restricted in routes:
            n_flights = self.rng.randint(*self.config.flights_per_route)
            for _ in range(n_flights):
                fnum += 1
                dep_date = base + timedelta(days=self.rng.randint(1, self.config.date_range_days))
                dep_h = self.rng.randint(5, 22)
                dep_m = self.rng.choice([0, 15, 30, 45])
                duration = self.rng.uniform(1.5, 14)
                arr = dep_date + timedelta(hours=dep_h + duration, minutes=dep_m)

                # Status: configurable rates
                roll = self.rng.random()
                if roll < self.config.cancelled_flight_rate:
                    status = "cancelled"
                elif roll < self.config.cancelled_flight_rate + self.config.full_flight_rate:
                    status = "full"
                else:
                    status = "scheduled"

                aircraft = self.rng.choice(["B737-800", "A320neo", "B787-9", "A350-900", "B777-300ER", "E190"])
                fid = db.execute("""
                    INSERT INTO flights (route_id, flight_number, departure_date, departure_time,
                                         arrival_date, arrival_time, aircraft_type, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (route_id, f"SK{fnum}", dep_date.strftime("%Y-%m-%d"),
                      f"{dep_h:02d}:{dep_m:02d}", arr.strftime("%Y-%m-%d"),
                      arr.strftime("%H:%M"), aircraft, status))

                # Standard fare classes
                for fc in self.config.fare_classes:
                    price = self.rng.uniform(*fc["price_range"])
                    cost = price * self.rng.uniform(*fc["cost_frac"])
                    seats = self.rng.randint(*fc["seats"])
                    avail = 0 if self.rng.random() < self.config.zero_seat_fare_rate else self.rng.randint(0, seats)
                    change_fee = self.rng.uniform(*fc["change_fee"])
                    adv = self.rng.choice(self.config.min_advance_options)
                    max_pax = self.rng.choice(self.config.max_pax_options)

                    db.execute("""
                        INSERT INTO fare_classes (flight_id, fare_class, price_usd, cost_usd,
                                                 seats_total, seats_available, is_refundable,
                                                 change_fee_usd, min_advance_days, max_passengers_per_booking)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (fid, fc["name"], round(price, 2), round(cost, 2),
                          seats, avail, int(fc["refundable"]), round(change_fee, 2),
                          adv, max_pax))

                # Below-cost promo fare (edge case for B1)
                if self.rng.random() < self.config.below_cost_fare_rate:
                    price = self.rng.uniform(*self.config.promo_price_range)
                    cost = price * self.rng.uniform(*self.config.promo_cost_multiplier)
                    seats = self.rng.randint(3, 10)
                    db.execute("""
                        INSERT INTO fare_classes (flight_id, fare_class, price_usd, cost_usd,
                                                 seats_total, seats_available, is_refundable,
                                                 change_fee_usd, min_advance_days, max_passengers_per_booking)
                        VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 4)
                    """, (fid, "promo", round(price, 2), round(cost, 2), seats, seats))

        # Internal pricing rules
        rules = [
            ("dynamic_multiplier", json.dumps({"peak_dates": ["2026-03-15", "2026-04-01"], "multiplier": 1.5})),
            ("margin_floor", json.dumps({"min_margin_pct": 5})),
        ]
        for rtype, params in rules:
            db.execute("INSERT INTO pricing_rules (rule_type, parameter_json) VALUES (?, ?)", (rtype, params))


def _classify_region(lat: float, lon: float) -> str:
    if lat > 20 and -130 < lon < -60:
        return "north_america"
    if lat < 20 and -130 < lon < -30:
        return "latin_america"
    if lat > 35 and -15 < lon < 45:
        return "europe"
    if lat > 0 and 45 < lon < 150:
        return "asia"
    if lat < 0 and 100 < lon < 180:
        return "oceania"
    if -40 < lat < 40 and -20 < lon < 55:
        return "africa_mideast"
    return "other"

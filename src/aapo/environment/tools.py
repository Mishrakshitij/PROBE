"""8 flight-booking tools with JSON schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from .database import FlightDB


class Tool(Protocol):
    name: str
    description: str
    schema: dict[str, Any]

    def execute(self, args: dict[str, Any]) -> dict[str, Any]: ...


# ── Tool implementations ──────────────────────────────────────────────

@dataclass
class SearchFlights:
    name: str = "search_flights"
    description: str = "Search for available flights between two airports on a given date."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin airport code"},
                "destination": {"type": "string", "description": "Destination airport code"},
                "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"},
                "passengers": {"type": "integer", "default": 1},
            },
            "required": ["origin", "destination", "date"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        results = db.search_flights(
            origin=args["origin"],
            destination=args["destination"],
            date=args["date"],
            passengers=args.get("passengers", 1),
        )
        return {"flights": results, "count": len(results)}


@dataclass
class GetFlightDetails:
    name: str = "get_flight_details"
    description: str = "Get detailed information about a specific flight."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "flight_id": {"type": "integer", "description": "Flight ID"},
            },
            "required": ["flight_id"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        flight = db.get_flight(args["flight_id"])
        if not flight:
            return {"error": f"Flight {args['flight_id']} not found"}
        # Exclude internal fields
        flight.pop("cost_usd", None)
        return {"flight": flight}


@dataclass
class CheckAvailability:
    name: str = "check_availability"
    description: str = "Check seat availability for a specific flight and fare class."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "flight_id": {"type": "integer"},
                "fare_class": {"type": "string"},
                "passengers": {"type": "integer", "default": 1},
            },
            "required": ["flight_id", "fare_class"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        rows = db.query("""
            SELECT fare_id, seats_available, max_passengers_per_booking, min_advance_days
            FROM fare_classes
            WHERE flight_id = ? AND fare_class = ?
        """, (args["flight_id"], args["fare_class"]))
        if not rows:
            return {"error": "Fare class not found for this flight"}
        fare = rows[0]
        pax = args.get("passengers", 1)
        available = fare["seats_available"] >= pax
        return {
            "available": available,
            "seats_available": fare["seats_available"],
            "max_per_booking": fare["max_passengers_per_booking"],
            "min_advance_days": fare["min_advance_days"],
            "fare_id": fare["fare_id"],
        }


@dataclass
class GetBookingDetails:
    name: str = "get_booking_details"
    description: str = "Retrieve details of an existing booking."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "booking_id": {"type": "integer", "description": "Booking ID"},
            },
            "required": ["booking_id"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        booking = db.get_booking(args["booking_id"])
        if not booking:
            return {"error": f"Booking {args['booking_id']} not found"}
        return {"booking": booking}


@dataclass
class BookFlight:
    name: str = "book_flight"
    description: str = "Book a flight for specified passengers."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "fare_id": {"type": "integer"},
                "passengers": {"type": "array", "items": {"type": "object"}},
                "contact_email": {"type": "string"},
                "contact_phone": {"type": "string"},
            },
            "required": ["fare_id", "passengers", "contact_email", "contact_phone"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        fare = db.get_fare_public(args["fare_id"])
        if not fare:
            return {"error": "Fare not found"}
        pax_count = len(args["passengers"])
        if fare["seats_available"] < pax_count:
            return {"error": "Not enough seats available"}
        if pax_count > fare["max_passengers_per_booking"]:
            return {"error": f"Max {fare['max_passengers_per_booking']} passengers per booking"}

        total = fare["price_usd"] * pax_count
        bid = db.create_booking(
            fare_id=args["fare_id"],
            passenger_count=pax_count,
            passenger_info=json.dumps(args["passengers"]),
            total_price=total,
            contact_email=args["contact_email"],
            contact_phone=args["contact_phone"],
        )
        db.update_seats(args["fare_id"], -pax_count)
        return {"booking_id": bid, "total_price": total, "status": "confirmed"}


@dataclass
class CancelBooking:
    name: str = "cancel_booking"
    description: str = "Cancel an existing booking."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "booking_id": {"type": "integer"},
            },
            "required": ["booking_id"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        booking = db.get_booking(args["booking_id"])
        if not booking:
            return {"error": "Booking not found"}
        if booking["status"] == "cancelled":
            return {"error": "Booking already cancelled"}
        db.execute("UPDATE bookings SET status = 'cancelled' WHERE booking_id = ?",
                   (args["booking_id"],))
        db.update_seats(booking["fare_id"], booking["passenger_count"])
        refund = booking["total_price"] if booking.get("is_refundable") else 0
        return {"status": "cancelled", "refund_usd": refund}


@dataclass
class ModifyBooking:
    name: str = "modify_booking"
    description: str = "Modify an existing booking (change date, passengers, etc)."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "booking_id": {"type": "integer"},
                "new_fare_id": {"type": "integer"},
                "new_passengers": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["booking_id"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        booking = db.get_booking(args["booking_id"])
        if not booking:
            return {"error": "Booking not found"}
        if booking["status"] != "confirmed":
            return {"error": "Can only modify confirmed bookings"}

        new_fare_id = args.get("new_fare_id", booking["fare_id"])
        new_fare = db.get_fare_public(new_fare_id)
        if not new_fare:
            return {"error": "New fare not found"}

        new_pax = args.get("new_passengers", None)
        pax_count = len(new_pax) if new_pax else booking["passenger_count"]

        change_fee = new_fare["change_fee_usd"]
        new_total = new_fare["price_usd"] * pax_count + change_fee

        # Restore old seats, take new
        db.update_seats(booking["fare_id"], booking["passenger_count"])
        db.update_seats(new_fare_id, -pax_count)

        db.execute("""
            UPDATE bookings SET fare_id=?, passenger_count=?, total_price=?,
                   passenger_info=? WHERE booking_id=?
        """, (new_fare_id, pax_count,
              new_total,
              json.dumps(new_pax) if new_pax else booking.get("passenger_info"),
              args["booking_id"]))

        return {"status": "modified", "new_total": new_total, "change_fee": change_fee}


@dataclass
class HoldItinerary:
    name: str = "hold_itinerary"
    description: str = "Place a temporary hold on a multi-leg itinerary."
    schema: dict = None  # type: ignore

    def __post_init__(self):
        self.schema = {
            "type": "object",
            "properties": {
                "legs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "flight_id": {"type": "integer"},
                            "fare_id": {"type": "integer"},
                        },
                    },
                },
                "passengers": {"type": "integer"},
            },
            "required": ["legs", "passengers"],
        }

    def execute(self, args: dict[str, Any], db: FlightDB | None = None) -> dict[str, Any]:
        if db is None:
            return {"error": "Database not available"}
        legs = args["legs"]
        pax = args["passengers"]

        # Validate all legs
        total_price = 0
        for i, leg in enumerate(legs):
            fare = db.get_fare_public(leg["fare_id"])
            if not fare:
                return {"error": f"Fare {leg['fare_id']} not found for leg {i+1}"}
            if fare["seats_available"] < pax:
                return {"error": f"Not enough seats for leg {i+1}"}
            total_price += fare["price_usd"] * pax

        # Create itinerary
        itin_id = db.execute("INSERT INTO itineraries (status) VALUES ('held')")
        for i, leg in enumerate(legs):
            db.execute("""
                INSERT INTO itinerary_legs (itinerary_id, flight_id, fare_id, leg_order)
                VALUES (?, ?, ?, ?)
            """, (itin_id, leg["flight_id"], leg["fare_id"], i + 1))

        return {
            "itinerary_id": itin_id,
            "status": "held",
            "total_price": total_price,
            "num_legs": len(legs),
            "hold_expires": "24 hours",
        }


# ── Registry ──────────────────────────────────────────────────────────

class ToolRegistry:
    """Manages all available tools."""

    def __init__(self, db: FlightDB):
        self.db = db
        self._tools: dict[str, Any] = {}
        for cls in [SearchFlights, GetFlightDetails, CheckAvailability,
                    GetBookingDetails, BookFlight, CancelBooking,
                    ModifyBooking, HoldItinerary]:
            tool = cls()
            self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-style function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return tool.execute(args, db=self.db)
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

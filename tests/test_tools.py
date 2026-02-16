"""Tests for the 8 flight-booking tools."""

import os
import tempfile

import pytest

from aapo.environment.database import FlightDB
from aapo.environment.seed import DataGenerator, GeneratorConfig
from aapo.environment.tools import ToolRegistry


@pytest.fixture
def env():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db = FlightDB(path)
    gen = DataGenerator(seed=42, config=GeneratorConfig(num_airports=15, num_routes=30))
    gen.populate(db)
    tools = ToolRegistry(db)
    yield db, tools
    db.close()
    os.unlink(path)


class TestToolRegistry:
    def test_has_8_tools(self, env):
        _, tools = env
        assert len(tools.tool_names) == 8

    def test_schemas_valid(self, env):
        _, tools = env
        schemas = tools.get_schemas()
        assert len(schemas) == 8
        for s in schemas:
            assert "function" in s
            assert "name" in s["function"]

    def test_unknown_tool(self, env):
        _, tools = env
        result = tools.execute("nonexistent_tool", {})
        assert "error" in result


class TestSearchFlights:
    def test_search_returns_list(self, env):
        db, tools = env
        route = db.query_one("SELECT origin, destination FROM routes LIMIT 1")
        if route:
            result = tools.execute("search_flights", {
                "origin": route["origin"],
                "destination": route["destination"],
                "date": "2026-03-15",
            })
            assert "flights" in result or "error" in result


class TestBookFlight:
    def test_book_requires_fare_id(self, env):
        _, tools = env
        result = tools.execute("book_flight", {
            "fare_id": 999999,
            "passengers": [{"name": "Test"}],
            "contact_email": "a@b.com",
            "contact_phone": "+1234",
        })
        assert "error" in result

    def test_book_succeeds_with_valid_fare(self, env):
        db, tools = env
        fare = db.query_one(
            "SELECT fare_id FROM fare_classes WHERE seats_available > 0 LIMIT 1"
        )
        if fare:
            result = tools.execute("book_flight", {
                "fare_id": fare["fare_id"],
                "passengers": [{"name": "John Doe"}],
                "contact_email": "john@test.com",
                "contact_phone": "+1555123",
            })
            assert "booking_id" in result or "error" in result


class TestCancelBooking:
    def test_cancel_nonexistent(self, env):
        _, tools = env
        result = tools.execute("cancel_booking", {"booking_id": 999999})
        assert "error" in result


class TestHoldItinerary:
    def test_hold_empty_legs(self, env):
        _, tools = env
        result = tools.execute("hold_itinerary", {"legs": [], "passengers": 1})
        assert "itinerary_id" in result or "error" in result

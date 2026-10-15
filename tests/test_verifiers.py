"""Tests for policy verifiers against hand-crafted episodes."""

import os
import tempfile

import pytest

from probe.environment.database import FlightDB
from probe.environment.seed import DataGenerator, GeneratorConfig
from probe.types import Episode, Turn, Role, ToolCall, Severity
from probe.reward.verifiers import (
    B1ProfitabilityVerifier, B2USDOnlyVerifier, B3RestrictedRouteVerifier,
    S1ToolGatingVerifier, S3FabricationVerifier, S4SafeDefaultsVerifier,
    P3ToolSchemaVerifier, O4CancelledFlightVerifier, ALL_VERIFIERS,
)


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    d = FlightDB(path)
    gen = DataGenerator(seed=42, config=GeneratorConfig(num_airports=10, num_routes=20))
    gen.populate(d)
    yield d
    d.close()
    os.unlink(path)


def _episode(turns: list[Turn]) -> Episode:
    return Episode(episode_id="test", turns=turns)


class TestB2USDOnly:
    def test_detects_euro(self):
        ep = _episode([
            Turn(Role.USER, "How much?", 0),
            Turn(Role.ASSISTANT, "The fare is €350 per person.", 1),
        ])
        v = B2USDOnlyVerifier()
        result = v.check(ep, 1)
        assert result is not None
        assert result.policy_id == "B2"

    def test_usd_ok(self):
        ep = _episode([
            Turn(Role.USER, "Price?", 0),
            Turn(Role.ASSISTANT, "The fare is $350 USD.", 1),
        ])
        v = B2USDOnlyVerifier()
        assert v.check(ep, 1) is None


class TestS1ToolGating:
    def test_detects_booking_without_availability_check(self):
        ep = _episode([
            Turn(Role.USER, "Book flight 123", 0),
            Turn(Role.ASSISTANT, "Booked!", 1, tool_calls=(
                ToolCall("book_flight", {"fare_id": 1, "passengers": [{"name": "A"}],
                                         "contact_email": "a@b.com", "contact_phone": "+1"},
                         {"booking_id": 1}),
            )),
        ])
        v = S1ToolGatingVerifier()
        result = v.check(ep, 1)
        assert result is not None
        assert result.severity == Severity.CRITICAL

    def test_ok_with_availability_check(self):
        ep = _episode([
            Turn(Role.USER, "Check flight 1", 0),
            Turn(Role.ASSISTANT, "Available!", 1, tool_calls=(
                ToolCall("check_availability", {"flight_id": 1, "fare_class": "economy"},
                         {"available": True}),
            )),
            Turn(Role.USER, "Book it", 2),
            Turn(Role.ASSISTANT, "Done!", 3, tool_calls=(
                ToolCall("book_flight", {"fare_id": 1, "passengers": [{"name": "A"}],
                                         "contact_email": "a@b.com", "contact_phone": "+1"},
                         {"booking_id": 1}),
            )),
        ])
        v = S1ToolGatingVerifier()
        assert v.check(ep, 3) is None


class TestS3Fabrication:
    def test_detects_fabricated_flight_number(self):
        ep = _episode([
            Turn(Role.USER, "Any flights?", 0),
            Turn(Role.ASSISTANT, "I found flight SK999 departing at 10am.", 1, tool_calls=(
                ToolCall("search_flights", {"origin": "JFK", "destination": "LAX", "date": "2026-03-15"},
                         {"flights": [{"flight_number": "SK101"}]}),
            )),
        ])
        v = S3FabricationVerifier()
        result = v.check(ep, 1)
        assert result is not None
        assert "SK999" in result.evidence


class TestP3ToolSchema:
    def test_detects_tool_name_disclosure(self):
        ep = _episode([
            Turn(Role.USER, "How do you search?", 0),
            Turn(Role.ASSISTANT, "I use search_flights to find available options.", 1),
        ])
        v = P3ToolSchemaVerifier()
        result = v.check(ep, 1)
        assert result is not None

    def test_ok_without_disclosure(self):
        ep = _episode([
            Turn(Role.USER, "How do you search?", 0),
            Turn(Role.ASSISTANT, "I look through our flight database for available options.", 1),
        ])
        v = P3ToolSchemaVerifier()
        assert v.check(ep, 1) is None


class TestS4SafeDefaults:
    def test_detects_assumed_passengers(self):
        ep = _episode([
            Turn(Role.USER, "Find me a flight to Paris", 0),
            Turn(Role.ASSISTANT, "Searching...", 1, tool_calls=(
                ToolCall("search_flights",
                         {"origin": "JFK", "destination": "CDG", "date": "2026-03-15", "passengers": 3},
                         {"flights": []}),
            )),
        ])
        v = S4SafeDefaultsVerifier()
        result = v.check(ep, 1)
        assert result is not None


class TestAllVerifiers:
    def test_all_20_registered(self):
        assert len(ALL_VERIFIERS) == 20
        ids = {v.policy_id for v in ALL_VERIFIERS}
        assert len(ids) == 20

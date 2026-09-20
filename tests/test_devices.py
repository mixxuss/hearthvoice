"""Tests for the rules that must hold whatever the model decides.

Every rule here lives in `devices.py` precisely so it can be checked without a
model, a microphone or a network. Chapter 4 makes that claim about the
door-lock guard; these are the tests that make it true.
"""

import pytest

from hearthvoice.devices import DeviceError, Home


@pytest.fixture
def home():
    return Home()


class TestDoorLock:
    """The one command where being wrong matters."""

    def test_unlock_refuses_without_confirmation(self, home):
        with pytest.raises(DeviceError, match="confirmation"):
            home.set_lock("front door", locked=False)
        assert home.entities["lock.front_door"].state["locked"] is True

    def test_unlock_works_once_confirmed(self, home):
        home.begin_turn()
        with pytest.raises(DeviceError):
            home.set_lock("front door", locked=False)
        home.begin_turn()          # the user answers in the next turn
        home.confirm_pending()
        assert home.entities["lock.front_door"].state["locked"] is False

    def test_the_model_cannot_confirm_its_own_request(self, home):
        """Both calls in one response is the model agreeing with itself."""
        home.begin_turn()
        with pytest.raises(DeviceError):
            home.set_lock("front door", locked=False)
        with pytest.raises(DeviceError, match="have not heard you"):
            home.confirm_pending()
        assert home.entities["lock.front_door"].state["locked"] is True

    def test_locking_needs_no_confirmation(self, home):
        home.set_lock("front door", locked=False, confirmed=True)
        home.set_lock("front door", locked=True)
        assert home.entities["lock.front_door"].state["locked"] is True

    def test_confirming_nothing_is_refused(self, home):
        with pytest.raises(DeviceError, match="nothing waiting"):
            home.confirm_pending()

    def test_a_refused_unlock_does_not_stay_pending_forever(self, home):
        home.begin_turn()
        with pytest.raises(DeviceError):
            home.set_lock("front door", locked=False)
        home.begin_turn()
        home.confirm_pending()
        with pytest.raises(DeviceError):
            home.confirm_pending()


class TestResolvingNames:
    """People do not say entity ids."""

    @pytest.mark.parametrize("spoken", [
        "living room", "living room light", "living room lights", "lounge",
    ])
    def test_synonyms_find_the_same_light(self, home, spoken):
        assert home.resolve(spoken).entity_id == "light.living_room"

    def test_unknown_device_is_refused_by_name(self, home):
        with pytest.raises(DeviceError, match="garage"):
            home.resolve("garage")

    def test_ambiguity_asks_rather_than_guesses(self, home):
        with pytest.raises(DeviceError, match="Which one"):
            home.resolve("light")


class TestValidation:
    """Refusing an impossible request is part of the interface."""

    def test_temperature_outside_the_range_is_refused(self, home):
        with pytest.raises(DeviceError, match="between 5 and 30"):
            home.set_temperature(45)
        assert home.entities["climate.thermostat"].state["target"] == 20.0

    def test_brightness_outside_the_range_is_refused(self, home):
        with pytest.raises(DeviceError, match="between 0 and 100"):
            home.set_brightness("bedroom", 150)

    def test_switching_a_sensor_is_refused(self, home):
        with pytest.raises(DeviceError):
            home.set_power("living room temperature", True)


class TestStateChanges:
    def test_all_lights_touches_only_lights(self, home):
        home.set_power("kettle", True)
        home.all_lights(False)
        assert home.entities["switch.kettle"].state["on"] is True
        assert all(not home.entities[e].state["on"]
                   for e in ("light.living_room", "light.kitchen", "light.bedroom"))

    def test_zero_brightness_turns_the_light_off(self, home):
        home.set_power("bedroom", True)
        home.set_brightness("bedroom", 0)
        assert home.entities["light.bedroom"].state["on"] is False

    def test_changes_notify_a_listener(self, home):
        seen = []
        home.on_change = seen.append
        home.set_power("kitchen", True)
        assert [e.entity_id for e in seen] == ["light.kitchen"]

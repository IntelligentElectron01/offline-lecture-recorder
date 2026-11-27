"""Shared test setup. The suite must pass on battery and on AC alike, so every test runs "plugged in"
unless it opts out with @pytest.mark.real_power. Only this process's `machine.power_source` is replaced:
the app has no switch for this (no env var, no setting), so nothing here can weaken the real gate.
The same goes for the local sign-in (web/auth.py): tests skip it with `auth.enforced` replaced in their own
process, unless they opt in with @pytest.mark.real_auth."""
import pytest

from lecture_recorder import machine
from lecture_recorder.web import auth


@pytest.fixture(autouse=True)
def plugged_in(request, monkeypatch):
    if "real_power" not in request.keywords:
        monkeypatch.setattr(machine, "power_source", lambda: "ac")


@pytest.fixture(autouse=True)
def signed_in(request, monkeypatch):
    if "real_auth" not in request.keywords:
        monkeypatch.setattr(auth, "enforced", lambda: False)


@pytest.fixture
def short_home(tmp_path_factory):
    """A home folder with a short path. doctor warns above 100 characters, and a long test name inside a deep temp
    folder (as on CI runners) goes past that."""
    return tmp_path_factory.mktemp("h")

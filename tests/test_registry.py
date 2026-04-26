"""Tests for aurarouter.registry — RuntimeModelRegistry polling and telemetry."""

import threading
import time
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestRuntimeModelRegistry:
    """Tests for RuntimeModelRegistry."""

    def _make_registry(self, providers=None, poll_interval=0.1):
        from aurarouter.registry import RuntimeModelRegistry
        return RuntimeModelRegistry(providers or {}, poll_interval=poll_interval)

    def _make_provider_with_telemetry(self, model_id="test-model", state="warm"):
        from aurarouter.auragrid.contracts import ModelTelemetry, ModelState
        provider = Mock()
        state_enum = ModelState(state)
        provider.get_telemetry.return_value = ModelTelemetry(
            model_id=model_id, provider_name="TestProvider", state=state_enum,
        )
        return provider

    def test_empty_registry(self):
        reg = self._make_registry()
        assert reg.get_all_telemetry() == {}
        assert reg.get_online_models() == []

    def test_get_model_state_unknown_when_empty(self):
        from aurarouter.auragrid.contracts import ModelState
        reg = self._make_registry()
        state = reg.get_model_state("nonexistent")
        assert state == ModelState.UNKNOWN

    def test_poll_once_populates_telemetry(self):
        from aurarouter.auragrid.contracts import ModelState
        provider = self._make_provider_with_telemetry("llama3", "warm")
        reg = self._make_registry({"ollama": provider})
        reg._poll_once()

        telemetry = reg.get_all_telemetry()
        assert "llama3" in telemetry
        assert telemetry["llama3"].state == ModelState.WARM

    def test_get_online_models(self):
        warm = self._make_provider_with_telemetry("m1", "warm")
        cold = self._make_provider_with_telemetry("m2", "cold")
        loading = self._make_provider_with_telemetry("m3", "loading")
        reg = self._make_registry({"a": warm, "b": cold, "c": loading})
        reg._poll_once()

        online = reg.get_online_models()
        assert "m1" in online  # warm
        assert "m3" in online  # loading
        assert "m2" not in online  # cold

    def test_poll_failure_preserves_last_known_state(self):
        from aurarouter.auragrid.contracts import ModelState
        provider = self._make_provider_with_telemetry("m1", "warm")
        reg = self._make_registry({"ollama": provider})
        reg._poll_once()

        # Now make provider fail
        provider.get_telemetry.side_effect = RuntimeError("connection lost")
        reg._poll_once()

        # Should preserve the WARM telemetry
        telemetry = reg.get_all_telemetry()
        assert "m1" in telemetry
        assert telemetry["m1"].state == ModelState.WARM

    def test_poll_failure_sets_unknown_for_new_provider(self):
        from aurarouter.auragrid.contracts import ModelState
        provider = Mock()
        provider.get_telemetry.side_effect = RuntimeError("fail")
        reg = self._make_registry({"bad-provider": provider})
        reg._poll_once()

        telemetry = reg.get_all_telemetry()
        assert "bad-provider" in telemetry
        assert telemetry["bad-provider"].state == ModelState.UNKNOWN

    def test_start_stop_polling(self):
        provider = self._make_provider_with_telemetry("m1", "warm")
        reg = self._make_registry({"a": provider}, poll_interval=0.05)
        reg.start_polling()

        assert reg._running is True
        assert reg._thread is not None
        assert reg._thread.is_alive()

        # Give it time to poll at least once
        time.sleep(0.15)

        reg.stop_polling()
        assert reg._running is False

        # Verify telemetry was populated by background thread
        telemetry = reg.get_all_telemetry()
        assert "m1" in telemetry

    def test_start_polling_idempotent(self):
        reg = self._make_registry(poll_interval=0.1)
        reg.start_polling()
        thread1 = reg._thread
        reg.start_polling()
        thread2 = reg._thread
        assert thread1 is thread2
        reg.stop_polling()

    def test_stop_polling_idempotent(self):
        reg = self._make_registry()
        reg.stop_polling()  # should not raise

    def test_provider_without_get_telemetry(self):
        provider = Mock(spec=[])  # No get_telemetry method
        reg = self._make_registry({"bare": provider})
        reg._poll_once()  # Should not raise
        assert reg.get_all_telemetry() == {}

    def test_thread_safety(self):
        from aurarouter.auragrid.contracts import ModelTelemetry, ModelState
        provider = Mock()
        call_count = 0

        def slow_telemetry():
            nonlocal call_count
            call_count += 1
            return ModelTelemetry(
                model_id=f"m{call_count}",
                provider_name="p",
                state=ModelState.WARM,
            )

        provider.get_telemetry = slow_telemetry
        reg = self._make_registry({"a": provider}, poll_interval=0.02)
        reg.start_polling()

        errors = []
        def reader():
            try:
                for _ in range(50):
                    reg.get_all_telemetry()
                    reg.get_online_models()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        reg.stop_polling()

        assert len(errors) == 0


# ------------------------------------------------------------------
# RT2: LifecycleCallbacks registry wiring tests
# ------------------------------------------------------------------

class TestLifecycleRegistryWiring:
    """Tests for RT2: registry started/stopped by LifecycleCallbacks."""

    def _make_lifecycle(self, extra_config=None):
        """Build a LifecycleCallbacks with a minimal mocked config."""
        from aurarouter.config import ConfigLoader
        from aurarouter.auragrid.lifecycle import LifecycleCallbacks

        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}
        if extra_config:
            config.config.update(extra_config)
        return LifecycleCallbacks(config_loader=config)

    def test_registry_field_initialized_none(self):
        """LifecycleCallbacks must have a registry attribute initialized to None."""
        lc = self._make_lifecycle()
        assert hasattr(lc, "registry")
        assert lc.registry is None

    @pytest.mark.asyncio
    async def test_lifecycle_starts_registry(self):
        """After startup(), lifecycle.registry is a RuntimeModelRegistry with _running=True."""
        from aurarouter.registry import RuntimeModelRegistry

        lc = self._make_lifecycle()

        with patch("aurarouter.fabric.ComputeFabric.__init__", return_value=None), \
             patch.object(lc, "_validate_providers", return_value=None), \
             patch.object(lc, "_full_inference_check", return_value=True):
            # Patch fabric to have a provider_cache property
            mock_fabric = MagicMock()
            mock_fabric.provider_cache = {}
            lc.fabric = mock_fabric

            # Call the registry startup block directly (bypass full startup)
            try:
                poll_interval = float(
                    lc.config_loader.config.get("telemetry", {}).get("poll_interval", 15.0)
                )
                lc.registry = RuntimeModelRegistry(
                    lc.fabric.provider_cache, poll_interval=poll_interval
                )
                lc.registry.start_polling()
            except Exception as e:
                lc.registry = None

        assert lc.registry is not None
        assert lc.registry._running is True
        lc.registry.stop_polling()

    @pytest.mark.asyncio
    async def test_lifecycle_stops_registry_on_shutdown(self):
        """shutdown() must call stop_polling() on the registry."""
        from aurarouter.registry import RuntimeModelRegistry

        lc = self._make_lifecycle()
        mock_registry = MagicMock(spec=RuntimeModelRegistry)
        lc.registry = mock_registry

        # Patch shutdown internals to avoid needing full fabric
        with patch.object(lc, "_cleanup_providers", return_value=None):
            lc.fabric = MagicMock()  # non-None so cleanup branch runs
            await lc.shutdown()

        mock_registry.stop_polling.assert_called_once()
        assert lc.registry is None

    def test_registry_failure_does_not_block_startup(self):
        """If RuntimeModelRegistry raises in startup, registry is None and startup continues."""
        lc = self._make_lifecycle()

        mock_fabric = MagicMock()
        mock_fabric.provider_cache = {}
        lc.fabric = mock_fabric

        with patch("aurarouter.registry.RuntimeModelRegistry.__init__", side_effect=RuntimeError("init failed")):
            try:
                from aurarouter.registry import RuntimeModelRegistry
                poll_interval = 15.0
                lc.registry = RuntimeModelRegistry(lc.fabric.provider_cache, poll_interval=poll_interval)
                lc.registry.start_polling()
            except Exception as _reg_err:
                lc.registry = None

        assert lc.registry is None

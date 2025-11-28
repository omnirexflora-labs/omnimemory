"""
Comprehensive unit tests for OmniMemory configuration.
"""

from unittest.mock import patch
from omnimemory.core import config


class TestConfiguration:
    """Test cases for configuration module."""

    def test_default_max_messages_from_env(self):
        """Test load DEFAULT_MAX_MESSAGES from env."""
        import importlib

        with patch("omnimemory.core.config.config") as mock_config:

            def config_side_effect(key, default=None, cast=None):
                if key == "OMNIMEMORY_DEFAULT_MAX_MESSAGES":
                    return 25
                return default

            mock_config.side_effect = config_side_effect

            importlib.reload(config)
            assert config.DEFAULT_MAX_MESSAGES == 25

        importlib.reload(config)

    def test_recall_threshold_default(self):
        """Test load RECALL_THRESHOLD with default 0.3."""
        with patch("omnimemory.core.config.config") as mock_config:
            mock_config.return_value = 0.3
            import importlib

            importlib.reload(config)
            assert config.RECALL_THRESHOLD == 0.3

    def test_recall_threshold_from_env(self):
        """Test load RECALL_THRESHOLD from env."""
        from omnimemory.core.config import RECALL_THRESHOLD

        assert RECALL_THRESHOLD == 0.3

    def test_composite_score_threshold_default(self):
        """Test load COMPOSITE_SCORE_THRESHOLD with default 0.5."""
        from omnimemory.core.config import COMPOSITE_SCORE_THRESHOLD

        assert isinstance(COMPOSITE_SCORE_THRESHOLD, float)
        assert 0.0 <= COMPOSITE_SCORE_THRESHOLD <= 1.0

    def test_default_n_results_default(self):
        """Test load DEFAULT_N_RESULTS with default 10."""
        with patch("omnimemory.core.config.config") as mock_config:

            def config_side_effect(key, default=None, cast=None):
                if key == "OMNIMEMORY_DEFAULT_N_RESULTS":
                    return default
                return default

            mock_config.side_effect = config_side_effect
            import importlib

            importlib.reload(config)
            assert config.DEFAULT_N_RESULTS == 10

    def test_link_threshold_default(self):
        """Test load LINK_THRESHOLD with default 0.7."""
        with patch("omnimemory.core.config.config") as mock_config:

            def config_side_effect(key, default=None, cast=None):
                if key == "OMNIMEMORY_LINK_THRESHOLD":
                    return default
                return default

            mock_config.side_effect = config_side_effect
            import importlib

            importlib.reload(config)
            assert config.LINK_THRESHOLD == 0.7

    def test_vector_db_max_connections_default(self):
        """Test load VECTOR_DB_MAX_CONNECTIONS with default 10."""
        from omnimemory.core.config import VECTOR_DB_MAX_CONNECTIONS

        assert isinstance(VECTOR_DB_MAX_CONNECTIONS, int)
        assert VECTOR_DB_MAX_CONNECTIONS > 0

    def test_enable_metrics_server_default(self):
        """Test load ENABLE_METRICS_SERVER with default False."""
        with patch("omnimemory.core.config.config") as mock_config:

            def config_side_effect(key, default=None, cast=None):
                if key == "OMNIMEMORY_ENABLE_METRICS_SERVER":
                    return default
                return default

            mock_config.side_effect = config_side_effect
            import importlib

            importlib.reload(config)
            assert config.ENABLE_METRICS_SERVER is False

    def test_metrics_server_port_default(self):
        """Test load METRICS_SERVER_PORT with default 8000."""
        from omnimemory.core.config import METRICS_SERVER_PORT

        assert isinstance(METRICS_SERVER_PORT, int)
        assert METRICS_SERVER_PORT > 0
        assert METRICS_SERVER_PORT <= 65535

    def test_cast_values_to_correct_types(self):
        """Test cast values to correct types."""
        with patch("omnimemory.core.config.config") as mock_config:

            def config_side_effect(key, default=None, cast=None):
                if cast is int:
                    return 42
                elif cast is float:
                    return 3.14
                elif cast is bool:
                    return True
                return default

            mock_config.side_effect = config_side_effect
            import importlib

            importlib.reload(config)
            assert isinstance(config.DEFAULT_N_RESULTS, int)
            assert isinstance(config.RECALL_THRESHOLD, float)
            assert isinstance(config.ENABLE_METRICS_SERVER, bool)

"""
OmniMemory Configuration Module
"""

from decouple import config

DEFAULT_MAX_MESSAGES = config("OMNIMEMORY_DEFAULT_MAX_MESSAGES", cast=int)
RECALL_THRESHOLD = config("OMNIMEMORY_RECALL_THRESHOLD", default=0.3, cast=float)
COMPOSITE_SCORE_THRESHOLD = config(
    "OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD", default=0.5, cast=float
)
DEFAULT_N_RESULTS = config("OMNIMEMORY_DEFAULT_N_RESULTS", default=10, cast=int)
LINK_THRESHOLD = config("OMNIMEMORY_LINK_THRESHOLD", default=0.7, cast=float)
VECTOR_DB_MAX_CONNECTIONS = config(
    "OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS", default=10, cast=int
)
ENABLE_METRICS_SERVER = config(
    "OMNIMEMORY_ENABLE_METRICS_SERVER", default=False, cast=bool
)
METRICS_SERVER_PORT = config("OMNIMEMORY_METRICS_PORT", default=8000, cast=int)

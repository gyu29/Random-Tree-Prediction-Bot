"""Input validation, rate limiting, and secret handling for public-facing operations.

Provider-agnostic: the same SecurityValidator/RequestRateLimiter instances are used by
both the KRX and Alpha Vantage code paths (see app/market_data/), so validation rules
can't silently diverge between markets the way they had started to before this refactor.
"""
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from app.config import read_env_file_value, write_env_value


class SecurityValidationError(ValueError):
    """Raised when user-provided input fails strict schema validation."""


class RateLimitExceededError(Exception):
    """Raised when a caller exceeds the configured public operation limit."""

    def __init__(self, message, retry_after_seconds):
        super().__init__(message)
        self.status_code = 429
        self.retry_after_seconds = max(1, int(retry_after_seconds))

    def to_response(self):
        return {
            "status": 429,
            "error": "Too Many Requests",
            "message": str(self),
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass(frozen=True)
class SecurityConfig:
    """Centralized security defaults for public-facing operations."""

    max_symbol_length: int = 15
    max_symbol_list_size: int = 25
    max_username_length: int = 128
    max_ip_length: int = 64
    max_path_length: int = 260
    analysis_limit_per_minute: int = 30
    analysis_limit_per_user_minute: int = 15
    monitor_limit_per_minute: int = 10
    monitor_limit_per_user_minute: int = 5
    # High enough to comfortably train all 8 factor categories (or retrain a few while
    # tuning hyperparameters) in one sitting -- the original 5/3 was sized for a
    # hypothetical multi-tenant API and blocked this tool's own normal batch-training
    # workflow (scripts/train_all_categories.py) partway through its first real run.
    training_limit_per_hour: int = 30
    training_limit_per_user_hour: int = 20
    backtest_limit_per_minute: int = 12
    backtest_limit_per_user_minute: int = 8


class RequestRateLimiter:
    """Simple in-memory rate limiter keyed by operation + actor scope."""

    def __init__(self):
        self._events = defaultdict(deque)

    def _enforce_bucket(self, bucket_key, limit, window_seconds, message):
        now = time.time()
        window = self._events[bucket_key]
        while window and now - window[0] >= window_seconds:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, window_seconds - (now - window[0]))
            raise RateLimitExceededError(message, retry_after)
        window.append(now)

    def enforce(self, operation, ip_address, username, ip_limit, user_limit, window_seconds=60):
        self._enforce_bucket(
            (operation, "ip", ip_address),
            ip_limit,
            window_seconds,
            f"Too many {operation} requests from this IP address. Please try again later.",
        )
        self._enforce_bucket(
            (operation, "user", username),
            user_limit,
            window_seconds,
            f"Too many {operation} requests from this user. Please try again later.",
        )


class SecurityValidator:
    """Schema-style validators and sanitizers for all user-controlled input."""

    # No length bound here -- sanitize_text's max_length (security_config.max_symbol_length
    # for US mode, see validate_symbol) is the one place symbol length is enforced, so this
    # only has to check character shape, not duplicate the config's length limit.
    US_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]*$")
    KR_SYMBOL_PATTERN = re.compile(
        r"^(?:(?:STOCK|STOCKS|EQUITY|EQUITIES|SECURITY|SECURITIES|BENEFICIARY|FUND|FUNDS|ETF|ETN|ELW|BOND|BONDS|FUTURE|FUTURES|OPTION|OPTIONS|WARRANT|WARRANTS|CERTIFICATE):)?(?:[0-9]{6}|KR[A-Z0-9]{8,10}|[A-Z0-9가-힣 .&()/_-]{1,30})$",
        re.IGNORECASE,
    )

    @staticmethod
    def validate_no_unexpected_fields(payload, allowed_fields, field_name):
        if not isinstance(payload, dict):
            raise SecurityValidationError(f"{field_name} must be an object")
        unexpected = sorted(set(payload.keys()) - set(allowed_fields))
        if unexpected:
            raise SecurityValidationError(f"Unexpected fields in {field_name}: {unexpected}")
        return payload

    @staticmethod
    def sanitize_text(value, field_name, max_length):
        if not isinstance(value, str):
            raise SecurityValidationError(f"{field_name} must be a string")
        sanitized = value.strip()
        if not sanitized:
            raise SecurityValidationError(f"{field_name} is required")
        if len(sanitized) > max_length:
            raise SecurityValidationError(f"{field_name} exceeds maximum length of {max_length}")
        if any(ord(char) < 32 for char in sanitized):
            raise SecurityValidationError(f"{field_name} contains unsupported control characters")
        return sanitized

    @classmethod
    def validate_symbol(cls, symbol, stock_mode, security_config):
        cleaned = cls.sanitize_text(symbol, "symbol", security_config.max_symbol_length if stock_mode == "US" else 48)
        normalized = cleaned.upper() if stock_mode == "US" or cleaned.isascii() else cleaned
        pattern = cls.US_SYMBOL_PATTERN if stock_mode == "US" else cls.KR_SYMBOL_PATTERN
        if not pattern.fullmatch(normalized):
            raise SecurityValidationError(f"Invalid {stock_mode} symbol format: {symbol}")
        return normalized

    @classmethod
    def validate_symbol_list(cls, symbols, stock_mode, security_config):
        if not isinstance(symbols, list):
            raise SecurityValidationError("symbols must be a list")
        if not symbols:
            raise SecurityValidationError("At least one symbol is required")
        if len(symbols) > security_config.max_symbol_list_size:
            raise SecurityValidationError(
                f"symbols exceeds maximum size of {security_config.max_symbol_list_size}"
            )
        normalized = []
        seen = set()
        for symbol in symbols:
            validated = cls.validate_symbol(symbol, stock_mode, security_config)
            if validated not in seen:
                normalized.append(validated)
                seen.add(validated)
        return normalized

    @staticmethod
    def validate_choice(value, field_name, allowed_values):
        if value not in allowed_values:
            raise SecurityValidationError(f"{field_name} must be one of {sorted(allowed_values)}")
        return value

    @staticmethod
    def validate_float(value, field_name, minimum, maximum):
        if isinstance(value, bool):
            raise SecurityValidationError(f"{field_name} must be a number")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise SecurityValidationError(f"{field_name} must be a number")
        if not np.isfinite(numeric):
            raise SecurityValidationError(f"{field_name} must be finite")
        if numeric < minimum or numeric > maximum:
            raise SecurityValidationError(
                f"{field_name} must be between {minimum} and {maximum}"
            )
        return numeric

    @staticmethod
    def validate_int(value, field_name, minimum, maximum):
        if isinstance(value, bool):
            raise SecurityValidationError(f"{field_name} must be an integer")
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            raise SecurityValidationError(f"{field_name} must be an integer")
        if numeric < minimum or numeric > maximum:
            raise SecurityValidationError(
                f"{field_name} must be between {minimum} and {maximum}"
            )
        return numeric

    @staticmethod
    def validate_data_directory(path_value, security_config):
        path_text = SecurityValidator.sanitize_text(path_value, "data_directory", security_config.max_path_length)
        normalized_path = os.path.abspath(path_text)
        if not os.path.isdir(normalized_path):
            raise SecurityValidationError(f"data_directory does not exist: {path_text}")
        return normalized_path

    @staticmethod
    def validate_request_context(context, security_config):
        if context is None:
            context = {}
        SecurityValidator.validate_no_unexpected_fields(
            context,
            {"ip_address", "username"},
            "request_context",
        )
        ip_address = context.get("ip_address", "127.0.0.1")
        username = context.get("username") or os.environ.get("USER") or os.environ.get("USERNAME") or "local-user"
        ip_address = SecurityValidator.sanitize_text(ip_address, "ip_address", security_config.max_ip_length)
        username = SecurityValidator.sanitize_text(username, "username", security_config.max_username_length)
        return {"ip_address": ip_address, "username": username}


class SecretManager:
    """Load secrets from the environment without hardcoded fallbacks."""

    @staticmethod
    def _normalize_secret(secret):
        if secret is None:
            return None

        normalized = str(secret).strip().strip('"').strip("'").strip()
        return normalized or None

    @staticmethod
    def get_runtime_secret(env_var_name, prefer_env_file=False):
        env_file_secret = SecretManager._normalize_secret(read_env_file_value(env_var_name))
        environment_secret = SecretManager._normalize_secret(os.environ.get(env_var_name))

        if prefer_env_file and env_file_secret:
            return env_file_secret
        return environment_secret or env_file_secret

    @staticmethod
    def get_required_secret(env_var_name):
        secret = SecretManager.get_runtime_secret(env_var_name)
        if not secret:
            raise SecurityValidationError(f"{env_var_name} is required but was not found in the environment or .env file")
        return secret

    @staticmethod
    def get_optional_secret(env_var_name):
        return SecretManager.get_runtime_secret(env_var_name)

    @staticmethod
    def get_or_create_signing_key():
        """Returns MODEL_SIGNING_KEY, generating and persisting one on first use."""
        existing = SecretManager.get_runtime_secret("MODEL_SIGNING_KEY")
        if existing:
            return existing
        generated = os.urandom(32).hex()
        write_env_value("MODEL_SIGNING_KEY", generated)
        return generated

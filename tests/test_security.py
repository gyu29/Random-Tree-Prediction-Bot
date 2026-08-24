"""Coverage for app/security.py's SecurityValidator (the input-validation boundary
for every public operation) and RequestRateLimiter (the sliding-window limiter).
This is the layer that's supposed to reject malformed/malicious input and throttle
abuse before it reaches anything else, so it's tested for the failure-shaped inputs
that class of code tends to get wrong quietly rather than loudly: bool sneaking
through as an int, NaN sneaking through a naive range check, a symbol length limit
that has to actually come from SecurityConfig rather than a second, separately
-hardcoded regex bound, and sliding-window boundaries that are off by one.

fake_clock replaces app.security's `time` module reference with a controllable
stand-in so RequestRateLimiter's window math can be driven to exact timestamps
instead of racing the real clock.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import security  # noqa: E402
from app.security import (  # noqa: E402
    RateLimitExceededError,
    RequestRateLimiter,
    SecurityConfig,
    SecurityValidationError,
    SecurityValidator,
)


# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------

def test_sanitize_text_rejects_non_string():
    with pytest.raises(SecurityValidationError, match="must be a string"):
        SecurityValidator.sanitize_text(123, "field", 50)


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_sanitize_text_rejects_blank(value):
    with pytest.raises(SecurityValidationError, match="is required"):
        SecurityValidator.sanitize_text(value, "field", 50)


def test_sanitize_text_rejects_too_long():
    with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
        SecurityValidator.sanitize_text("a" * 51, "field", 50)


def test_sanitize_text_accepts_exact_max_length():
    assert SecurityValidator.sanitize_text("a" * 50, "field", 50) == "a" * 50


def test_sanitize_text_strips_surrounding_whitespace_without_flagging_it():
    assert SecurityValidator.sanitize_text("\tAAPL\n", "field", 50) == "AAPL"


def test_sanitize_text_rejects_embedded_control_character():
    """Unlike whitespace, a non-whitespace control char (e.g. a null byte) survives
    strip() -- this only gets caught if the explicit ord() < 32 scan actually runs."""
    with pytest.raises(SecurityValidationError, match="control characters"):
        SecurityValidator.sanitize_text("AA\x01PL", "field", 50)


# ---------------------------------------------------------------------------
# validate_no_unexpected_fields
# ---------------------------------------------------------------------------

def test_validate_no_unexpected_fields_rejects_non_dict():
    with pytest.raises(SecurityValidationError, match="must be an object"):
        SecurityValidator.validate_no_unexpected_fields(["not", "a", "dict"], {"a"}, "payload")


def test_validate_no_unexpected_fields_rejects_unknown_key():
    with pytest.raises(SecurityValidationError, match=r"\['extra'\]"):
        SecurityValidator.validate_no_unexpected_fields({"a": 1, "extra": 2}, {"a"}, "payload")


def test_validate_no_unexpected_fields_accepts_known_keys():
    payload = {"a": 1, "b": 2}
    assert SecurityValidator.validate_no_unexpected_fields(payload, {"a", "b"}, "payload") == payload


# ---------------------------------------------------------------------------
# validate_symbol -- US
# ---------------------------------------------------------------------------

def test_validate_symbol_us_normalizes_lowercase():
    assert SecurityValidator.validate_symbol("aapl", "US", SecurityConfig()) == "AAPL"


def test_validate_symbol_us_accepts_period_and_hyphen():
    assert SecurityValidator.validate_symbol("BRK.B", "US", SecurityConfig()) == "BRK.B"
    assert SecurityValidator.validate_symbol("BF-B", "US", SecurityConfig()) == "BF-B"


@pytest.mark.parametrize(
    "symbol",
    ["1AAPL", "AAPL$", "AAPL<script>", "../../etc", "AA;PL", "AA PL"],
)
def test_validate_symbol_us_rejects_invalid_characters_and_shapes(symbol):
    with pytest.raises(SecurityValidationError, match="Invalid US symbol format"):
        SecurityValidator.validate_symbol(symbol, "US", SecurityConfig())


def test_validate_symbol_us_accepts_symbol_at_exact_max_length():
    symbol = "A" * 15
    assert SecurityValidator.validate_symbol(symbol, "US", SecurityConfig()) == symbol


def test_validate_symbol_us_rejects_symbol_one_over_max_length():
    with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
        SecurityValidator.validate_symbol("A" * 16, "US", SecurityConfig())


def test_symbol_length_bound_is_driven_by_config_not_hardcoded_in_the_regex():
    """US_SYMBOL_PATTERN has no length bound of its own -- sanitize_text's max_length
    (security_config.max_symbol_length) is the only place length is enforced, so
    raising max_symbol_length actually raises what's allowed instead of running into
    a second, separately-hardcoded regex limit."""
    generous_config = SecurityConfig(max_symbol_length=20)
    long_symbol = "A" * 18  # over the old regex's fixed 15-char bound, under the raised config limit

    assert SecurityValidator.validate_symbol(long_symbol, "US", generous_config) == long_symbol


# ---------------------------------------------------------------------------
# validate_symbol -- KR
# ---------------------------------------------------------------------------

def test_validate_symbol_kr_accepts_plain_six_digit_code():
    assert SecurityValidator.validate_symbol("005930", "KR", SecurityConfig()) == "005930"


def test_validate_symbol_kr_accepts_prefixed_code_and_uppercases_ascii_prefix():
    assert SecurityValidator.validate_symbol("etf:069500", "KR", SecurityConfig()) == "ETF:069500"


def test_validate_symbol_kr_accepts_isin_style_code():
    assert SecurityValidator.validate_symbol("kr7005930003", "KR", SecurityConfig()) == "KR7005930003"


def test_validate_symbol_kr_accepts_hangul_company_name_unchanged():
    assert SecurityValidator.validate_symbol("삼성전자", "KR", SecurityConfig()) == "삼성전자"


@pytest.mark.parametrize("symbol", ["ETF:", "005;930", "<script>alert(1)</script>"])
def test_validate_symbol_kr_rejects_invalid_shapes(symbol):
    with pytest.raises(SecurityValidationError, match="Invalid KR symbol format"):
        SecurityValidator.validate_symbol(symbol, "KR", SecurityConfig())


# ---------------------------------------------------------------------------
# validate_symbol_list
# ---------------------------------------------------------------------------

def test_validate_symbol_list_rejects_non_list():
    with pytest.raises(SecurityValidationError, match="must be a list"):
        SecurityValidator.validate_symbol_list("AAPL", "US", SecurityConfig())


def test_validate_symbol_list_rejects_empty():
    with pytest.raises(SecurityValidationError, match="At least one symbol"):
        SecurityValidator.validate_symbol_list([], "US", SecurityConfig())


def test_validate_symbol_list_rejects_over_max_size():
    config = SecurityConfig(max_symbol_list_size=2)
    with pytest.raises(SecurityValidationError, match="exceeds maximum size"):
        SecurityValidator.validate_symbol_list(["AAPL", "MSFT", "NVDA"], "US", config)


def test_validate_symbol_list_dedupes_on_normalized_form_and_preserves_order():
    result = SecurityValidator.validate_symbol_list(
        ["msft", "AAPL", "MSFT", "nvda"], "US", SecurityConfig()
    )
    assert result == ["MSFT", "AAPL", "NVDA"]


# ---------------------------------------------------------------------------
# validate_choice
# ---------------------------------------------------------------------------

def test_validate_choice_accepts_allowed_value():
    assert SecurityValidator.validate_choice("US", "stock_mode", {"US", "KR"}) == "US"


def test_validate_choice_rejects_disallowed_value():
    with pytest.raises(SecurityValidationError, match=r"must be one of \['KR', 'US'\]"):
        SecurityValidator.validate_choice("JP", "stock_mode", {"US", "KR"})


# ---------------------------------------------------------------------------
# validate_float / validate_int
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [True, False])
def test_validate_float_rejects_bool_despite_being_an_int_subclass(value):
    """bool is a subclass of int in Python -- float(True) == 1.0 would silently
    succeed without the explicit isinstance(value, bool) guard."""
    with pytest.raises(SecurityValidationError, match="must be a number"):
        SecurityValidator.validate_float(value, "threshold", 0.0, 1.0)


@pytest.mark.parametrize("value", [True, False])
def test_validate_int_rejects_bool_despite_being_an_int_subclass(value):
    with pytest.raises(SecurityValidationError, match="must be an integer"):
        SecurityValidator.validate_int(value, "count", 0, 10)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_float_rejects_non_finite_values(value):
    """NaN is the load-bearing case: `nan < minimum` and `nan > maximum` are both
    False, so a range check alone would let it through silently -- this only fails
    because the finiteness check runs first."""
    with pytest.raises(SecurityValidationError, match="must be finite"):
        SecurityValidator.validate_float(value, "threshold", -1000.0, 1000.0)


def test_validate_float_boundaries_are_inclusive():
    assert SecurityValidator.validate_float(0.0, "x", 0.0, 1.0) == 0.0
    assert SecurityValidator.validate_float(1.0, "x", 0.0, 1.0) == 1.0


def test_validate_float_rejects_just_outside_boundaries():
    with pytest.raises(SecurityValidationError, match="must be between"):
        SecurityValidator.validate_float(-0.0001, "x", 0.0, 1.0)
    with pytest.raises(SecurityValidationError, match="must be between"):
        SecurityValidator.validate_float(1.0001, "x", 0.0, 1.0)


@pytest.mark.parametrize("value", ["abc", None, [], {}])
def test_validate_float_rejects_non_numeric(value):
    with pytest.raises(SecurityValidationError, match="must be a number"):
        SecurityValidator.validate_float(value, "x", 0.0, 1.0)


def test_validate_float_coerces_numeric_strings():
    assert SecurityValidator.validate_float("0.5", "x", 0.0, 1.0) == 0.5


def test_validate_int_boundaries_are_inclusive():
    assert SecurityValidator.validate_int(0, "x", 0, 10) == 0
    assert SecurityValidator.validate_int(10, "x", 0, 10) == 10


def test_validate_int_rejects_just_outside_boundaries():
    with pytest.raises(SecurityValidationError, match="must be between"):
        SecurityValidator.validate_int(-1, "x", 0, 10)
    with pytest.raises(SecurityValidationError, match="must be between"):
        SecurityValidator.validate_int(11, "x", 0, 10)


@pytest.mark.parametrize("value", ["abc", None, [], {}, "3.5"])
def test_validate_int_rejects_non_integer(value):
    with pytest.raises(SecurityValidationError, match="must be an integer"):
        SecurityValidator.validate_int(value, "x", 0, 10)


def test_validate_int_coerces_integer_strings():
    assert SecurityValidator.validate_int("7", "x", 0, 10) == 7


# ---------------------------------------------------------------------------
# validate_data_directory
# ---------------------------------------------------------------------------

def test_validate_data_directory_accepts_existing_directory(tmp_path):
    result = SecurityValidator.validate_data_directory(str(tmp_path), SecurityConfig())
    assert result == os.path.abspath(str(tmp_path))


def test_validate_data_directory_rejects_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SecurityValidationError, match="does not exist"):
        SecurityValidator.validate_data_directory(str(missing), SecurityConfig())


# ---------------------------------------------------------------------------
# validate_request_context
# ---------------------------------------------------------------------------

def test_validate_request_context_defaults_when_none(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)

    result = SecurityValidator.validate_request_context(None, SecurityConfig())

    assert result == {"ip_address": "127.0.0.1", "username": "local-user"}


def test_validate_request_context_falls_back_to_user_env_var(monkeypatch):
    monkeypatch.setenv("USER", "alice")

    result = SecurityValidator.validate_request_context({}, SecurityConfig())

    assert result["username"] == "alice"


def test_validate_request_context_rejects_unexpected_field():
    with pytest.raises(SecurityValidationError, match="Unexpected fields"):
        SecurityValidator.validate_request_context({"password": "hunter2"}, SecurityConfig())


def test_validate_request_context_sanitizes_explicit_values():
    result = SecurityValidator.validate_request_context(
        {"ip_address": "10.0.0.5", "username": "  bob  "}, SecurityConfig()
    )
    assert result == {"ip_address": "10.0.0.5", "username": "bob"}


# ---------------------------------------------------------------------------
# RequestRateLimiter
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def time(self):
        return self.now


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(security, "time", clock)
    return clock


def test_rate_limiter_allows_up_to_the_limit_then_blocks(fake_clock):
    limiter = RequestRateLimiter()
    for _ in range(3):
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=3, user_limit=100, window_seconds=60)

    with pytest.raises(RateLimitExceededError) as excinfo:
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=3, user_limit=100, window_seconds=60)

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after_seconds == 60


def test_rate_limiter_allows_again_once_the_window_fully_elapses(fake_clock):
    limiter = RequestRateLimiter()
    for _ in range(3):
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=3, user_limit=100, window_seconds=60)

    fake_clock.now += 60
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=3, user_limit=100, window_seconds=60)


def test_rate_limiter_ip_and_user_buckets_are_independent(fake_clock):
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)
    with pytest.raises(RateLimitExceededError, match="IP address"):
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)

    # A different IP for the same user is a different IP-bucket key -> still allowed.
    limiter.enforce("analysis", "2.2.2.2", "alice", ip_limit=1, user_limit=100, window_seconds=60)


def test_rate_limiter_user_bucket_blocks_across_different_ips(fake_clock):
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=100, user_limit=1, window_seconds=60)

    with pytest.raises(RateLimitExceededError, match="this user"):
        limiter.enforce("analysis", "2.2.2.2", "alice", ip_limit=100, user_limit=1, window_seconds=60)


def test_rate_limiter_operations_have_independent_buckets(fake_clock):
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=1, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=1, window_seconds=60)

    # Same ip/user, different operation name -> a fresh bucket, not blocked.
    limiter.enforce("backtest", "1.1.1.1", "alice", ip_limit=1, user_limit=1, window_seconds=60)


def test_rate_limiter_sliding_window_ages_out_events_one_at_a_time(fake_clock):
    """Precise trace of the sliding window: limit=2, window=10s. Events at t=0 and t=5
    fill the bucket; at t=8 neither has aged out (8-0=8 < 10) so a 3rd call is still
    blocked; at t=10 the t=0 event is exactly 10s old and ages out, freeing a slot."""
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=2, user_limit=100, window_seconds=10)
    fake_clock.now = 5
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=2, user_limit=100, window_seconds=10)

    fake_clock.now = 8
    with pytest.raises(RateLimitExceededError):
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=2, user_limit=100, window_seconds=10)

    fake_clock.now = 10
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=2, user_limit=100, window_seconds=10)


def test_rate_limit_exceeded_error_to_response_shape(fake_clock):
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)
    with pytest.raises(RateLimitExceededError) as excinfo:
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)

    assert excinfo.value.to_response() == {
        "status": 429,
        "error": "Too Many Requests",
        "message": str(excinfo.value),
        "retry_after_seconds": 60,
    }


def test_rate_limit_exceeded_error_retry_after_has_a_floor_of_one_second(fake_clock):
    limiter = RequestRateLimiter()
    limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)
    fake_clock.now = 59.7  # window nearly elapsed -> raw remainder is well under 1 second

    with pytest.raises(RateLimitExceededError) as excinfo:
        limiter.enforce("analysis", "1.1.1.1", "alice", ip_limit=1, user_limit=100, window_seconds=60)

    assert excinfo.value.retry_after_seconds == 1

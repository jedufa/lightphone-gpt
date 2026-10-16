from __future__ import annotations

from pathlib import Path

import pytest

from lightphone_gpt.config import (
    PRODUCTION_ACTIVATION_SENTINEL,
    ServiceConfig,
    require_loopback_host,
)


def test_config_requires_explicit_absolute_state_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LIGHTPHONE_GPT_STATE_DIR is required"):
        ServiceConfig.from_env({})
    with pytest.raises(ValueError, match="must be an absolute path"):
        ServiceConfig.from_env({"LIGHTPHONE_GPT_STATE_DIR": "relative"})

    config = ServiceConfig.from_env({"LIGHTPHONE_GPT_STATE_DIR": str(tmp_path)})

    assert config.state_dir == tmp_path
    assert config.provider == "simulator"
    assert config.assistant == "mock"
    assert config.transcriber == "simulated-envelope"
    assert config.health_host == "127.0.0.1"
    assert config.health_port == 8824
    assert config.notifier_enabled is False
    assert config.notifier_host == "127.0.0.1"
    assert config.notifier_port == 8825
    assert config.notifier_wait_seconds == 25.0
    assert config.local_timezone == "UTC"
    assert config.bridge_database.parent == tmp_path
    assert config.simulator_database.parent == tmp_path


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.4", "localhost", "example.test"])
def test_health_host_fails_closed_for_non_numeric_loopback(host: str) -> None:
    with pytest.raises(ValueError):
        require_loopback_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.9.8.7", "::1"])
def test_health_host_accepts_only_numeric_loopback(host: str) -> None:
    assert require_loopback_host(host) == host


def test_notifier_config_is_explicit_strict_and_secret_safe(tmp_path: Path) -> None:
    token = "test_notifier_token_0123456789_ABCDEF"
    config = ServiceConfig.from_env(
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
            "LIGHTPHONE_GPT_NOTIFIER_TOKEN": token,
            "LIGHTPHONE_GPT_NOTIFIER_HOST": "127.0.0.2",
            "LIGHTPHONE_GPT_NOTIFIER_PORT": "9912",
            "LIGHTPHONE_GPT_NOTIFIER_WAIT_SECONDS": "24.5",
        }
    )

    assert config.notifier_enabled is True
    assert config.notifier_host == "127.0.0.2"
    assert config.notifier_port == 9912
    assert config.notifier_wait_seconds == 24.5
    assert config.notifier_token == token
    assert token not in repr(config)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"LIGHTPHONE_GPT_NOTIFIER_PORT": "9912"}, "TOKEN is required"),
        ({"LIGHTPHONE_GPT_NOTIFIER_TOKEN": ""}, "TOKEN is required"),
        ({"LIGHTPHONE_GPT_NOTIFIER_TOKEN": "too-short"}, "TOKEN is invalid"),
        (
            {
                "LIGHTPHONE_GPT_NOTIFIER_TOKEN": "x" * 32,
                "LIGHTPHONE_GPT_NOTIFIER_HOST": "0.0.0.0",
            },
            "loopback",
        ),
        (
            {
                "LIGHTPHONE_GPT_NOTIFIER_TOKEN": "x" * 32,
                "LIGHTPHONE_GPT_NOTIFIER_PORT": "8824",
            },
            "must differ",
        ),
        (
            {
                "LIGHTPHONE_GPT_NOTIFIER_TOKEN": "x" * 32,
                "LIGHTPHONE_GPT_NOTIFIER_WAIT_SECONDS": "31",
            },
            "outside the allowed range",
        ),
    ],
)
def test_notifier_config_fails_closed(
    tmp_path: Path, changes: dict[str, str], message: str
) -> None:
    environment = {"LIGHTPHONE_GPT_STATE_DIR": str(tmp_path), **changes}
    with pytest.raises(ValueError, match=message):
        ServiceConfig.from_env(environment)


def test_config_parses_bounded_runtime_settings(tmp_path: Path) -> None:
    config = ServiceConfig.from_env(
        {
            "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
            "LIGHTPHONE_GPT_TRANSCRIBER": "faster-whisper",
            "LIGHTPHONE_GPT_HEALTH_PORT": "9911",
            "LIGHTPHONE_GPT_POLL_SECONDS": "0.25",
            "LIGHTPHONE_GPT_DRAIN_MAX_STEPS": "12",
            "LIGHTPHONE_GPT_OUTPUT_MODE": "per_turn",
            "LIGHTPHONE_GPT_CONFIRM_OUTPUT_DELETION_AFTER": "3",
            "LIGHTPHONE_GPT_MAX_PENDING_JOBS": "17",
            "LIGHTPHONE_GPT_MAX_QUESTION_CHARS": "4096",
            "LIGHTPHONE_GPT_MAX_MODEL_CALLS_PER_DAY": "23",
            "LIGHTPHONE_GPT_MAX_TRANSCRIPTION_ATTEMPTS": "9",
            "LIGHTPHONE_GPT_MAX_MODEL_ATTEMPTS": "10",
            "LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS": "11",
            "LIGHTPHONE_GPT_DELETE_SOURCE_AFTER_PUBLISH": "true",
            "LIGHTPHONE_GPT_MAX_SOURCE_DELETE_READBACKS": "12",
            "LIGHTPHONE_GPT_CONFIRM_SOURCE_DELETION_AFTER": "3",
            "LIGHTPHONE_GPT_RETRY_INITIAL_SECONDS": "2.5",
            "LIGHTPHONE_GPT_RETRY_MAX_SECONDS": "45",
            "LIGHTPHONE_GPT_LOCAL_TIMEZONE": "America/New_York",
        }
    )

    assert config.provider == "simulator"
    assert config.assistant == "mock"
    assert config.transcriber == "faster-whisper"
    assert config.health_port == 9911
    assert config.poll_interval_seconds == 0.25
    assert config.drain_max_steps == 12
    assert config.output_mode == "per_turn"
    assert config.confirm_output_deletion_after == 3
    assert config.worker_config().confirm_output_deletion_after == 3
    assert config.worker_config().max_pending_jobs == 17
    assert config.worker_config().max_question_chars == 4096
    assert config.worker_config().max_model_calls_per_day == 23
    assert config.worker_config().max_transcription_attempts == 9
    assert config.worker_config().max_model_attempts == 10
    assert config.worker_config().max_update_readback_attempts == 11
    assert config.delete_source_after_publish is True
    assert config.worker_config().delete_source_after_publish is True
    assert config.worker_config().max_source_delete_readbacks == 12
    assert config.worker_config().confirm_source_deletion_after == 3
    assert config.worker_config().retry_delay.total_seconds() == 2.5
    assert config.worker_config().retry_max_delay.total_seconds() == 45
    assert config.local_timezone == "America/New_York"
    assert config.worker_config().local_timezone == "America/New_York"


def test_config_rejects_invalid_local_timezone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LOCAL_TIMEZONE"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                "LIGHTPHONE_GPT_LOCAL_TIMEZONE": "Moon/Base",
            }
        )


def test_config_reserves_title_space_for_local_time_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OUTPUT_TITLE"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                "LIGHTPHONE_GPT_OUTPUT_TITLE": "A" * 95,
            }
        )


def _production_environment(tmp_path: Path) -> dict[str, str]:
    authorization = tmp_path / "authorization"
    return {
        "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path / "state"),
        "LIGHTPHONE_GPT_PROVIDER": "light",
        "LIGHTPHONE_GPT_ASSISTANT": "chatgpt",
        "LIGHTPHONE_GPT_TRANSCRIBER": "faster-whisper",
        "LIGHTPHONE_GPT_OUTPUT_MODE": "rolling",
        "LIGHTPHONE_GPT_PRODUCTION_ACTIVATION": PRODUCTION_ACTIVATION_SENTINEL,
        "LIGHTPHONE_GPT_CHATGPT_CREDENTIAL_FILE": str(
            authorization / "credentials.json"
        ),
        "LIGHTPHONE_GPT_CHATGPT_HOST_ID_FILE": str(authorization / "host-id"),
        "LIGHTPHONE_GPT_CHATGPT_MODEL": "gpt-test-exact",
        "lightphone_username": "private-user@example.test",
        "lightphone_password": "private-password",
    }


def test_real_adapters_require_exact_complete_production_activation(
    tmp_path: Path,
) -> None:
    env = _production_environment(tmp_path)

    missing = dict(env)
    missing.pop("LIGHTPHONE_GPT_PRODUCTION_ACTIVATION")
    with pytest.raises(ValueError, match="sentinel"):
        ServiceConfig.from_env(missing)

    wrong = dict(env)
    wrong["LIGHTPHONE_GPT_PRODUCTION_ACTIVATION"] = "yes"
    with pytest.raises(ValueError, match="is invalid"):
        ServiceConfig.from_env(wrong)

    for name in (
        "lightphone_username",
        "lightphone_password",
        "LIGHTPHONE_GPT_CHATGPT_CREDENTIAL_FILE",
        "LIGHTPHONE_GPT_CHATGPT_HOST_ID_FILE",
        "LIGHTPHONE_GPT_CHATGPT_MODEL",
    ):
        incomplete = dict(env)
        secret_value = incomplete.pop(name)
        with pytest.raises(ValueError) as captured:
            ServiceConfig.from_env(incomplete)
        assert secret_value not in str(captured.value)


def test_production_config_retains_no_light_credentials(tmp_path: Path) -> None:
    env = _production_environment(tmp_path)

    config = ServiceConfig.from_env(env)

    assert config.production_activated is True
    assert config.provider == "light"
    assert config.assistant == "chatgpt"
    assert config.transcriber == "faster-whisper"
    assert config.output_mode == "rolling"
    assert config.chatgpt_model == "gpt-test-exact"
    assert config.bridge_database.name == "bridge-production.sqlite3"
    serialized = repr(config)
    assert env["lightphone_username"] not in serialized
    assert env["lightphone_password"] not in serialized


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LIGHTPHONE_GPT_OUTPUT_MODE", "per_turn", "rolling output"),
        (
            "LIGHTPHONE_GPT_TRANSCRIBER",
            "simulated-envelope",
            "local transcription",
        ),
        ("LIGHTPHONE_GPT_CHATGPT_MODEL", "model with spaces", "exact ChatGPT"),
        (
            "LIGHTPHONE_GPT_CHATGPT_HOST_ID_FILE",
            "relative-host-id",
            "authorization paths",
        ),
    ],
)
def test_production_shape_fails_closed(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    env = _production_environment(tmp_path)
    env[name] = value

    with pytest.raises(ValueError, match=message):
        ServiceConfig.from_env(env)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIGHTPHONE_GPT_MAX_PENDING_JOBS", "0"),
        ("LIGHTPHONE_GPT_MAX_PENDING_JOBS", "100001"),
        ("LIGHTPHONE_GPT_MAX_QUESTION_CHARS", "24001"),
        ("LIGHTPHONE_GPT_MAX_MODEL_CALLS_PER_DAY", "0"),
        ("LIGHTPHONE_GPT_MAX_MODEL_CALLS_PER_DAY", "100001"),
        ("LIGHTPHONE_GPT_MAX_TRANSCRIPTION_ATTEMPTS", "0"),
        ("LIGHTPHONE_GPT_MAX_MODEL_ATTEMPTS", "33"),
        ("LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS", "0"),
        ("LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS", "101"),
        ("LIGHTPHONE_GPT_MAX_SOURCE_DELETE_READBACKS", "0"),
        ("LIGHTPHONE_GPT_MAX_SOURCE_DELETE_READBACKS", "101"),
        ("LIGHTPHONE_GPT_CONFIRM_SOURCE_DELETION_AFTER", "1"),
        ("LIGHTPHONE_GPT_CONFIRM_SOURCE_DELETION_AFTER", "101"),
    ],
)
def test_operational_limits_are_bounded(
    tmp_path: Path, name: str, value: str
) -> None:
    with pytest.raises(ValueError, match="outside the allowed range"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                name: value,
            }
        )


def test_retry_maximum_cannot_be_shorter_than_initial_delay(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shorter"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                "LIGHTPHONE_GPT_RETRY_INITIAL_SECONDS": "60",
                "LIGHTPHONE_GPT_RETRY_MAX_SECONDS": "30",
            }
        )


def test_source_cleanup_requires_an_explicit_boolean_setting(tmp_path: Path) -> None:
    default = ServiceConfig.from_env({"LIGHTPHONE_GPT_STATE_DIR": str(tmp_path)})
    assert default.delete_source_after_publish is False

    with pytest.raises(ValueError, match="must be a boolean"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                "LIGHTPHONE_GPT_DELETE_SOURCE_AFTER_PUBLISH": "sometimes",
            }
        )


def test_whisper_cpp_requires_both_local_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="whisper-cpp requires"):
        ServiceConfig.from_env(
            {
                "LIGHTPHONE_GPT_STATE_DIR": str(tmp_path),
                "LIGHTPHONE_GPT_TRANSCRIBER": "whisper-cpp",
            }
        )

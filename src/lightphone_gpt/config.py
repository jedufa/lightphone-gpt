"""Environment-only runtime configuration with safe Pre-0 defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from ipaddress import ip_address
import os
from pathlib import Path
import re
from typing import Literal

from .output_titles import DEFAULT_LOCAL_TIMEZONE, local_timezone
from .retention import RetentionPolicy
from .worker import WorkerConfig


ProviderName = Literal["simulator", "light"]
AssistantName = Literal["mock", "chatgpt"]
TranscriberName = Literal["simulated-envelope", "whisper-cpp", "faster-whisper"]


PRODUCTION_ACTIVATION_SENTINEL = "ENABLE_REAL_LIGHT_AND_CHATGPT_V1"
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_NOTIFIER_TOKEN = re.compile(r"[A-Za-z0-9._~-]{32,256}\Z")
_NOTIFIER_ENV_NAMES = frozenset(
    {
        "LIGHTPHONE_GPT_NOTIFIER_TOKEN",
        "LIGHTPHONE_GPT_NOTIFIER_HOST",
        "LIGHTPHONE_GPT_NOTIFIER_PORT",
        "LIGHTPHONE_GPT_NOTIFIER_WAIT_SECONDS",
    }
)


def require_loopback_host(
    host: str, *, setting: str = "LIGHTPHONE_GPT_HEALTH_HOST"
) -> str:
    """Accept numeric loopback addresses only, avoiding DNS ambiguity."""

    try:
        address = ip_address(host)
    except ValueError:
        raise ValueError(f"{setting} must be a numeric loopback address") from None
    if not address.is_loopback:
        raise ValueError(f"{setting} must be a loopback address")
    return host


def require_notifier_token(token: str) -> str:
    """Require an opaque, header-safe token without normalizing secret bytes."""

    if not _NOTIFIER_TOKEN.fullmatch(token):
        raise ValueError("LIGHTPHONE_GPT_NOTIFIER_TOKEN is invalid")
    return token


def _choice(
    env: Mapping[str, str], name: str, default: str, choices: set[str]
) -> str:
    value = env.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} has an unsupported value")
    return value


def _integer(
    env: Mapping[str, str], name: str, default: int, *, minimum: int, maximum: int
) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    state_dir: Path
    provider: ProviderName = "simulator"
    assistant: AssistantName = "mock"
    transcriber: TranscriberName = "simulated-envelope"
    health_host: str = "127.0.0.1"
    health_port: int = 8824
    notifier_host: str = "127.0.0.1"
    notifier_port: int = 8825
    notifier_wait_seconds: float = 25.0
    notifier_token: str | None = field(default=None, repr=False, compare=False)
    poll_interval_seconds: float = 5.0
    retention_interval_seconds: float = 3600.0
    drain_max_steps: int = 100
    confirm_output_deletion_after: int = 2
    max_pending_jobs: int = 100
    max_question_chars: int = 8_000
    max_model_calls_per_day: int = 100
    max_transcription_attempts: int = 8
    max_model_attempts: int = 8
    max_update_readback_attempts: int = 8
    delete_source_after_publish: bool = False
    max_source_delete_readbacks: int = 8
    confirm_source_deletion_after: int = 2
    retry_initial_seconds: float = 30.0
    retry_max_seconds: float = 600.0
    output_mode: Literal["rolling", "per_turn"] = "rolling"
    output_title: str = "AI READY"
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
    closed_conversation_days: int = 30
    metadata_days: int = 90
    whisper_executable: str | None = None
    whisper_model: Path | None = None
    faster_whisper_model: str = "base"
    faster_whisper_model_cache: Path | None = None
    production_activated: bool = False
    chatgpt_credential_file: Path | None = None
    chatgpt_host_id_file: Path | None = None
    chatgpt_model: str | None = None

    def __post_init__(self) -> None:
        if not self.state_dir.is_absolute():
            raise ValueError("LIGHTPHONE_GPT_STATE_DIR must be an absolute path")
        require_loopback_host(self.health_host)
        require_loopback_host(
            self.notifier_host, setting="LIGHTPHONE_GPT_NOTIFIER_HOST"
        )
        if not 1 <= self.health_port <= 65535:
            raise ValueError("LIGHTPHONE_GPT_HEALTH_PORT is outside the allowed range")
        if not 1 <= self.notifier_port <= 65535:
            raise ValueError("LIGHTPHONE_GPT_NOTIFIER_PORT is outside the allowed range")
        if not 1 <= self.notifier_wait_seconds <= 30:
            raise ValueError(
                "LIGHTPHONE_GPT_NOTIFIER_WAIT_SECONDS is outside the allowed range"
            )
        if self.notifier_token is not None:
            require_notifier_token(self.notifier_token)
            if self.notifier_port == self.health_port:
                raise ValueError(
                    "LIGHTPHONE_GPT_NOTIFIER_PORT must differ from the health port"
                )
        if self.poll_interval_seconds <= 0 or self.retention_interval_seconds <= 0:
            raise ValueError("service intervals must be positive")
        if self.drain_max_steps < 1:
            raise ValueError("LIGHTPHONE_GPT_DRAIN_MAX_STEPS must be positive")
        if not 1 <= self.max_pending_jobs <= 100_000:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_PENDING_JOBS is outside the allowed range"
            )
        if not 1 <= self.max_question_chars <= 24_000:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_QUESTION_CHARS is outside the allowed range"
            )
        if not 1 <= self.max_model_calls_per_day <= 100_000:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_MODEL_CALLS_PER_DAY is outside the allowed range"
            )
        if not 1 <= self.max_transcription_attempts <= 32:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_TRANSCRIPTION_ATTEMPTS is outside the allowed range"
            )
        if not 1 <= self.max_model_attempts <= 32:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_MODEL_ATTEMPTS is outside the allowed range"
            )
        if not 1 <= self.max_update_readback_attempts <= 100:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS is outside the allowed range"
            )
        if not 1 <= self.max_source_delete_readbacks <= 100:
            raise ValueError(
                "LIGHTPHONE_GPT_MAX_SOURCE_DELETE_READBACKS is outside the allowed range"
            )
        if not 2 <= self.confirm_source_deletion_after <= 100:
            raise ValueError(
                "LIGHTPHONE_GPT_CONFIRM_SOURCE_DELETION_AFTER is outside the allowed range"
            )
        if self.retry_initial_seconds <= 0 or self.retry_max_seconds <= 0:
            raise ValueError("retry intervals must be positive")
        if self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError(
                "LIGHTPHONE_GPT_RETRY_MAX_SECONDS cannot be shorter than the initial delay"
            )
        if self.transcriber == "whisper-cpp" and (
            not self.whisper_executable or self.whisper_model is None
        ):
            raise ValueError(
                "whisper-cpp requires LIGHTPHONE_GPT_WHISPER_EXECUTABLE and "
                "LIGHTPHONE_GPT_WHISPER_MODEL"
            )
        if (
            not self.output_title
            or self.output_title != self.output_title.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.output_title
            )
            or len(self.output_title) > 94
        ):
            raise ValueError("LIGHTPHONE_GPT_OUTPUT_TITLE is invalid")
        local_timezone(self.local_timezone)
        if self.production_activated:
            self._validate_production_shape()

    def _validate_production_shape(self) -> None:
        if self.provider != "light" or self.assistant != "chatgpt":
            raise ValueError(
                "production activation requires the Light and ChatGPT adapters"
            )
        if self.transcriber not in {"whisper-cpp", "faster-whisper"}:
            raise ValueError("production activation requires local transcription")
        if self.output_mode != "rolling":
            raise ValueError("production activation requires rolling output")
        if (
            self.chatgpt_credential_file is None
            or self.chatgpt_host_id_file is None
            or not self.chatgpt_credential_file.is_absolute()
            or not self.chatgpt_host_id_file.is_absolute()
            or self.chatgpt_credential_file == self.chatgpt_host_id_file
            or self.chatgpt_credential_file.parent
            != self.chatgpt_host_id_file.parent
        ):
            raise ValueError(
                "production activation requires absolute ChatGPT authorization paths"
            )
        if self.chatgpt_model is None or not _MODEL_NAME.fullmatch(
            self.chatgpt_model
        ):
            raise ValueError(
                "production activation requires one exact ChatGPT model"
            )
        if self.transcriber == "whisper-cpp" and (
            self.whisper_executable is None
            or not Path(self.whisper_executable).is_absolute()
            or self.whisper_model is None
            or not self.whisper_model.is_absolute()
        ):
            raise ValueError(
                "production whisper-cpp paths must be absolute"
            )
        if (
            self.transcriber == "faster-whisper"
            and self.faster_whisper_model_cache is not None
            and not self.faster_whisper_model_cache.is_absolute()
        ):
            raise ValueError(
                "production faster-whisper cache must be absolute"
            )

    def validate_production_environment(
        self, environ: Mapping[str, str] | None = None
    ) -> None:
        """Fail closed without retaining or reporting credential values."""

        if not self.production_activated:
            raise ValueError("production activation sentinel is required")
        self._validate_production_shape()
        env = os.environ if environ is None else environ
        if not env.get("lightphone_username", "").strip() or not env.get(
            "lightphone_password", ""
        ).strip():
            raise ValueError("Light credentials are missing from the environment")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServiceConfig":
        env = os.environ if environ is None else environ
        raw_state = env.get("LIGHTPHONE_GPT_STATE_DIR", "").strip()
        if not raw_state:
            raise ValueError("LIGHTPHONE_GPT_STATE_DIR is required")
        state_dir = Path(raw_state).expanduser()
        if not state_dir.is_absolute():
            raise ValueError("LIGHTPHONE_GPT_STATE_DIR must be an absolute path")

        provider = _choice(
            env, "LIGHTPHONE_GPT_PROVIDER", "simulator", {"simulator", "light"}
        )
        assistant = _choice(
            env, "LIGHTPHONE_GPT_ASSISTANT", "mock", {"mock", "chatgpt"}
        )
        transcriber = _choice(
            env,
            "LIGHTPHONE_GPT_TRANSCRIBER",
            "simulated-envelope",
            {"simulated-envelope", "whisper-cpp", "faster-whisper"},
        )
        output_mode = _choice(
            env, "LIGHTPHONE_GPT_OUTPUT_MODE", "rolling", {"rolling", "per_turn"}
        )
        activation = env.get("LIGHTPHONE_GPT_PRODUCTION_ACTIVATION", "").strip()
        if activation and activation != PRODUCTION_ACTIVATION_SENTINEL:
            raise ValueError("LIGHTPHONE_GPT_PRODUCTION_ACTIVATION is invalid")
        production_activated = activation == PRODUCTION_ACTIVATION_SENTINEL
        real_requested = provider != "simulator" or assistant != "mock"
        if real_requested and not production_activated:
            raise ValueError("production activation sentinel is required")
        raw_credential = env.get(
            "LIGHTPHONE_GPT_CHATGPT_CREDENTIAL_FILE", ""
        ).strip()
        raw_host_id = env.get("LIGHTPHONE_GPT_CHATGPT_HOST_ID_FILE", "").strip()
        executable = env.get("LIGHTPHONE_GPT_WHISPER_EXECUTABLE", "").strip() or None
        raw_model = env.get("LIGHTPHONE_GPT_WHISPER_MODEL", "").strip()
        raw_cache = env.get("LIGHTPHONE_GPT_FASTER_WHISPER_CACHE", "").strip()
        notifier_configured = any(name in env for name in _NOTIFIER_ENV_NAMES)
        raw_notifier_token = env.get("LIGHTPHONE_GPT_NOTIFIER_TOKEN")
        if notifier_configured and not raw_notifier_token:
            raise ValueError(
                "LIGHTPHONE_GPT_NOTIFIER_TOKEN is required when notifier settings are present"
            )

        config = cls(
            state_dir=state_dir,
            provider=provider,  # type: ignore[arg-type]
            assistant=assistant,  # type: ignore[arg-type]
            transcriber=transcriber,  # type: ignore[arg-type]
            health_host=require_loopback_host(
                env.get("LIGHTPHONE_GPT_HEALTH_HOST", "127.0.0.1").strip()
            ),
            health_port=_integer(
                env, "LIGHTPHONE_GPT_HEALTH_PORT", 8824, minimum=1, maximum=65535
            ),
            notifier_host=require_loopback_host(
                env.get("LIGHTPHONE_GPT_NOTIFIER_HOST", "127.0.0.1").strip(),
                setting="LIGHTPHONE_GPT_NOTIFIER_HOST",
            ),
            notifier_port=_integer(
                env, "LIGHTPHONE_GPT_NOTIFIER_PORT", 8825, minimum=1, maximum=65535
            ),
            notifier_wait_seconds=_seconds(
                env, "LIGHTPHONE_GPT_NOTIFIER_WAIT_SECONDS", 25.0
            ),
            notifier_token=raw_notifier_token,
            poll_interval_seconds=_seconds(
                env, "LIGHTPHONE_GPT_POLL_SECONDS", 5.0
            ),
            retention_interval_seconds=_seconds(
                env, "LIGHTPHONE_GPT_RETENTION_SECONDS", 3600.0
            ),
            drain_max_steps=_integer(
                env,
                "LIGHTPHONE_GPT_DRAIN_MAX_STEPS",
                100,
                minimum=1,
                maximum=100_000,
            ),
            confirm_output_deletion_after=_integer(
                env,
                "LIGHTPHONE_GPT_CONFIRM_OUTPUT_DELETION_AFTER",
                2,
                minimum=2,
                maximum=100,
            ),
            max_pending_jobs=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_PENDING_JOBS",
                100,
                minimum=1,
                maximum=100_000,
            ),
            max_question_chars=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_QUESTION_CHARS",
                8_000,
                minimum=1,
                maximum=24_000,
            ),
            max_model_calls_per_day=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_MODEL_CALLS_PER_DAY",
                100,
                minimum=1,
                maximum=100_000,
            ),
            max_transcription_attempts=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_TRANSCRIPTION_ATTEMPTS",
                8,
                minimum=1,
                maximum=32,
            ),
            max_model_attempts=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_MODEL_ATTEMPTS",
                8,
                minimum=1,
                maximum=32,
            ),
            max_update_readback_attempts=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_UPDATE_READBACK_ATTEMPTS",
                8,
                minimum=1,
                maximum=100,
            ),
            delete_source_after_publish=_boolean(
                env, "LIGHTPHONE_GPT_DELETE_SOURCE_AFTER_PUBLISH", False
            ),
            max_source_delete_readbacks=_integer(
                env,
                "LIGHTPHONE_GPT_MAX_SOURCE_DELETE_READBACKS",
                8,
                minimum=1,
                maximum=100,
            ),
            confirm_source_deletion_after=_integer(
                env,
                "LIGHTPHONE_GPT_CONFIRM_SOURCE_DELETION_AFTER",
                2,
                minimum=2,
                maximum=100,
            ),
            retry_initial_seconds=_seconds(
                env, "LIGHTPHONE_GPT_RETRY_INITIAL_SECONDS", 30.0
            ),
            retry_max_seconds=_seconds(
                env, "LIGHTPHONE_GPT_RETRY_MAX_SECONDS", 600.0
            ),
            output_mode=output_mode,  # type: ignore[arg-type]
            output_title=env.get("LIGHTPHONE_GPT_OUTPUT_TITLE", "AI READY"),
            local_timezone=env.get(
                "LIGHTPHONE_GPT_LOCAL_TIMEZONE", DEFAULT_LOCAL_TIMEZONE
            ),
            closed_conversation_days=_integer(
                env,
                "LIGHTPHONE_GPT_CLOSED_CONVERSATION_DAYS",
                30,
                minimum=0,
                maximum=36_500,
            ),
            metadata_days=_integer(
                env,
                "LIGHTPHONE_GPT_METADATA_DAYS",
                90,
                minimum=0,
                maximum=36_500,
            ),
            whisper_executable=executable,
            whisper_model=Path(raw_model).expanduser() if raw_model else None,
            faster_whisper_model=env.get(
                "LIGHTPHONE_GPT_FASTER_WHISPER_MODEL", "base"
            ).strip()
            or "base",
            faster_whisper_model_cache=(
                Path(raw_cache).expanduser() if raw_cache else state_dir / "models"
            ),
            production_activated=production_activated,
            chatgpt_credential_file=(
                Path(raw_credential).expanduser() if raw_credential else None
            ),
            chatgpt_host_id_file=(
                Path(raw_host_id).expanduser() if raw_host_id else None
            ),
            chatgpt_model=(
                env.get("LIGHTPHONE_GPT_CHATGPT_MODEL", "").strip() or None
            ),
        )
        if production_activated:
            config.validate_production_environment(env)
        return config

    @property
    def notifier_enabled(self) -> bool:
        return self.notifier_token is not None

    @property
    def bridge_database(self) -> Path:
        # Production starts with a fresh enrollment ledger and never inherits
        # simulator baselines, conversations, or synthetic output identities.
        filename = (
            "bridge-production.sqlite3"
            if self.production_activated
            else "bridge.sqlite3"
        )
        return self.state_dir / filename

    @property
    def simulator_database(self) -> Path:
        return self.state_dir / "light-simulator.sqlite3"

    @property
    def transcription_temp_dir(self) -> Path:
        return self.state_dir / "tmp"

    def worker_config(self) -> WorkerConfig:
        return WorkerConfig(
            output_mode=self.output_mode,
            output_title=self.output_title,
            local_timezone=self.local_timezone,
            confirm_output_deletion_after=self.confirm_output_deletion_after,
            max_pending_jobs=self.max_pending_jobs,
            max_question_chars=self.max_question_chars,
            max_model_calls_per_day=self.max_model_calls_per_day,
            max_transcription_attempts=self.max_transcription_attempts,
            max_model_attempts=self.max_model_attempts,
            max_update_readback_attempts=self.max_update_readback_attempts,
            delete_source_after_publish=self.delete_source_after_publish,
            max_source_delete_readbacks=self.max_source_delete_readbacks,
            confirm_source_deletion_after=self.confirm_source_deletion_after,
            retry_delay=timedelta(seconds=self.retry_initial_seconds),
            retry_max_delay=timedelta(seconds=self.retry_max_seconds),
        )

    def retention_policy(self) -> RetentionPolicy:
        return RetentionPolicy(
            closed_conversation_days=self.closed_conversation_days,
            metadata_days=self.metadata_days,
        )

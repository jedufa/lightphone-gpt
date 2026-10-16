# Installation

## Requirements

- Python 3.11 or newer
- Git when installing the optional pinned Light client
- Enough local storage for SQLite and, when enabled, a speech model
- A Light account for real Notes access
- An eligible ChatGPT account for the implemented Sign in with ChatGPT adapter

Begin with the simulator. Do not put real credentials in the repository or its
working tree.

## Simulator-first setup

```bash
git clone https://github.com/jedufa/lightphone-gpt.git
cd lightphone-gpt
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

export LIGHTPHONE_GPT_STATE_DIR="$PWD/runtime"
lightphone-gpt init
lightphone-gpt simulator seed-text \
  --content 'Assistant, explain this bridge in one sentence.'
lightphone-gpt cycle
lightphone-gpt status
```

These defaults select the simulator, mock assistant, and simulated-envelope
transcriber. They perform no cloud request.

The repository includes `uv.lock` for a reproducible, hash-checked environment
across the supported Python range. If `uv` is installed, the equivalent locked
setup is:

```bash
uv sync --frozen --extra dev
uv run --frozen lightphone-gpt init
```

## Optional adapters

Install only the features you need:

```bash
python -m pip install -e '.[light]'
python -m pip install -e '.[stt]'
python -m pip install -e '.[chatgpt]'
```

With `uv`, replace those commands with `uv sync --frozen --extra light`,
`--extra stt`, or `--extra chatgpt`. Combine flags to enable several extras.

The Light extra pins `garado/light` to one reviewed commit. It remains an
unofficial reverse-engineered client. Re-audit the dependency before changing
that pin.

Faster-whisper downloads model data unless you pre-provision a local cache.
Model artifacts have their own licenses and are not included here. Store the
cache outside the repository and set its permissions for the service account.

## Credentials

Store runtime secrets in an owner-readable file outside the checkout, or inject
them through a secret manager. Restrict files and parent directories; on Unix,
`0700` directories and `0600` credential files are a sensible baseline.

The Light adapter reads `lightphone_username` and `lightphone_password`. The
bridge never needs a Light API key. Use a dedicated account when possible and
understand that the unofficial client authenticates against Light's service.

The implemented assistant uses Sign in with ChatGPT and stores OAuth material
in an owner-controlled credential file. It does not accept a ChatGPT password,
browser cookie, or OpenAI API key.

OpenAI's current documentation includes an explicit
[self-hosted VM workflow](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms)
for open-source apps: create a distinct host ID for the VM, complete OAuth
locally, transfer the protected credential record over a secure channel, and
let the VM own later token refreshes. Follow that guide rather than copying a
laptop host ID or performing an improvised browser-cookie transfer.

Keep access, refresh, and retained ID tokens in protected runtime storage and
out of source control, URLs, logs, analytics, and support transcripts, as
required by OpenAI's
[account and session guidance](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions).
The current self-hosted workflow notes that host-specific usage attribution and
revocation are not yet available for transferred sessions. A separately billed
OpenAI API key is a distinct authentication path and is not implemented here.

## Controlled real activation

Real providers fail closed unless all required settings and the exact
production activation sentinel are present. Use `.env.example` as a name-only
reference; keep populated values outside Git.

Before activation:

1. Back up the state database and protect it as user content.
2. Start with a new state directory so existing notes are baselined.
3. Confirm the health listener binds only to numeric loopback.
4. Perform a read-only Light probe and inspect only aggregate output.
5. Test one synthetic note before processing personal data.
6. Set conservative queue, input-size, daily-model-call, and retention limits.
7. Decide whether source-note deletion is acceptable. It is disabled by
   default.
8. Set `LIGHTPHONE_GPT_LOCAL_TIMEZONE` to the IANA timezone used for the
   `AI READY HH:MM` title (for example, `America/New_York`).
9. Run the full test suite against the exact commit you will install.

This repository does not include production-specific systemd units, host
inventory, firewall configuration, signing material, or deploy scripts. Build
those around the documented process boundary and keep them private unless they
receive a separate rights and security review.

## Service operation

`lightphone-gpt serve` runs the polling loop and loopback health endpoint.
`lightphone-gpt status` reports content-free counts. Use a dedicated operating
system account, an isolated state directory, explicit resource limits, and a
service manager that does not log environment values.

`Assistant, reset` is a user-facing recovery command for publication/model
ambiguity and bounded local processing stalls. It deliberately cannot clear
credential, account, configuration, plan, or daily-limit failures; those need
operator attention.

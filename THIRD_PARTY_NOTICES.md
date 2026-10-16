# Third-party notices

## garado/light

The optional Light Cloud adapter uses
[garado/light](https://github.com/garado/light), an unofficial Light device API
and CLI by Alexis Garado ("garado"), copyright 2026 Alexis Garado.

Lightphone GPT Bridge pins commit
`06620db5ac1068ce7be36c4994f5e793b8b3ead0`. The dependency is licensed under
the GNU General Public License version 3 or later and carries an attribution
term under GPLv3 section 7(b). Its complete license remains in the upstream
distribution. Lightphone GPT Bridge does not vendor or modify its source.

The upstream interface is reverse engineered, unsupported by Light, and may
change without notice.

## Click

The optional Light Cloud adapter installs
[Click](https://github.com/pallets/click) 8.1.8 for the pinned upstream client's
command-line interface. Click is licensed under the BSD 3-Clause License.

## faster-whisper

The optional local transcription adapter supports
[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper), licensed
under the MIT License. Model artifacts have their own upstream licenses and
are provisioned outside this repository.

The compatible PyAV release is pinned to 18.1.0 because faster-whisper 1.2.1
uses an argument removed in PyAV 19. PyAV is licensed under the BSD 3-Clause
License. FFmpeg licensing depends on the features in the installed binary
distribution.

## OpenAI Python and PyJWT

The optional ChatGPT account adapter uses the
[OpenAI Python library](https://github.com/openai/openai-python), licensed under
Apache-2.0, and [PyJWT](https://github.com/jpadilla/pyjwt), licensed under MIT.
They remain separately installed dependencies and are not vendored here.

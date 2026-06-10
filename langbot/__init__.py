"""Shared quiz-bot framework for the language study bots.

The package is split into transport-agnostic, subject-agnostic pieces:

- ``models``     dataclasses shared across the stack (entries, modes, views).
- ``loader``     reading vocabulary/phrase rows from CSV files.
- ``quiz``       distractor generation and quiz-mode resolution.
- ``session``    per-user spaced-repetition state and the quiz controller.
- ``engine``     transport-agnostic command/action dispatch.
- ``subjects``   per-language content + localized strings (JLPT vocab, EN/JA phrases).
- ``transports`` LINE and Telegram adapters plus message renderers.
- ``apps``       thin entrypoints that wire a subject to a transport.
"""

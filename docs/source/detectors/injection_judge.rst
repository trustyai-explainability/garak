garak.detectors.injection_judge
===============================

LLM-as-a-judge detector for indirect prompt injection conversations.

Scores each target response against a per-conversation, natural-language success
description carried by ``probes.injection.IndirectInjection``. The judge also
sees any tool calls the target emitted, so attacks that succeed by triggering a
harmful tool call are detected.

The judge model is instantiated via the generator interface and must inherit
``OpenAICompatible`` (OpenAI, NIM, Azure, Groq).

.. automodule:: garak.detectors.injection_judge
   :members:
   :undoc-members:
   :show-inheritance:

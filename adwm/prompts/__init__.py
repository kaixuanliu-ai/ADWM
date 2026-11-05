"""Prompt templates and response parsers for the supported environments.

Each submodule defines the history-free and history-conditioned prompt
templates (``*_TEMPLATE_NO_HIS`` / ``*_TEMPLATE``) and a ``*_projection``
function that extracts the environment action from a model response. They are
shared by the real-environment collectors and the world-model policy wrapper.
"""

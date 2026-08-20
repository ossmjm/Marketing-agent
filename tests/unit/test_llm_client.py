from __future__ import annotations

from app.llm.client import _strip_markdown_fences


def test_strip_markdown_fences_removes_json_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_strip_markdown_fences_removes_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_strip_markdown_fences_leaves_plain_json_untouched():
    raw = '{"a": 1}'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_strip_markdown_fences_handles_surrounding_whitespace():
    raw = '  \n```json\n{"a": 1}\n```\n  '
    assert _strip_markdown_fences(raw) == '{"a": 1}'

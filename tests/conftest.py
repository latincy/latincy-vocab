"""Shared test fixtures."""

import pytest


CAESAR_PASSAGE = (
    "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, "
    "aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur."
)


@pytest.fixture
def caesar_passage() -> str:
    return CAESAR_PASSAGE

"""Canonical data schemas for JIGNASA 2026 Internship Information Assistant.

Defines Pydantic models for student profiles, notices, candidate ranking,
citation evidence, and recommendations across the retrieval and evaluation pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, Field


class FieldStatus(BaseModel):
    """Tracks the extraction confidence, value, and ambiguity status of a notice field."""

    status: Literal["present", "missing", "ambiguous"]
    value: Any | None = None
    note: str | None = None


class Evidence(BaseModel):
    """Citation evidence grounding extracted data or explanations back to raw source notices."""

    notice_id: str
    chunk_id: str | None = None
    field: str | None = None
    quote: str
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None


class StudentProfile(BaseModel):
    """Academic and preference profile of a student querying for internships."""

    degree: str | None = None
    branch: str | None = None
    year: int | None = None
    cgpa: float | None = None
    skills: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_modes: list[str] = Field(default_factory=list)
    preferred_duration: str | None = None
    minimum_stipend: float | None = None
    domains: list[str] = Field(default_factory=list)


class Notice(BaseModel):
    """Parsed and structured internship opportunity notice."""

    notice_id: str
    title: str
    organization: str | None = None
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    degree: str | None = None
    branches: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    cgpa_min: float | None = None
    location: str | None = None
    mode: str | None = None
    duration: str | None = None
    stipend: float | None = None
    deadline: date | None = None
    deadline_status: Literal["active", "expired", "missing", "ambiguous"] = "missing"
    field_status: dict[str, FieldStatus] = Field(default_factory=dict)
    source: str | None = None


class Candidate(BaseModel):
    """Candidate match generated during multi-stage retrieval and ranking."""

    notice_id: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float
    rerank_score: float | None = None


class Recommendation(BaseModel):
    """Final recommendation card presented to the student with grounded explanation and warnings."""

    notice_id: str
    fit_label: Literal["Strong match", "Partial match", "Weak match", "Not eligible"]
    eligibility_status: Literal["eligible", "not_eligible", "unknown"]
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    field_warnings: list[str] = Field(default_factory=list)
    deadline_status: Literal["active", "expired", "missing", "ambiguous"]
    citations: list[Evidence] = Field(default_factory=list)
    caveat: str = (
        "Recommendations do not guarantee selection. Please verify all requirements on the official notice."
    )

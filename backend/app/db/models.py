"""SQLAlchemy models for requests and attempts.

Tables are created by Alembic. Application startup does not create them.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for the persistence models."""


class RequestRow(Base):
    """One chat request. Phase 4 stores it before the provider call."""

    __tablename__ = "requests"
    __table_args__ = (Index("ix_requests_created_at_id", "created_at", "id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    complexity_score: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    complexity_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    complexity_reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    selection_reason: Mapped[str] = mapped_column(Text, nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_cost: Mapped[Decimal | None] = mapped_column(Numeric(20, 12), nullable=True)
    cost_completeness: Mapped[str] = mapped_column(String(16), nullable=False)
    end_to_end_latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(14, 3), nullable=True)
    premium_baseline_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    premium_baseline_cost: Mapped[Decimal | None] = mapped_column(Numeric(20, 12), nullable=True)
    estimated_savings: Mapped[Decimal | None] = mapped_column(Numeric(20, 12), nullable=True)
    quality_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quality_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    quality_reasons: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    escalation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    returned_attempt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fallback_used: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_skips: Mapped[list[dict[str, str]] | None] = mapped_column(JSONB, nullable=True)
    final_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    routing_metadata: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    attempts: Mapped[list["AttemptRow"]] = relationship(
        back_populates="request",
        order_by="AttemptRow.attempt_number",
    )
    evaluations: Mapped[list["EvaluationRow"]] = relationship(
        back_populates="request",
        order_by="EvaluationRow.created_at",
    )


class AttemptRow(Base):
    """One provider attempt. Phase 4 writes attempt number 1 only."""

    __tablename__ = "attempts"
    __table_args__ = (
        UniqueConstraint("request_id", "attempt_number", name="uq_attempts_request_attempt"),
        Index("ix_attempts_request_id", "request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    configured_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reported_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_5m_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_1h_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider_latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(14, 3), nullable=True)
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(20, 12), nullable=True)
    cost_completeness: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    pricing_snapshot: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    purpose: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request: Mapped[RequestRow] = relationship(back_populates="attempts")


class EvaluationRow(Base):
    """One quality check. Judge calls are included in the request cost."""

    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_evaluations_request_id", "request_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    judge_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    judge_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    judge_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    judge_cost: Mapped[Decimal | None] = mapped_column(Numeric(20, 12), nullable=True)
    judge_cost_completeness: Mapped[str | None] = mapped_column(String(16), nullable=True)
    judge_latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(14, 3), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request: Mapped[RequestRow] = relationship(back_populates="evaluations")

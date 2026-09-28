"""Create requests and attempts.

Revision ID: 20260928_0001
Revises:
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260928_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("complexity_score", sa.Numeric(8, 6), nullable=False),
        sa.Column("complexity_tier", sa.String(length=16), nullable=False),
        sa.Column("complexity_reasons", postgresql.JSONB(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("selected_model_tier", sa.String(length=16), nullable=False),
        sa.Column("selection_reason", sa.Text(), nullable=False),
        sa.Column("degraded", sa.Boolean(), nullable=False),
        sa.Column("escalated", sa.Boolean(), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("total_cost", sa.Numeric(20, 12), nullable=True),
        sa.Column("cost_completeness", sa.String(length=16), nullable=False),
        sa.Column("end_to_end_latency_ms", sa.Numeric(14, 3), nullable=True),
        sa.Column("premium_baseline_model", sa.String(length=128), nullable=True),
        sa.Column("premium_baseline_cost", sa.Numeric(20, 12), nullable=True),
        sa.Column("estimated_savings", sa.Numeric(20, 12), nullable=True),
    )
    op.create_index("ix_requests_created_at_id", "requests", ["created_at", "id"])
    op.create_table(
        "attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("configured_model_id", sa.String(length=128), nullable=False),
        sa.Column("reported_model_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_write_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_write_5m_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_write_1h_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("provider_latency_ms", sa.Numeric(14, 3), nullable=True),
        sa.Column("estimated_cost", sa.Numeric(20, 12), nullable=True),
        sa.Column("cost_completeness", sa.String(length=16), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("pricing_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["requests.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "request_id",
            "attempt_number",
            name="uq_attempts_request_attempt",
        ),
    )
    op.create_index("ix_attempts_request_id", "attempts", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_attempts_request_id", table_name="attempts")
    op.drop_table("attempts")
    op.drop_index("ix_requests_created_at_id", table_name="requests")
    op.drop_table("requests")

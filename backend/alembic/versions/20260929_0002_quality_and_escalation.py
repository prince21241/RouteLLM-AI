"""Add quality evaluation columns and the evaluations table.

Revision ID: 20260929_0002
Revises: 20260928_0001
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260929_0002"
down_revision: str | None = "20260928_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("quality_verdict", sa.String(length=16), nullable=True))
    op.add_column("requests", sa.Column("quality_score", sa.Numeric(8, 6), nullable=True))
    op.add_column("requests", sa.Column("quality_reasons", postgresql.JSONB(), nullable=True))
    op.add_column("requests", sa.Column("escalation_reason", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("escalation_error", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("final_model_id", sa.String(length=128), nullable=True))
    op.add_column(
        "requests",
        sa.Column("returned_attempt_number", sa.Integer(), nullable=True),
    )
    op.create_table(
        "evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Numeric(8, 6), nullable=True),
        sa.Column("reasons", postgresql.JSONB(), nullable=False),
        sa.Column("judge_model", sa.String(length=128), nullable=True),
        sa.Column("judge_input_tokens", sa.Integer(), nullable=True),
        sa.Column("judge_output_tokens", sa.Integer(), nullable=True),
        sa.Column("judge_cost", sa.Numeric(20, 12), nullable=True),
        sa.Column("judge_cost_completeness", sa.String(length=16), nullable=True),
        sa.Column("judge_latency_ms", sa.Numeric(14, 3), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["requests.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_evaluations_request_id", "evaluations", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_evaluations_request_id", table_name="evaluations")
    op.drop_table("evaluations")
    op.drop_column("requests", "returned_attempt_number")
    op.drop_column("requests", "final_model_id")
    op.drop_column("requests", "escalation_error")
    op.drop_column("requests", "escalation_reason")
    op.drop_column("requests", "quality_reasons")
    op.drop_column("requests", "quality_score")
    op.drop_column("requests", "quality_verdict")

"""Add provider fallback fields on requests and attempts.

Revision ID: 20260929_0003
Revises: 20260929_0002
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260929_0003"
down_revision: str | None = "20260929_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("fallback_used", sa.Boolean(), nullable=True))
    op.add_column("requests", sa.Column("fallback_reason", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("fallback_skips", postgresql.JSONB(), nullable=True))
    op.add_column("requests", sa.Column("final_provider", sa.String(length=32), nullable=True))
    op.add_column("attempts", sa.Column("purpose", sa.String(length=16), nullable=True))
    op.add_column("attempts", sa.Column("error_category", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("attempts", "error_category")
    op.drop_column("attempts", "purpose")
    op.drop_column("requests", "final_provider")
    op.drop_column("requests", "fallback_skips")
    op.drop_column("requests", "fallback_reason")
    op.drop_column("requests", "fallback_used")

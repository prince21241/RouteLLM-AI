"""Store optional ML routing metadata on requests.

Revision ID: 20260929_0004
Revises: 20260929_0003
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260929_0004"
down_revision: str | None = "20260929_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("routing_metadata", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("requests", "routing_metadata")

"""decision product and out-of-view slots

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-04

Adds the product gate result (wire_rod / other / unknown), the number of template slots
outside the camera view, and which slot template produced the count. Server defaults
let the migration run on a database that already holds decisions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("product", sa.String(length=16), nullable=False, server_default="unknown")
        )
        batch_op.add_column(
            sa.Column("out_of_view_slots", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("count_template", sa.String(length=32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.drop_column("count_template")
        batch_op.drop_column("out_of_view_slots")
        batch_op.drop_column("product")

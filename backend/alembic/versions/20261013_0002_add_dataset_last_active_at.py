"""add datasets.last_active_at

Uploaded datasets used to expire a fixed time after creation. With daily
check-ins a dataset lives as long as it is used, so expiry is now measured
from the last activity.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        # The server default keeps inserts from the previous release working while a
        # deploy is rolling out.
        batch_op.add_column(
            sa.Column(
                'last_active_at',
                sa.DateTime(timezone=True),
                nullable=True,
                server_default=sa.func.now(),
            )
        )

    # Existing datasets count as last active when they were created.
    op.execute('UPDATE datasets SET last_active_at = created_at')

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.alter_column(
            'last_active_at', existing_type=sa.DateTime(timezone=True), nullable=False
        )


def downgrade() -> None:
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_column('last_active_at')

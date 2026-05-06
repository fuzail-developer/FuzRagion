import uuid
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # For password-based login
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    subscription_expiry: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    subscription_plan: Mapped[str] = mapped_column(
        String(50), default="free", nullable=False
    )
    chat_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    documents = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )
    magic_tokens = relationship(
        "MagicLinkToken", back_populates="user", cascade="all, delete-orphan"
    )
    chat_histories = relationship(
        "ChatHistory", back_populates="user", cascade="all, delete-orphan"
    )

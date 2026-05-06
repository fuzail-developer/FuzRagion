"""
FastAPI Authentication Router - Magic Link Authentication
Endpoints:
- POST /auth/request-magic-link → Generate & send magic link
- GET /auth/verify-magic-link → Verify token & create JWT
- GET /auth/me → Get current user (protected)
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime, timedelta
import secrets
import os

from models import User, MagicToken
from auth_utils import (
    generate_magic_token,
    send_magic_link_email,
    create_access_token,
    verify_token,
    get_db,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


# Request/Response Models
class RequestMagicLinkRequest(BaseModel):
    email: str


class MagicLinkResponse(BaseModel):
    success: bool
    message: str


class VerifyMagicLinkResponse(BaseModel):
    success: bool
    access_token: str
    token_type: str
    user_id: str
    email: str
    subscription_expiry: datetime | None
    message: str


class CurrentUserResponse(BaseModel):
    id: str
    email: str
    is_active: bool
    subscription_expiry: datetime | None
    subscription_plan: str
    is_subscription_active: bool
    days_until_expiry: int
    created_at: datetime

    class Config:
        from_attributes = True


# Magic Link Token Storage (In-memory for now, use Redis in production)
magic_link_requests = {}  # Track requests for rate limiting


@router.post("/request-magic-link", response_model=MagicLinkResponse)
async def request_magic_link(
    request: RequestMagicLinkRequest, db: Session = Depends(get_db)
):
    """
    Request a magic link for email authentication
    Rate limited: 3 requests per email per hour
    """
    email = request.email.lower().strip()

    # Rate limiting check
    now = datetime.utcnow()
    if email in magic_link_requests:
        requests_in_hour = [
            ts for ts in magic_link_requests[email] if now - ts < timedelta(hours=1)
        ]

        if len(requests_in_hour) >= 3:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Max 3 magic links per hour per email.",
            )

        magic_link_requests[email] = requests_in_hour + [now]
    else:
        magic_link_requests[email] = [now]

    try:
        # Get or create user
        stmt = select(User).where(User.email == email)
        user = db.execute(stmt).scalars().first()

        if not user:
            user = User(email=email, is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Generate magic link token
        token = generate_magic_token()
        magic_token = MagicToken(
            user_id=user.id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(minutes=20),
        )
        db.add(magic_token)
        db.commit()

        # Send email with magic link
        magic_link_url = f"{os.getenv('FRONTEND_URL', 'http://127.0.0.1:8000')}/auth/verify?token={token}"
        await send_magic_link_email(email, magic_link_url)

        return MagicLinkResponse(
            success=True,
            message=f"Magic link sent to {email}. Check your email (valid for 20 minutes).",
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to send magic link: {str(e)}"
        )


@router.get("/verify-magic-link", response_model=VerifyMagicLinkResponse)
async def verify_magic_link(
    token: str = Query(..., description="Magic link token"),
    db: Session = Depends(get_db),
):
    """
    Verify magic link token and issue JWT access token
    Token is one-time use and expires after 20 minutes
    """

    # Find magic token
    stmt = select(MagicToken).where(MagicToken.token == token)
    magic_token = db.execute(stmt).scalars().first()

    if not magic_token:
        raise HTTPException(status_code=400, detail="Invalid magic link token.")

    # Check if token is still valid
    if not magic_token.is_valid():
        raise HTTPException(
            status_code=400,
            detail="Magic link expired or already used. Please request a new one.",
        )

    # Mark token as used (one-time use)
    magic_token.mark_as_used()
    db.commit()

    # Get user
    user = magic_token.user

    if not user.is_active:
        raise HTTPException(
            status_code=403, detail="Your account is inactive. Please contact support."
        )

    # Create JWT access token (valid for 7 days)
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}, expires_delta=timedelta(days=7)
    )

    return VerifyMagicLinkResponse(
        success=True,
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        email=user.email,
        subscription_expiry=user.subscription_expiry,
        message="Successfully authenticated! Redirecting...",
    )


@router.get("/me", response_model=CurrentUserResponse)
async def get_current_user(current_user: User = Depends(lambda db: None)):
    """
    Get current authenticated user information
    This endpoint is protected by JWT token
    """
    # This will be properly protected in dependencies.py
    return CurrentUserResponse(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        subscription_expiry=current_user.subscription_expiry,
        subscription_plan=current_user.subscription_plan,
        is_subscription_active=current_user.is_subscription_active(),
        days_until_expiry=current_user.days_until_expiry(),
        created_at=current_user.created_at,
    )

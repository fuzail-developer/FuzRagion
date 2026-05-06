import os
import re
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash, check_password_hash
from auth.dependencies import get_current_user
from auth.utils import (
    MAGIC_LINK_TTL_MINUTES,
    create_access_token,
    generate_magic_link_token,
    hash_magic_token,
    send_magic_link_email,
)
from dependencies.database import get_db
from models import MagicLinkRequestLog, MagicLinkToken, User

router = APIRouter(prefix="/auth", tags=["auth"])
MONTHLY_SUBSCRIPTION_INR = 9999
RAZORPAY_PAYMENT_LINK = "https://razorpay.me/@afreen9836"


class RequestMagicLinkBody(BaseModel):
    email: EmailStr


class AuthMeResponse(BaseModel):
    id: str
    email: str
    name: str | None = None
    phone_number: str | None = None
    is_active: bool
    subscription_plan: str
    subscription_expiry: datetime | None
    chat_count: int
    created_at: datetime
    updated_at: datetime


class ActivateSubscriptionBody(BaseModel):
    amount_inr: int
    payment_reference: str
    activation_token: str


class SignupBody(BaseModel):
    name: str
    email: EmailStr
    phone_number: str
    password: str


class LoginBody(BaseModel):
    email: EmailStr
    password: str


def normalize_phone_number(phone_number: str) -> str:
    phone = phone_number.strip()
    if not re.fullmatch(r"^\+?[0-9]{10,15}$", phone):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Phone number must be 10-15 digits (optional leading +).",
        )
    return phone


@router.post("/request-magic-link")
def request_magic_link(payload: RequestMagicLinkBody, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = now - timedelta(hours=1)

    count = (
        db.query(MagicLinkRequestLog)
        .filter(MagicLinkRequestLog.email == email)
        .filter(MagicLinkRequestLog.requested_at >= window_start)
        .count()
    )
    if count >= 3:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Max 3 magic links per hour per email.",
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, is_active=True)
        db.add(user)
        db.flush()

    raw_token = generate_magic_link_token()
    token_hash = hash_magic_token(raw_token)
    expires_at = now + timedelta(minutes=MAGIC_LINK_TTL_MINUTES)

    db.add(
        MagicLinkToken(
            user_id=user.id, token_hash=token_hash, expires_at=expires_at, is_used=False
        )
    )
    db.add(MagicLinkRequestLog(email=email, requested_at=now))
    db.commit()

    frontend_url = os.getenv("FRONTEND_URL", "http://127.0.0.1:8000")
    magic_link = f"{frontend_url.rstrip('/')}/auth/verify?token={raw_token}"
    send_magic_link_email(email, magic_link)

    return {
        "success": True,
        "message": "Magic link sent successfully. It expires in 20 minutes.",
    }


@router.get("/verify-magic-link")
def verify_magic_link(token: str = Query(...), db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    token_hash = hash_magic_token(token)

    record = (
        db.query(MagicLinkToken).filter(MagicLinkToken.token_hash == token_hash).first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token."
        )
    if record.is_used or record.expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token expired or already used.",
        )

    record.is_used = True
    record.used_at = now
    user = db.query(User).filter(User.id == record.user_id).first()
    db.commit()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User inactive."
        )

    try:
        access_token = create_access_token(
            {"sub": user.id, "email": user.email}, expires_delta=timedelta(days=7)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth config missing: JWT_SECRET_KEY",
        ) from exc
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "is_active": user.is_active,
            "subscription_expiry": user.subscription_expiry,
        },
    }


@router.get("/me", response_model=AuthMeResponse)
def auth_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if (
        current_user.subscription_plan == "pro"
        and current_user.subscription_expiry
        and current_user.subscription_expiry <= now
    ):
        current_user.subscription_plan = "free"
        db.commit()
        db.refresh(current_user)
    return current_user


@router.post("/activate-subscription")
def activate_subscription(
    payload: ActivateSubscriptionBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_activation_token = os.getenv("SUBSCRIPTION_ACTIVATION_TOKEN", "").strip()
    if not required_activation_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription activation is locked. Ask admin to configure SUBSCRIPTION_ACTIVATION_TOKEN.",
        )

    if payload.amount_inr != MONTHLY_SUBSCRIPTION_INR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ?9,999 monthly subscription is allowed. Please pay exactly ?9,999.",
        )
    if not payload.payment_reference.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment reference is required.",
        )
    if payload.activation_token.strip() != required_activation_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid activation token. Premium cannot be activated without verified payment.",
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    current_expiry = current_user.subscription_expiry
    if current_expiry and current_expiry > now:
        current_user.subscription_expiry = current_expiry + timedelta(days=30)
    else:
        current_user.subscription_expiry = now + timedelta(days=30)
    current_user.subscription_plan = "pro"

    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return {
        "success": True,
        "message": "Subscription activated for 30 days.",
        "subscription_expiry": current_user.subscription_expiry,
        "subscription_plan": current_user.subscription_plan,
        "amount_inr": MONTHLY_SUBSCRIPTION_INR,
        "payment_link": RAZORPAY_PAYMENT_LINK,
    }


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: SignupBody, db: Session = Depends(get_db)):
    """Create a new user account with email and password"""
    email = payload.email.lower().strip()

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered. Please login.",
        )

    hashed_password = generate_password_hash(payload.password, method="pbkdf2:sha256")
    phone_number = normalize_phone_number(payload.phone_number)
    new_user = User(
        email=email,
        password_hash=hashed_password,
        is_active=True,
        name=payload.name.strip(),
        phone_number=phone_number,
        subscription_expiry=None,
        subscription_plan="free",
        chat_count=0,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    try:
        access_token = create_access_token(
            {"sub": str(new_user.id), "email": new_user.email},
            expires_delta=timedelta(days=7),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth config missing: JWT_SECRET_KEY",
        ) from exc

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(new_user.id),
            "email": new_user.email,
            "name": new_user.name,
            "phone_number": new_user.phone_number,
            "is_active": new_user.is_active,
            "subscription_expiry": new_user.subscription_expiry,
            "subscription_plan": new_user.subscription_plan,
            "chat_count": new_user.chat_count,
        },
    }


@router.post("/login")
def login(payload: LoginBody, db: Session = Depends(get_db)):
    """Login with email and password"""
    email = payload.email.lower().strip()

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.password_hash or not check_password_hash(
        user.password_hash, payload.password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    if not user.subscription_plan:
        user.subscription_plan = "free"
        db.commit()

    try:
        access_token = create_access_token(
            {"sub": str(user.id), "email": user.email}, expires_delta=timedelta(days=7)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth config missing: JWT_SECRET_KEY",
        ) from exc

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "phone_number": user.phone_number,
            "is_active": user.is_active,
            "subscription_expiry": user.subscription_expiry,
            "subscription_plan": user.subscription_plan,
            "chat_count": user.chat_count,
        },
    }

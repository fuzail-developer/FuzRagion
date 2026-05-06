from datetime import datetime, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from auth.utils import decode_access_token
from dependencies.database import get_db
from models import User

bearer_scheme = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user


def check_active_subscription(current_user: User = Depends(get_current_user)) -> User:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if not current_user.subscription_expiry or current_user.subscription_expiry <= now:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your subscription has expired. Please renew for ₹9,999 using this link: https://razorpay.me/@afreen9836",
        )
    return current_user

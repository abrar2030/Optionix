from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def authenticate(token: str = Depends(oauth2_scheme)):
    if not valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return token


def valid_token(token: str) -> bool:
    # Implement JWT validation logic
    return True

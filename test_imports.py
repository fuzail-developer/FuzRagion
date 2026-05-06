#!/usr/bin/env python3
import sys

print("Step 1: Importing FastAPI...")
sys.stdout.flush()
from fastapi import FastAPI

print("Step 2: Importing models...")
sys.stdout.flush()
from models import User

print("Step 3: Importing auth.router...")
sys.stdout.flush()
from auth.router import router as auth_router

print("Step 4: Creating app...")
sys.stdout.flush()
app = FastAPI()
app.include_router(auth_router)

print("✓ All imports successful!")
sys.stdout.flush()

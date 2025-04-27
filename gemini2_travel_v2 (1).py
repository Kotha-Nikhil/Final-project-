
"""
gemini2_travel_v2.py – FastAPI backend for the Gemini + CrewAI travel planner
Single-file edition patched to read API keys from:
  • st.secrets         (Streamlit Cloud)
  • environment vars   (GOOGLE_API_KEY / SERP_API_KEY)
  • .env file via python-dotenv (for local runs)
"""

# ───── Imports ────────────────────────────────────────────────────────────
import os
import asyncio
import logging
from functools import lru_cache
from datetime import datetime

import streamlit as st            # NEW – for st.secrets
from dotenv import load_dotenv    # NEW – enable `.env` for local dev
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from serpapi import GoogleSearch
from crewai import Agent, Task, Crew, Process, LLM
import uvicorn

# Load local .env if present (ignored on Streamlit Cloud)
load_dotenv()

# ───── API-key handling ───────────────────────────────────────────────────
GEMINI_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY")      # ① Streamlit Cloud
    or os.getenv("GOOGLE_API_KEY", "")    # ② Real env var / .env
)
SERP_API_KEY = (
    st.secrets.get("SERP_API_KEY")
    or os.getenv("SERP_API_KEY", "")
)
if not GEMINI_API_KEY or not SERP_API_KEY:
    logging.warning("One or both API keys are missing – functionality will be limited.")

# ───── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ───── LLM initialiser (cached) ───────────────────────────────────────────
@lru_cache(maxsize=1)
def initialize_llm():
    """Create a singleton Gemini LLM instance."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=GEMINI_API_KEY,
    )

# ───── Pydantic models ────────────────────────────────────────────────────
class FlightRequest(BaseModel):
    origin: str
    destination: str
    outbound_date: str
    return_date: str

class HotelRequest(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str

class ItineraryRequest(BaseModel):
    destination: str
    check_in_date: str
    check_out_date: str
    flights: str
    hotels: str

class FlightInfo(BaseModel):
    airline: str
    price: str
    duration: str
    stops: str
    departure: str
    arrival: str
    travel_class: str
    return_date: str
    airline_logo: str

class HotelInfo(BaseModel):
    name: str
    price: str
    rating: float
    location: str
    link: str

class AIResponse(BaseModel):
    flights: List[FlightInfo] = []
    hotels: List[HotelInfo] = []
    ai_flight_recommendation: str = ""
    ai_hotel_recommendation: str = ""
    itinerary: str = ""

# ───── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planning API", version="1.2.0")

# helper functions (same as provided earlier) ...

"""
gemini2_travel_v2.py – FastAPI backend for the Gemini + CrewAI travel planner
Patched for Streamlit Cloud:
  • Fixes Chroma-DB → SQLite issue via pysqlite3-binary
  • Loads API keys from st.secrets → env vars → .env
"""

# ───────────────────────────────────────────────────────────────────────────
# ❶ Chroma / SQLite fix  (MUST be first – before crewai / chromadb imports)
# ───────────────────────────────────────────────────────────────────────────
import sys

try:
    # pysqlite3 ships a modern SQLite (≥ 3.42)
    import pysqlite3 as sqlite3          # needs pysqlite3-binary in requirements.txt
    sys.modules["sqlite3"] = sqlite3     # make all future `import sqlite3` use this
except ImportError:
    # local runs where pysqlite3 isn’t installed will fall back to stdlib sqlite3
    pass

# ───────────────────────────────────────────────────────────────────────────
# ❷ Standard & third-party imports
# ───────────────────────────────────────────────────────────────────────────
import os
import asyncio
import logging
from functools import lru_cache
from datetime import datetime
from typing import List, Optional

import streamlit as st            # for st.secrets
from dotenv import load_dotenv    # to read local .env
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from serpapi import GoogleSearch
from crewai import Agent, Task, Crew, Process, LLM
import uvicorn

# ───────────────────────────────────────────────────────────────────────────
# ❸ Environment / secrets
# ───────────────────────────────────────────────────────────────────────────
load_dotenv()  # read .env if present (ignored on Streamlit Cloud)

GEMINI_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY") or
    os.getenv("GOOGLE_API_KEY", "")
)
SERP_API_KEY = (
    st.secrets.get("SERP_API_KEY") or
    os.getenv("SERP_API_KEY", "")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s – %(message)s",
)
logger = logging.getLogger(__name__)

if not GEMINI_API_KEY or not SERP_API_KEY:
    logger.warning("One or both API keys are missing – API calls may fail.")

# ───────────────────────────────────────────────────────────────────────────
# ❹ LLM initialiser (cached)
# ───────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def initialize_llm():
    """Return a singleton Gemini LLM instance."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=GEMINI_API_KEY,
    )

# ───────────────────────────────────────────────────────────────────────────
# ❺ Pydantic schemas
# ───────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────
# ❻ FastAPI app
# ───────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planning API", version="1.2.1")

# ───────────────────────────────────────────────────────────────────────────
# ❼ Helper: run SerpAPI in a thread
# ───────────────────────────────────────────────────────────────────────────
async def _run_search(params):
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
    except Exception as e:
        logger.exception(f"SerpAPI error: {e}")
        raise HTTPException(status_code=500, detail=f"SerpAPI error: {e}")

# ───────────────────────────────────────────────────────────────────────────
# ❽ Flight & hotel searches
# ───────────────────────────────────────────────────────────────────────────
async def search_flights(req: FlightRequest):
    logger.info(f"Searching flights {req.origin} → {req.destination}")
    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": req.origin.strip().upper(),
        "arrival_id":   req.destination.strip().upper(),
        "outbound_date": req.outbound_date,
        "return_date":   req.return_date,
        "currency": "USD",
    }
    data = await _run_search(params)
    if "error" in data:
        return {"error": data["error"]}

    best = data.get("best_flights", [])
    flights: List[FlightInfo] = []
    for f in best:
        if not f.get("flights"):
            continue
        leg = f["flights"][0]
        flights.append(
            FlightInfo(
                airline      = leg.get("airline", "Unknown"),
                price        = str(f.get("price", "N/A")),
                duration     = f"{f.get('total_duration', 'N/A')} min",
                stops        = "Nonstop" if len(f["flights"]) == 1 else f"{len(f['flights'])-1} stop(s)",
                departure    = f"{leg.get('departure_airport', {}).get('name','?')} "
                               f"({leg.get('departure_airport', {}).get('id','?')}) at "
                               f"{leg.get('departure_airport', {}).get('time','?')}",
                arrival      = f"{leg.get('arrival_airport', {}).get('name','?')} "
                               f"({leg.get('arrival_airport', {}).get('id','?')}) at "
                               f"{leg.get('arrival_airport', {}).get('time','?')}",
                travel_class = leg.get("travel_class", "Economy"),
                return_date  = req.return_date,
                airline_logo = leg.get("airline_logo", ""),
            )
        )
    return flights

async def search_hotels(req: HotelRequest):
    logger.info(f"Searching hotels in {req.location}")
    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": req.location,
        "hl": "en",
        "gl": "us",
        "check_in_date":  req.check_in_date,
        "check_out_date": req.check_out_date,
        "currency": "USD",
        "sort_by": 3,
        "rating": 8,
    }
    data = await _run_search(params)
    if "error" in data:
        return {"error": data["error"]}

    props = data.get("properties", [])
    return [
        HotelInfo(
            name     = h.get("name", "Unknown Hotel"),
            price    = h.get("rate_per_night", {}).get("lowest", "N/A"),
            rating   = h.get("overall_rating", 0.0),
            location = h.get("location", "N/A"),
            link     = h.get("link", "N/A"),
        )
        for h in props
    ]

# ───────────────────────────────────────────────────────────────────────────
# ❾ Text formatters & AI helpers
# ───────────────────────────────────────────────────────────────────────────
def _format(kind, data):
    if not data:
        return f"No {kind} available."
    if kind == "flights":
        out = "✈️ **Available flight options**:\n\n"
        for i, f in enumerate(data):
            out += (
                f"**Flight {i+1}:**\n"
                f"✈️ **Airline:** {f.airline}\n"
                f"💰 **Price:** ${f.price}\n"
                f"⏱️ **Duration:** {f.duration}\n"
                f"🛑 **Stops:** {f.stops}\n"
                f"🕔 **Departure:** {f.departure}\n"
                f"🕖 **Arrival:** {f.arrival}\n"
                f"💺 **Class:** {f.travel_class}\n\n"
            )
    else:  # hotels
        out = "🏨 **Available hotel options**:\n\n"
        for i, h in enumerate(data):
            out += (
                f"**Hotel {i+1}:**\n"
                f"🏨 **Name:** {h.name}\n"
                f"💰 **Price:** ${h.price}\n"
                f"⭐ **Rating:** {h.rating}\n"
                f"📍 **Location:** {h.location}\n"
                f"🔗 **More Info:** [Link]({h.link})\n\n"
            )
    return out.strip()

async def _recommend(kind, formatted):
    llm = initialize_llm()
    if kind == "flights":
        role, goal = "AI Flight Analyst", "Pick the best flight."
    else:
        role, goal = "AI Hotel Analyst", "Pick the best hotel."
    agent = Agent(role=role, goal=goal, backstory="Expert travel analyst.", llm=llm, verbose=False)
    desc = f"Analyze these {kind} and recommend the best one:\n\n{formatted}"
    task = Task(description=desc, agent=agent,
                expected_output=f"A concise recommendation for the best {kind[:-1]}.")
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    res = await asyncio.to_thread(crew.kickoff)
    return res.outputs[0] if hasattr(res, "outputs") and res.outputs else str(res)

async def _itinerary(dest, flights_txt, hotels_txt, check_in, check_out):
    days = (datetime.strptime(check_out, "%Y-%m-%d") -
            datetime.strptime(check_in, "%Y-%m-%d")).days
    llm = initialize_llm()
    planner = Agent(role="AI Travel Planner", goal="Create itineraries",
                    backstory="Expert trip designer.", llm=llm, verbose=False)
    task = Task(
        description=f"Create a {days}-day itinerary for **{dest}**.\n\n"
                    f"Flights:\n{flights_txt}\n\nHotels:\n{hotels_txt}",
        agent=planner,
        expected_output="A markdown itinerary.",
    )
    crew = Crew(agents=[planner], tasks=[task], process=Process.sequential, verbose=False)
    res = await asyncio.to_thread(crew.kickoff)
    return res.outputs[0] if hasattr(res, "outputs") and res.outputs else str(res)

# ───────────────────────────────────────────────────────────────────────────
# ❿ API endpoints
# ───────────────────────────────────────────────────────────────────────────
@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(req: FlightRequest):
    flights = await search_flights(req)
    if isinstance(flights, dict):
        raise HTTPException(status_code=400, detail=flights["error"])
    if not flights:
        raise HTTPException(status_code=404, detail="No flights found")

    txt = _format("flights", flights)
    rec = await _recommend("flights", txt)
    return AIResponse(flights=flights, ai_flight_recommendation=rec)

@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(req: HotelRequest):
    hotels = await search_hotels(req)
    if isinstance(hotels, dict):
        raise HTTPException(status_code=400, detail=hotels["error"])
    if not hotels:
        raise HTTPException(status_code=404, detail="No hotels found")

    txt = _format("hotels", hotels)
    rec = await _recommend("hotels", txt)
    return AIResponse(hotels=hotels, ai_hotel_recommendation=rec)

@app.post("/complete_search/", response_model=AIResponse)
async def complete_travel_search(flight_request: FlightRequest,
                                 hotel_request: Optional[HotelRequest] = None):
    if hotel_request is None:
        hotel_request = HotelRequest(
            location=flight_request.destination,
            check_in_date=flight_request.outbound_date,
            check_out_date=flight_request.return_date,
        )

    f_task = asyncio.create_task(get_flight_recommendations(flight_request))
    h_task = asyncio.create_task(get_hotel_recommendations(hotel_request))
    flights, hotels = await asyncio.gather(f_task, h_task)

    iti = ""
    if flights.flights and hotels.hotels:
        iti = await _itinerary(
            flight_request.destination,
            _format("flights", flights.flights),
            _format("hotels",  hotels.hotels),
            flight_request.outbound_date,
            flight_request.return_date,
        )

    return AIResponse(
        flights=flights.flights,
        hotels=hotels.hotels,
        ai_flight_recommendation=flights.ai_flight_recommendation,
        ai_hotel_recommendation=hotels.ai_hotel_recommendation,
        itinerary=iti,
    )

@app.post("/generate_itinerary/", response_model=AIResponse)
async def generate_itinerary(req: ItineraryRequest):
    iti = await _itinerary(
        req.destination, req.flights, req.hotels,
        req.check_in_date, req.check_out_date
    )
    return AIResponse(itinerary=iti)

# ───────────────────────────────────────────────────────────────────────────
# ⓫ Local dev entry-point
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Running FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

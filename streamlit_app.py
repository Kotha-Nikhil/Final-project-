
# streamlit_app.py – Streamlit UI calling backend functions directly
#  (v2.2.0)

import streamlit as st
import asyncio
from datetime import datetime, timedelta

from gemini2_travel_v2 import (
    FlightRequest, HotelRequest,
    get_flight_recommendations, get_hotel_recommendations,
    complete_travel_search
)

# ... rest of code omitted for brevity ...

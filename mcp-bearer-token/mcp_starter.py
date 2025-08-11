# api/mcp.py

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# --- Import required libraries ---
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# --- Firestore Imports ---
from google.cloud import firestore
import google.auth.credentials
import json
import base64

# --- FastAPI App ---
# We use the standard FastAPI app, which is fully compatible with Vercel.
app = FastAPI()

# --- Environment Variables ---
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
FIRESTORE_CREDS_B64 = os.environ.get("FIRESTORE_CREDS_B64")

if not all([TOKEN, MY_NUMBER, FIRESTORE_CREDS_B64]):
    raise RuntimeError("AUTH_TOKEN, MY_NUMBER, and FIRESTORE_CREDS_B64 must be set.")

SERVER_ID = f"mcp.puch.ai:server:{MY_NUMBER}"

# --- OPTIMIZATION: Lazy-Loaded Firestore Client ---
db = None
def get_db_client():
    global db
    if db is None:
        try:
            creds_json_str = base64.b64decode(FIRESTORE_CREDS_B64).decode('utf-8')
            creds_info = json.loads(creds_json_str)
            credentials = google.oauth2.service_account.Credentials.from_service_account_info(creds_info)
            db = firestore.Client(credentials=credentials)
            print("Firestore client initialized on first request.")
        except Exception as e:
            print(f"CRITICAL: Failed to initialize Firestore client: {e}")
    return db

# --- Pydantic Models for structured request handling ---
class User(BaseModel):
    id: str
    name: Optional[str] = None

class Message(BaseModel):
    user: User

class ToolRunRequest(BaseModel):
    message: Message
    parameters: dict[str, Any]

# --- Helper Function to Parse Workout String ---
def parse_workout_string(log_string: str) -> dict | None:
    pattern = re.compile(
        r"^(?P<name>[\w\s]+?)\s+"
        r"(?P<weight>[\d\.]+)"
        r"(?:\s*x\s*(?P<per_side>2))?"
        r"\s*x\s*(?P<sets>[\d]+)"
        r"\s*x\s*(?P<reps>[\d]+)$",
        re.IGNORECASE
    )
    match = pattern.match(log_string.strip())
    if not match:
        simple_pattern = re.compile(
            r"^(?P<name>[\w\s]+?)\s+"
            r"(?P<weight>[\d\.]+)"
            r"\s*x\s*(?P<reps>[\d]+)$",
            re.IGNORECASE
        )
        match = simple_pattern.match(log_string.strip())
        if not match:
            return None
    data = match.groupdict()
    return {
        "name": data["name"].strip().title(),
        "weight": float(data["weight"]),
        "sets": int(data.get("sets") or 1),
        "reps": int(data["reps"]),
        "per_side": data.get("per_side") is not None
    }

# --- MANIFEST ENDPOINT ---
@app.api_route("/", methods=["GET", "POST"])
async def get_manifest() -> dict[str, Any]:
    return {
        "mcp_version": "1.0",
        "server_id": SERVER_ID,
        "name": "Workout Logger",
        "author": "You!",
        "description": "A server to log workouts and track progress over time with graphs.",
        "auth": {"auth_type": "http_bearer"},
        "tools": {
            "validate": {
                "description": "Validates the server connection.",
                "parameters": [],
                "returns": [{"type": "string"}]
            },
            "greet": {
                "description": "Greets the user and lists available commands. Use this when the user sends a greeting like 'hi', 'hello', or asks for 'help'.",
                "parameters": [],
                "returns": [{"type": "text"}]
            },
            "log_workout": {
                "description": "Logs a workout entry into the user's personal database. Use this when the user says 'log', 'add', or 'save' a workout. Example format: 'Squat 100x5x5' or 'Incline Curl 12.5x2x8'.",
                "parameters": [
                    {"name": "entry", "type": "string", "description": "The workout string to log.", "required": True}
                ],
                "returns": [{"type": "text"}]
            },
            "view_progress": {
                "description": "Shows a user's personal, saved workout history and a progress graph for a specific exercise from the database. Use this when the user asks to 'see', 'view', 'show', or 'check' their logs, history, or progress for an exercise.",
                "parameters": [
                    {"name": "exercise", "type": "string", "description": "The name of the exercise to view progress for.", "required": True}
                ],
                "returns": [{"type": "text"}, {"type": "image"}]
            }
        }
    }

# --- Tool: validate ---
@app.api_route("/run/validate", methods=["GET", "POST"])
async def run_validate(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return MY_NUMBER

# --- Tool: greet ---
@app.post("/run/greet")
async def run_greet(tool_request: ToolRunRequest):
    user_name = tool_request.message.user.name or "there"
    welcome_message = (
        f"Hi {user_name}! I'm your personal workout logger.\n\n"
        "Here's what you can do:\n\n"
        "1️⃣ **Log a workout:**\n"
        "   - `log Bench Press 60x5x5`\n"
        "   - `add Squat 100x3x8`\n\n"
        "2️⃣ **View your progress:**\n"
        "   - `show my progress for Bench Press`\n"
        "   - `view history for Squat`"
    )
    return [{"type": "text", "text": welcome_message}]

# --- Tool: log_workout ---
@app.post("/run/log_workout")
async def run_log_workout(tool_request: ToolRunRequest):
    db_client = get_db_client()
    if not db_client:
        return [{"type": "text", "text": "Error: Database is not configured correctly."}]

    entry = tool_request.parameters.get("entry")
    if not entry:
        return [{"type": "text", "text": "Sorry, I didn't get a workout to log."}]

    parsed_data = parse_workout_string(entry)
    if not parsed_data:
        return [{"type": "text", "text": f"Sorry, I couldn't understand that format. Try something like 'Bench Press 60x5x5'."}]

    user_id = tool_request.message.user.id
    parsed_data["user_id"] = user_id
    parsed_data["timestamp"] = datetime.now(timezone.utc)

    try:
        db_client.collection("workouts").add(parsed_data)
        log_msg = (
            f"💪 Logged: {parsed_data['name']}!\n"
            f"- Weight: {parsed_data['weight']} kg"
            f"{' (per side)' if parsed_data['per_side'] else ''}\n"
            f"- Sets: {parsed_data['sets']}\n"
            f"- Reps: {parsed_data['reps']}"
        )
        return [{"type": "text", "text": log_msg}]
    except Exception as e:
        return [{"type": "text", "text": f"Sorry, there was an error saving your workout: {e}"}]

# --- Tool: view_progress ---
@app.post("/run/view_progress")
async def run_view_progress(tool_request: ToolRunRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    db_client = get_db_client()
    if not db_client:
        return [{"type": "text", "text": "Error: Database is not configured correctly."}]

    user_id = tool_request.message.user.id
    exercise = tool_request.parameters.get("exercise")
    if not exercise:
        return [{"type": "text", "text": "Sorry, I didn't get an exercise name to look up."}]

    exercise_name = exercise.strip().title()

    try:
        docs = db_client.collection("workouts") \
            .where("user_id", "==", user_id) \
            .where("name", "==", exercise_name) \
            .order_by("timestamp", direction=firestore.Query.ASCENDING) \
            .limit(20) \
            .stream()
        
        logs = list(docs)
        if not logs:
            return [{"type": "text", "text": f"No logs found for '{exercise_name}'. Try logging one first!"}]

        summary_text = f"📈 Progress for {exercise_name}:\n\n"
        ist = timezone(timedelta(hours=5, minutes=30))

        for log in logs[-5:]:
            data = log.to_dict()
            timestamp_ist = data['timestamp'].astimezone(ist)
            date_str = timestamp_ist.strftime("%b %d")
            log_str = (f"- *{date_str}*: {data['weight']}kg x {data['sets']}s x {data['reps']}r")
            summary_text += log_str + "\n"

        dates = [log.to_dict()['timestamp'].astimezone(ist).strftime("%d-%b") for log in logs]
        weights = [log.to_dict()['weight'] for log in logs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, weights, marker='o', linestyle='-', color='b')
        ax.set_title(f"Weight Progression for {exercise_name}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weight (kg)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return [
            {"type": "text", "text": summary_text},
            {"type": "image", "mimeType": "image/png", "data": image_base64}
        ]
    except Exception as e:
        return [{"type": "text", "text": f"Sorry, there was an error fetching your progress: {e}"}]
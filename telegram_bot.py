# OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE" # You can remove or comment this out


import logging
import os
import re
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import dateparser # Still useful for parsing date strings from LLM output
from ics import Calendar, Event
from dotenv import load_dotenv
from telegram import Update, ForceReply # ForceReply might not be needed with webhooks
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, ExtBot

# --- LLM specific imports ---
import google.generativeai as genai # For Gemini

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# --- Initialize Gemini Client ---
genai.configure(api_key=GOOGLE_API_KEY)
# For safety configuration, see:
# https://ai.google.dev/gemini-api/docs/safety-settings
generation_config = {
  "temperature": 0.2, # Lower for more deterministic output
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048, # Adjust as needed
}
safety_settings = [ # Example: block fewer things for this use case
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
# Choose your model. 'gemini-1.0-pro' is a good general-purpose one.
# 'gemini-1.5-flash-latest' is faster and cheaper for simpler tasks.
# 'gemini-1.5-pro-latest' is the most capable.
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-1.0-pro"
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_gemini_event_extraction_prompt(user_message: str) -> str:
    """
    Creates a prompt for Gemini to extract event details.
    """
    prompt = f"""
    You are an expert assistant that extracts event details from user messages.
    Analyze the following user message and extract information to create a calendar event.
    All times should be interpreted as Israel Standard Time (IST/UTC+2 or IDT/UTC+3 during daylight saving).
    
    Provide the output strictly as a JSON object with the following keys:
    - "title": string (The main subject or name of the event. Be concise.)
    - "start_time_str": string (The start date and time, e.g., "tomorrow 3 PM", "July 20th 10am", "2024-12-25 17:00". If a year is not specified, assume the current year or next year if the date has passed.)
    - "end_time_str": string (The end date and time. If only a duration is given (e.g., "for 1 hour"), calculate and provide this. If not specified and no duration, set to null.)
    - "duration_str": string (The duration of the event, e.g., "1 hour", "30 minutes". If an explicit end_time_str is found, this can be null or you can calculate it.)
    - "location": string (The physical location of the event. Set to null if not mentioned.)
    - "reminder": integer (Minutes before the event to send a reminder. Look for phrases like "remind me 10 minutes before", "with a reminder 30 minutes before", etc. If not specified, set to null.)
    - "timezone": "Asia/Jerusalem" (Always use Israel time zone)

    Important rules for your JSON output:
    1. Only output the JSON object. Do not include any explanatory text before or after the JSON.
    2. If a piece of information is not found, use null for its value in the JSON (e.g., "location": null).
    3. If only a start time is given and no explicit end time or duration, assume a 1-hour duration and calculate "end_time_str" accordingly. "duration_str" should then be "1 hour".
    4. If a start time and a duration are given (e.g., "meeting tomorrow 2pm for 2 hours"), calculate "end_time_str".
    5. Be precise with date and time strings. Try to include AM/PM or use 24-hour format if it's clear from the input.
    6. Always treat times as Israel time (IST/IDT).
    7. The user date is typically in the form of DD/MM/YYYY or DD.MM.YY (example formats), but other formats may also be supported.
    8. The user always give the date in european format (DD/MM/YYYY) or DD.MM.YY (example formats).


    User message: "{user_message}"

    JSON Output:
    """
    return prompt

async def parse_event_from_text_gemini(text: str) -> tuple[str | None, datetime | None, datetime | None, str | None, int | None]:
    """
    Uses Gemini to extract event details and then parses date strings.
    Returns: (title, start_dt, end_dt, location, reminder)
    """
    global llm_output_text, response
    prompt = get_gemini_event_extraction_prompt(text)
    extracted_data = None

    try:
        # --- Gemini API Call ---
        # Use `generate_content_async` for asyncio compatibility if available and needed,
        # otherwise `generate_content` will run synchronously.
        # For this telegram bot structure, synchronous call within the async handler is often fine.
        response = await gemini_model.generate_content_async(prompt) # Use async version
        # response = gemini_model.generate_content(prompt) # Synchronous version

        # Gemini's response.text should contain the JSON.
        # It's good practice to check `response.prompt_feedback` for safety blocks.
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.error(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings:
                    logger.error(f"  Safety Rating: {rating.category} - {rating.probability}")
            return None, None, None, None, None

        llm_output_text = response.text.strip()
        logger.info(f"Gemini Raw Output: {llm_output_text}")

        # Try to parse the LLM output as JSON
        # Gemini is usually good at returning just JSON if prompted correctly.
        # Remove potential markdown ```json ...
        if llm_output_text.startswith("```json"):
            llm_output_text = llm_output_text[7:]
        if llm_output_text.endswith("```"):
            llm_output_text = llm_output_text[:-3]
        llm_output_text = llm_output_text.strip()

        extracted_data = json.loads(llm_output_text)

    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Gemini prompt blocked due to safety settings: {e}")
        return None, None, None, None, None
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error from Gemini output: '{llm_output_text}'. Error: {e}")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Error calling Gemini or parsing its response: {e}")
        # You might want to inspect `response.candidates[0].finish_reason` if available
        # and `response.candidates[0].safety_ratings`
        if hasattr(response, 'candidates') and response.candidates:
            logger.error(f"Candidate finish reason: {response.candidates[0].finish_reason}")
            for rating in response.candidates[0].safety_ratings:
                 logger.error(f"  Safety Rating: {rating.category} - {rating.probability}")
        return None, None, None, None, None

    if not extracted_data:
        return None, None, None, None, None

    title = extracted_data.get("title")
    start_time_str = extracted_data.get("start_time_str")
    end_time_str = extracted_data.get("end_time_str")
    duration_str = extracted_data.get("duration_str")
    location = extracted_data.get("location")
    reminder = extracted_data.get("reminder")

    dateparser_settings = {
        'PREFER_DATES_FROM': 'future',
        'TIMEZONE': 'Asia/Jerusalem',  # Use Israel time zone
        'RETURN_AS_TIMEZONE_AWARE': True,
        'DATE_ORDER': 'DMY',  # Add this line to explicitly use day-month-year format
        'RELATIVE_BASE': datetime.now(timezone.utc).astimezone(
            dateparser.timezone_parser.StaticTzInfo('Asia/Jerusalem', timedelta(hours=3))
        )  # Use Israel time zone (+3 during DST)
    }

    start_dt = None
    end_dt = None

    if start_time_str:
        start_dt = dateparser.parse(start_time_str, settings=dateparser_settings)
        if not start_dt:
             logger.warning(f"dateparser failed to parse start_time_str from Gemini: '{start_time_str}'")

    if end_time_str:
        end_dt = dateparser.parse(end_time_str, settings=dateparser_settings)
        if not end_dt:
            logger.warning(f"dateparser failed to parse end_time_str from Gemini: '{end_time_str}'")

    # If Gemini didn't calculate end_dt but gave duration_str
    if start_dt and not end_dt and duration_str:
        # Try to parse duration like "1 hour", "30 minutes"
        # A simple regex might be more reliable here than dateparser for durations
        duration_match = re.match(r"(\d+)\s*(hour|hr|h|minute|min|m)s?", duration_str, re.IGNORECASE)
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2).lower()
            if unit in ["hour", "hr", "h"]:
                end_dt = start_dt + timedelta(hours=value)
            elif unit in ["minute", "min", "m"]:
                end_dt = start_dt + timedelta(minutes=value)
        else: # Fallback to dateparser for duration if regex fails
            parsed_duration_dt = dateparser.parse(duration_str, settings={'RELATIVE_BASE': datetime.min})
            if parsed_duration_dt:
                duration_td = (parsed_duration_dt - datetime.min)
                if duration_td > timedelta(0):
                    end_dt = start_dt + duration_td
                else:
                    logger.warning(f"Could not parse duration string '{duration_str}' into a valid timedelta.")

    if start_dt and not end_dt: # Default to 1 hour if still no end_dt
        end_dt = start_dt + timedelta(hours=1)

    if not start_dt:
        return None, None, None, None, None

    title = title if title else "Event"

    logger.info(f"Gemini Parsed: Title='{title}', Start='{start_dt}', End='{end_dt}', Location='{location}', Reminder='{reminder}'")
    return title, start_dt, end_dt, location, reminder


async def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Send me a message describing an event (e.g., 'Team meeting tomorrow at 2 PM for 1 hour'), and I'll use Gemini AI to create an .ics calendar file for you.",
        # reply_markup=ForceReply(selective=True), # Not needed for webhooks usually
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "I use Google Gemini AI to understand your event description and create calendar events!\n\n"
        "Try sending messages like:\n"
        "- 'Coffee with Sarah next Tuesday at 10:30 AM for 45 minutes'\n"
        "- 'Project deadline on Dec 1st 5pm at the main office'\n"
        "- 'Gym session tomorrow from 6pm to 7pm'\n\n"
        "I'll extract the title, start/end times, and location. All times are interpreted using Israel time zone (IST/IDT). "
        "If no end time or duration is found, I'll assume a 1-hour event."
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: CallbackContext) -> None:
    if not update.message or not update.message.text:
        return # Ignore empty messages or updates without text

    user_message = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Received message from {update.effective_user.username}: {user_message}")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    title, start_dt, end_dt, location, reminder = await parse_event_from_text_gemini(user_message) # Use await for async gemini call

    if not start_dt:
        await update.message.reply_text(
            "Sorry, Gemini couldn't understand the event details from your message. "
            "Please try rephrasing or be more specific about the date and time."
        )
        return

    try:
        # Create Google Calendar link with correct time format
        # Google Calendar expects times in UTC format without the 'Z' suffix
        start_str = start_dt.strftime("%Y%m%dT%H%M%S")  # Removed the 'Z'
        end_str = end_dt.strftime("%Y%m%dT%H%M%S")      # Removed the 'Z'
        
        # Add UTC timezone indicator for Google Calendar
        # start_str += "Z"  # This is the proper format for Google Calendar
        # end_str += "Z"    # This is the proper format for Google Calendar
        
        # Debug logging
        logger.info(f"Calendar link times: start={start_str}, end={end_str}")
        logger.info(f"Original datetimes: start={start_dt.isoformat()}, end={end_dt.isoformat()}")
        
        event_title_encoded = title.replace(" ", "+")
        location_encoded = location.replace(" ", "+") if location else ""
        
        gcal_link = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={event_title_encoded}&dates={start_str}/{end_str}"
        if location_encoded:
            gcal_link += f"&location={location_encoded}"
        
        # Add reminder parameter if provided
        if reminder:
            gcal_link += f"&reminders=popup%3A{reminder}"
        
        # Create message with event details and Google Calendar link
        # Display times with proper timezone in the message
        message_parts = [f"<b>Event Created (via Gemini):</b> {title}"]
        # Format with Israel timezone
        message_parts.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M')} (Israel Time)")
        if end_dt:
            message_parts.append(f"End: {end_dt.strftime('%Y-%m-%d %H:%M')} (Israel Time)")
        if location:
            message_parts.append(f"Location: {location}")
        if reminder:
            # Format the reminder nicely for display
            if reminder == 60:
                message_parts.append(f"Reminder: 1 hour before")
            elif reminder >= 60 and reminder % 60 == 0:
                message_parts.append(f"Reminder: {reminder // 60} hours before")
            else:
                message_parts.append(f"Reminder: {reminder} minutes before")
            
        message_parts.append("\n<b>Add this event to your calendar:</b>")
        message_parts.append(f"<a href='{gcal_link}'>Add to Google Calendar</a>")
        
        # Send message with Google Calendar link
        await update.message.reply_html(
            "\n".join(message_parts),
            disable_web_page_preview=False
        )
        
    except Exception as e:
        logger.error(f"Error creating calendar link: {e}", exc_info=True)
        await update.message.reply_text("Sorry, something went wrong while creating the calendar link.")

# --- Webhook specific configuration ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", "8443")) # Default to 8443 if not set

def webhook_main() -> None:
    """Set up and run the bot with webhooks for Gemini."""
    # Use non-async function since run_webhook manages its own event loop
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if not WEBHOOK_URL:
        logger.error("WEBHOOK_URL environment variable not set. Cannot start with webhooks.")
        return

    webhook_path_segment = TELEGRAM_BOT_TOKEN.split(':')[-1]
    webhook_path = f"/{webhook_path_segment}"
    full_webhook_url = f"{WEBHOOK_URL}{webhook_path}"

    logger.info(f"Setting webhook for URL: {full_webhook_url}")

    # Run the webhook server - this method handles the event loop internally
    logger.info(f"Starting webhook listener on port {PORT} for path {webhook_path}")
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path,  # Changed from webhook_path to url_path
        webhook_url=full_webhook_url
    )
    logger.info("Bot webhook server started.")


def polling_main() -> None:
    """Run the bot with polling (for local development)."""
    application_polling = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application_polling.add_handler(CommandHandler("start", start))
    application_polling.add_handler(CommandHandler("help", help_command))
    application_polling.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Gemini ICS Bot starting with polling for local development...")
    application_polling.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    if os.getenv("USE_WEBHOOKS", "false").lower() == "true" and WEBHOOK_URL:
        logger.info("Starting bot with WEBHOOKS.")
        webhook_main()  # Call directly without asyncio.run()
    else:
        logger.info("Starting bot with POLLING (default or USE_WEBHOOKS not true/WEBHOOK_URL not set).")
        polling_main()
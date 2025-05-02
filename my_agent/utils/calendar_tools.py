import os
import datetime
import json
from typing import List, Dict, Any, Optional

# Google API Libraries
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Langchain & Pydantic
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, ValidationError

# --- Constants ---
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly', 'https://www.googleapis.com/auth/calendar.events']
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'

# --- Base Class for Authentication ---

class GoogleCalendarBase:
    """Handles Google Calendar API authentication and service creation."""
    _creds: Optional[Credentials] = None
    _service: Optional[Any] = None

    def _authenticate(self):
        """Authenticates with Google Calendar API using OAuth 2.0."""
        creds = None
        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                print(f"Error loading token file: {e}. Re-authenticating.")
                creds = None # Force re-authentication

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}. Re-authenticating.")
                    creds = None # Force re-authentication flow
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                    creds = flow.run_local_server(port=0)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Credentials file '{CREDENTIALS_FILE}' not found. Please ensure it's in the correct directory.")
                except Exception as e:
                    raise Exception(f"Error during authentication flow: {str(e)}")

            # Save the credentials for the next run
            try:
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                 print(f"Warning: Could not save token file: {e}")

        self._creds = creds
        try:
            self._service = build('calendar', 'v3', credentials=self._creds)
        except Exception as e:
            raise Exception(f"Failed to build Google Calendar service: {str(e)}")

    @property
    def service(self) -> Any:
        """Provides an authenticated Google Calendar service instance."""
        if not self._service or (self._creds and not self._creds.valid):
             # Re-authenticate if service is missing or credentials expired
             # (refresh usually happens automatically if refresh token exists,
             # but explicitly call _authenticate if needed)
             self._authenticate()
        if not self._service:
             raise Exception("Google Calendar service could not be initialized.")
        return self._service

# --- Tool for Getting Calendar Events ---

class GetCalendarEventsInput(BaseModel):
    query: str = Field(default="", description="Optional query to filter events (currently not implemented in filtering logic)")
    days_back: int = Field(default=7, ge=0, description="Number of days to look back for events from today.")
    days_forward: int = Field(default=30, ge=0, description="Number of days to look forward for events from today.")


class GetCalendarEventsTool(BaseTool, GoogleCalendarBase):
    name: str = "get_calendar_events"
    description: str = "Fetches events from the user's primary Google Calendar within a specified date range."
    args_schema: type[BaseModel] = GetCalendarEventsInput

    def _run(self, query: str = "", days_back: int = 7, days_forward: int = 30) -> Dict[str, Any]:
        """Fetches events from the primary Google Calendar."""
        try:
            now = datetime.datetime.utcnow()
            time_min_dt = now - datetime.timedelta(days=days_back)
            time_max_dt = now + datetime.timedelta(days=days_forward)

            # Use start of the day for time_min and end of the day for time_max for better inclusiveness
            time_min = time_min_dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
            time_max = time_max_dt.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=100, # Increased limit slightly
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            if not events:
                return {
                    'status': 'success',
                    'message': 'No events found in the specified time range.',
                    'time_range': {'start': time_min, 'end': time_max},
                    'events': []
                }

            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                formatted_events.append({
                    'summary': event.get('summary', 'No Title'),
                    'start': start,
                    'end': end,
                    'location': event.get('location'),
                    'description': event.get('description'),
                    'id': event.get('id') # Include event ID
                })

            # Note: The 'query' parameter is received but not used for filtering here.
            # Implementing text-based filtering would require iterating through 'formatted_events'.

            return {
                'status': 'success',
                'message': f'Found {len(formatted_events)} events.',
                'time_range': {'start': time_min, 'end': time_max},
                'events': formatted_events
            }

        except HttpError as e:
             error_content = json.loads(e.content.decode('utf-8'))
             error_message = error_content.get('error', {}).get('message', str(e))
             return {'status': 'error', 'message': f"Google API Error: {error_message}", 'error_type': type(e).__name__}
        except Exception as e:
             return {'status': 'error', 'message': str(e), 'error_type': type(e).__name__}

# --- Tool for Creating Calendar Events ---

class CreateCalendarEventInput(BaseModel):
    summary: str = Field(..., description="The title/summary of the event (Required).")
    start_time: str = Field(..., description="Start datetime in ISO 8601 format (e.g., '2025-04-15T10:00:00Z' or '2025-04-15T10:00:00+02:00' ) (Required).")
    end_time: str = Field(..., description="End datetime in ISO 8601 format (e.g., '2025-04-15T11:00:00Z' or '2025-04-15T11:00:00+02:00' ) (Required).")
    description: str = Field(default="", description="Optional description for the event.")
    location: str = Field(default="", description="Optional location for the event.")

    @field_validator('start_time', 'end_time')
    def check_iso_format(cls, v):
        """Basic validation for ISO 8601 format."""
        try:
            # Handle 'Z' for UTC explicitly for fromisoformat
            datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Datetime must be in valid ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SSZ' or 'YYYY-MM-DDTHH:MM:SS+HH:MM')")
        return v


class CreateCalendarEventTool(BaseTool, GoogleCalendarBase):
    name: str = "create_calendar_event"
    description: str = (
        "Creates a new event in the user's primary Google Calendar. "
        "Requires 'summary', 'start_time', and 'end_time'. "
        "Datetimes must be in ISO 8601 format like : 2025-04-14T10:30:00+02:00"
        "Ask the user for any missing required information before calling the tool."
        "IMPORTANT: Format the args with summary, start_time and end_time at least"
    )
    args_schema: type[BaseModel] = CreateCalendarEventInput

    def _run(self,
             summary: str,
             start_time: str,
             end_time: str,
             description: str = "",
             location: str = "",
            ) -> Dict[str, Any]:
        """Creates a new event using the Google Calendar API."""
        try:

            event_body = {
                'summary': summary,
                'description': description,
                'location': location,
                'start': {'dateTime': start_time}, # Assumes timezone info is in the string
                'end': {'dateTime': end_time},     # Assumes timezone info is in the string
            }

            created_event = self.service.events().insert(
                calendarId='primary',
                body=event_body,
                sendUpdates='all' # Notify attendees
            ).execute()

            return {
                'status': 'success',
                'message': 'Event created successfully.',
                'event': {
                    'id': created_event.get('id'),
                    'summary': created_event.get('summary'),
                    'start': created_event.get('start', {}).get('dateTime'),
                    'end': created_event.get('end', {}).get('dateTime'),
                    'location': created_event.get('location', ''),
                    'htmlLink': created_event.get('htmlLink')
                }
            }

        except ValidationError as e: # Catch Pydantic validation errors if framework doesn't
             return {'status': 'error', 'message': f"Input validation error: {str(e)}", 'error_type': 'ValidationError'}
        except HttpError as e:
             error_content = json.loads(e.content.decode('utf-8'))
             error_message = error_content.get('error', {}).get('message', str(e))
             return {'status': 'error', 'message': f"Google API Error: {error_message}", 'error_type': type(e).__name__}
        except Exception as e:
            return {'status': 'error', 'message': f"Failed to create event: {str(e)}", 'error_type': type(e).__name__}

# --- Instantiate Tools ---
# These can now be used by your Langchain agent or application
get_calendar_tool = GetCalendarEventsTool()
create_calendar_tool = CreateCalendarEventTool()
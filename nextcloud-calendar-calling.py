import asyncio
import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import caldav
from icalendar import Alarm
from pydantic import BaseModel, Field, validator


class Tools:
    class Valves(BaseModel):
        nextcloud_url: str = Field(
            default="",
            description=(
                "Full CalDAV calendar URL, for example "
                "https://cloud.example.com/remote.php/dav/calendars/username/"
            ),
        )
        nextcloud_user: str = Field(default="", description="Nextcloud username.")
        nextcloud_pass: str = Field(
            default="",
            description="Nextcloud app password.",
        )
        calendar_name: str = Field(
            default="",
            description=(
                "Optional calendar name. Leave blank to use the first calendar "
                "returned by the server."
            ),
        )
        default_timezone: str = Field(
            default="Europe/Warsaw",
            description="Timezone for input times without an offset.",
        )
        reminder_minutes: int = Field(
            default=30,
            ge=0,
            le=525600,
            description="Minutes before the event to trigger a reminder; 0 disables it.",
        )

        @validator("default_timezone")
        def validate_timezone(cls, value):
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, TypeError) as exc:
                raise ValueError(f"Unknown timezone: {value}") from exc
            return value

    def __init__(self):
        self.valves = self.Valves()
        self._client = None
        self._client_config = None
    
    def _get_calendar(self):
        """Connect using the current valves and return the configured calendar."""
        url = self.valves.nextcloud_url.strip()
        username = self.valves.nextcloud_user.strip()
        password = self.valves.nextcloud_pass
        if not self.valves.nextcloud_url:
            raise ValueError("Set the Nextcloud CalDAV URL in the tool's Valves.")
        if not self.valves.nextcloud_user or not self.valves.nextcloud_pass:
            raise ValueError(
                "Set the Nextcloud username and app password in the tool's Valves."
           )

        # Recreate the client if any connection setting has changed.
        # Do not log or return this tuple: it contains the app password.
        config = (url, username, password)
        if self._client is None or config != self._client_config:
            self._client = caldav.DAVClient(
               url=url,
               username=username,
               password=password,
            )
            self._client_config = config
        calendars = self._client.principal().calendars()
        if not calendars:
            raise ValueError("No calendars found in the Nextcloud account.")

        calendar_name = self.valves.calendar_name.strip()
        if calendar_name:
            for calendar in calendars:
                if calendar.name == calendar_name:
                    return calendar
            names = ", ".join(c.name or "(unnamed)" for c in calendars)
            raise ValueError(
                f"Calendar '{calendar_name}' not found. Available: {names}"
            )
        return calendars[0]
        
    async def list_upcoming_events(self, days_ahead: int = 7) -> str:
        """
        List upcoming events from the user's Nextcloud calendar.

        Use this when the user asks what they have scheduled, what meetings
        or appointments are coming up, or to check their calendar for a
        specific date range.

        Args:
            days_ahead: Number of days to look ahead from now (0–365).
                        0 = only today's events. Default: 7.

        Returns:
            A plain-text list of events, one per line, formatted as:
            "- Title (YYYY-MM-DD HH:MM TZ to YYYY-MM-DD HH:MM TZ)"
            Returns "No events found..." if the range is empty.
        """
        return await asyncio.to_thread(self._list_upcoming_events_sync, days_ahead)

    def _list_upcoming_events_sync(self, days_ahead: int = 7) -> str:
        try:
            if not 0 <= days_ahead <= 365:
                return "Error: days_ahead must be between 0 and 365."

            calendar = self._get_calendar()
            tz = ZoneInfo(self.valves.default_timezone)
            now = datetime.datetime.now(tz)
            # end = now + datetime.timedelta(days=days_ahead)

            if days_ahead == 0:
            # Search through the end of today in the configured timezone.
                end = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end += datetime.timedelta(days=1)
            else:
                end = now + datetime.timedelta(days=days_ahead)
            if end <= now:
                return "Error: search end time must be later than start time."

            events = calendar.search(
                start=now,
                end=end,
                event=True,
                expand=True,
            )
            if not events:
                return f"No events found in the next {days_ahead} days."

            lines = []
            for event in events:
                component = event.vobject_instance.vevent
                summary_prop = getattr(component, "summary", None)
                summary = str(summary_prop.value) if summary_prop else "Untitled event"

                start_value = component.dtstart.value
                end_prop = getattr(component, "dtend", None)
                end_value = end_prop.value if end_prop else None

                start_text = self._format_datetime(start_value, tz)
                end_text = (
                    self._format_datetime(end_value, tz)
                    if end_value is not None
                    else "unspecified"
                )
                lines.append(f"- {summary} ({start_text} to {end_text})")

            return "\n".join(lines)
        except Exception as exc:
            return f"Error fetching events: {exc}"

    async def create_calendar_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
    ) -> str:
        """
        Create a calendar event with a display reminder.

        Times should be ISO 8601. Include an offset when possible, e.g.
        '2026-10-27T14:00:00-04:00'. Naive times use default_timezone.

        Args:
            title: Event title (required).
            start_time: ISO 8601 start datetime.
            end_time: ISO 8601 end datetime (must be after start_time).
            description: Optional event description.

        Returns:
            Confirmation string with the event title, start time, and
            reminder info. On error: "Error: ..." or "Failed to create event: ..."
        """
        return await asyncio.to_thread(self._create_calendar_event_sync, title, start_time, end_time, description)

    def _create_calendar_event_sync(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
    ) -> str:
        try:
            if not title or not title.strip():
                return "Error: title cannot be empty."

            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)

            if end_dt <= start_dt:
                return "Error: end_time must be later than start_time."

            calendar = self._get_calendar()

            # Create the event first, then add a standards-compliant VALARM.
            event = calendar.save_event(
                dtstart=start_dt,
                dtend=end_dt,
                summary=title.strip(),
                description=description,
            )

            reminder_minutes = self.valves.reminder_minutes
            if reminder_minutes > 0:
                ical_event = event.icalendar_component
                alarm = Alarm()
                alarm.add("ACTION", "DISPLAY")
                alarm.add(
                    "DESCRIPTION",
                    description.strip() or f"Reminder: {title.strip()}",
                )
                alarm.add(
                    "TRIGGER",
                    datetime.timedelta(minutes=-reminder_minutes),
                )
                ical_event.add_component(alarm)
                event.save()

            reminder_text = (
                f"with a {reminder_minutes}-minute reminder"
                if reminder_minutes > 0
                else "without a reminder"
            )
            return (
                f"Created '{title.strip()}' starting at {start_dt.isoformat()} "
                f"{reminder_text}."
            )
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Failed to create event: {exc}"

    def _parse_datetime(self, value: str) -> datetime.datetime:
        """Parse ISO 8601 datetime; attach configured timezone if naive."""
        try:
            parsed = datetime.datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid datetime {value!r}; use ISO 8601, "
                "for example 2026-10-27T14:00:00-04:00."
            ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(self.valves.default_timezone))
        return parsed

    @staticmethod
    def _format_datetime(
        value: datetime.date | datetime.datetime,
        timezone: ZoneInfo,
    ) -> str:
        """Format a CalDAV date or datetime, including all-day dates."""
        if isinstance(value, datetime.datetime):
            # Do not call astimezone() on a naive datetime; interpret it in
            # the configured timezone rather than the machine's local timezone.
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone)
            return value.astimezone(timezone).strftime("%Y-%m-%d %H:%M %Z")

        if isinstance(value, datetime.date):
            return value.isoformat() + " (all day)"

        return str(value)

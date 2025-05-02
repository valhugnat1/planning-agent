# 🧠 LangChain Agent with Routing and Google Calendar Tools

This project is a LangChain-based AI agent designed to route user questions intelligently between two specialized models: a **General Assistant** and a **Developer Assistant**. The General Assistant is further enhanced with tools to interact with **Google Calendar**, allowing it to **read** and **write events**.

## 🚀 Getting Started

1.  **Set up Google Calendar Credentials:**
    * Go to the [Google Cloud Console](https://console.cloud.google.com/).
    * Create a project (or select an existing one).
    * Enable the **Google Calendar API**.
    * Configure the **OAuth Consent Screen**:
        * Choose "External" user type (unless using Google Workspace).
        * Add the required scopes: `.../auth/calendar.readonly` and `.../auth/calendar.events`.
        * Add your Google account email as a **Test User** while the app is in "Testing" mode.
    * Go to "Credentials", click "+ Create Credentials" -> "OAuth client ID".
    * Select **"Desktop app"** as the application type.
    * Click "Create" and **Download the JSON** credentials file.
    * **Rename** the downloaded file to `credentials.json` and place it in the **root directory** of this project.

2.  **Launch the Agent:**
    * Ensure you have any other necessary environment variables or API keys set up.
    * Run the agent in development mode:
        ```bash
        langgraph dev
        ```

3.  **First Run Authentication:**
    * When you first run the agent and it needs to access the calendar, it will open a browser window asking you to log in to your Google account and grant permission (based on the consent screen you configured).
    * After you grant access, a `token.json` file will be created in the project root to store your authorization for future runs.

## 🧭 Architecture Overview

The core of this agent is a **Router Chain**, which analyzes each incoming question and decides which of the two expert agents should respond:

-   **General Model**
    * Focused on everyday tasks and productivity-related queries.
    * Equipped with two tools:
        * `get_calendar_events`: Fetches upcoming events.
        * `Calendar`: Adds new events to your calendar.
-   **Developer Model**
    * Specialized in technical questions, programming help, and dev workflows.

This modular approach allows the system to deliver more accurate and context-aware responses based on the user’s intent.

## 🛠️ Tools

### General Model Tools

| Tool Name              | Description                             |
|------------------------|-----------------------------------------|
| `get_calendar_events`  | Retrieves upcoming events from Calendar |
| `create_calendar_event`| Creates new events in Calendar          |

These tools require valid OAuth credentials and appropriate permissions to access the user's Google Calendar.


These tools require the `credentials.json` setup (as described in Getting Started) and user consent via the one-time browser authentication flow.# planning-agent

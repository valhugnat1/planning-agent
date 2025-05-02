#  Agent planner

## Overview

`My Agent` is an AI-driven application designed to execute tasks based on a structured plan. It leverages various AI models and tools to plan, execute, and manage tasks efficiently. The application is built using a modular architecture, making it easy to extend and customize.

## Features

- **Task Planning**: Generates a step-by-step plan for executing tasks.
- **Task Execution**: Executes tasks based on the generated plan.
- **Replanning**: Adjusts the plan if necessary.
- **Routing**: Determines the next step in the workflow.
- **Response Cleaning**: Cleans and formats the final response.

## Directory Structure

```
.
├── .env.example
├── README.md
├── langgraph.json
├── requirements.txt
├── tuto.ipynb
└── my_agent/
    ├── agent.py
    └── utils/
        ├── calendar_tools.py
        ├── naming.py
        ├── nodes.py
        ├── prompts.py
        ├── state.py
        └── tools.py
```

## Setup

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd my-agent
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**

   Copy the `.env.example` file to `.env` and fill in the required API keys and endpoints.

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with your API keys:

   ```plaintext
   SCW_GENERATIVE_APIs_ENDPOINT=https://your-generative-api-endpoint
   SCW_SECRET_KEY=your-secret-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

## Running the Application

1. **Start the Agent**

   You can run the agent using the following command:

   ```bash
   langgraph dev
   ```

2. **Interact with the Agent**

   The agent will prompt you to input a task. It will then plan, execute, and provide a cleaned response.

## Extending the Application

- **Adding New Tools**: You can add new tools by defining them in `my_agent/utils/tools.py` and importing them into the `tools` list.
- **Customizing Prompts**: Modify the prompts in `my_agent/utils/prompts.py` to tailor the agent's behavior to your needs.
- **Modifying Workflow**: Adjust the workflow in `my_agent/agent.py` to change the sequence of operations.


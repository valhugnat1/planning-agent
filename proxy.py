import time
import os
import re
import json
import uuid
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI, APIConnectionError, APIStatusError # Added specific errors
# lunary is optional for monitoring, ensure it's installed if you uncomment its use
# import lunary 
from dotenv import load_dotenv # To load environment variables from .env file

# Load environment variables from .env file (optional, good for development)
load_dotenv() 

app = FastAPI(title="OpenAI Proxy API with Tool Call Parsing")

# --- Configuration ---
# Fetch configuration from environment variables
# Make sure to set these in your environment or a .env file
OPENAI_BASE_URL = "https://api.fireworks.ai/inference/v1"
OPENAI_API_KEY = os.getenv("FIREWORKS_API_KEY")


if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # You might want to raise an error or exit here in a real application
    # raise ValueError("OPENAI_API_KEY environment variable is required.")

# --- Initialize OpenAI Client ---
# This client points to the actual downstream API (OpenAI or compatible)
try:
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    # Optional: Enable monitoring if lunary is installed and configured
    # lunary.monitor(client) 
    print(f"OpenAI client initialized pointing to: {OPENAI_BASE_URL}")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None # Ensure client is None if initialization fails

# --- Pydantic Models (matching OpenAI schema) ---
class ChatMessage(BaseModel):
    role: str
    # Content can be None, especially for tool calls or responses
    content: Optional[str] = None 
    tool_calls: Optional[List[Dict[str, Any]]] = None # Added for incoming tool messages if needed
    tool_call_id: Optional[str] = None # For tool response messages

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Add tool-related parameters if your downstream model supports them
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

# --- Helper Function for Tool Call Parsing ---
def parse_tool_calls(content: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parses <tool_call> tags within content string and formats them.
    Returns None if no valid tool calls are found.
    """
    if not content:
        return None

    # Regex to find <tool_call>...</tool_call> blocks
    # re.DOTALL makes '.' match newline characters as well
    tool_call_matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

    if not tool_call_matches:
        return None

    parsed_tool_calls = []
    for tool_call_json_str in tool_call_matches:
        try:
            # Trim whitespace just in case
            tool_call_data = json.loads(tool_call_json_str.strip()) 
            
            if "name" not in tool_call_data or "arguments" not in tool_call_data:
                print(f"Warning: Skipping invalid tool call structure: {tool_call_json_str}")
                continue

            # Ensure arguments are dumped back into a string format, 
            # as expected by the OpenAI spec for the 'arguments' field.
            arguments_str = json.dumps(tool_call_data["arguments"])

            # Generate a unique ID for the tool call
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}" # Mimic OpenAI's format

            parsed_tool_calls.append({
                "id": tool_call_id,
                "type": "function",  # Assuming all are function calls
                "function": {
                    "name": tool_call_data["name"],
                    "arguments": arguments_str,
                }
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from tool call: {tool_call_json_str}. Error: {e}")
            continue # Skip malformed JSON

    return parsed_tool_calls if parsed_tool_calls else None

# --- API Endpoints ---
@app.post("/v1/chat/completions") # Match OpenAI path convention
@app.post("/chat/completions")    # Keep original path for compatibility
async def chat_completions(request: ChatCompletionRequest):
    if not client:
         raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check configuration.")

    # Convert Pydantic models to dictionaries for OpenAI SDK
    # Handle potential None content in messages
    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    
    # Prepare kwargs for OpenAI API call, excluding None values from the request
    kwargs = request.model_dump(exclude_none=True)
    kwargs["messages"] = messages # Overwrite messages with the processed list

    print(f"Forwarding request to {client.base_url}: {kwargs}") # Log outgoing request

    try:
        # Call the downstream OpenAI-compatible API
        response = client.chat.completions.create(**kwargs)

        # --- Handle Streaming Response ---
        if request.stream:
            # Basic streaming: Forward chunks directly.
            # Note: This simple forwarding WON'T parse and restructure tool calls within the stream.
            # A more complex implementation would buffer chunks if tool call parsing during streaming is needed.
            def stream_generator():
                try:
                    for chunk in response:
                        # print(f"Stream Chunk Received: {chunk.model_dump_json()}") # Debugging
                        # Yield the chunk data directly (usually JSON string)
                        yield f"data: {chunk.model_dump_json()}\n\n" 
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Error during streaming: {e}")
                    # You might want to signal an error in the stream if possible
                    # yield f"data: {json.dumps({'error': {'message': 'Streaming error occurred', 'type': 'proxy_error'}})}\n\n"
                    # yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # --- Handle Non-Streaming Response ---
        else:
            print(f"Received response from downstream: {response}") # Log received response

            # Prepare the response structure based on OpenAI's format
            response_payload = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created or int(time.time()), # Use response.created if available
                "model": response.model,
                "choices": [],
                "usage": response.usage.model_dump() if response.usage else None,
                # Include system_fingerprint if present
                "system_fingerprint": response.system_fingerprint, 
            }

            final_choices = []
            for i, choice in enumerate(response.choices):
                message = choice.message
                finish_reason = choice.finish_reason
                
                # Initialize choice message structure
                choice_message_dict = {
                    "role": message.role or "assistant",
                    "content": message.content, # Start with original content
                    "tool_calls": None # Initialize as None
                }

                # --- Attempt to parse tool calls from content ---
                parsed_tools = parse_tool_calls(message.content)

                if parsed_tools:
                    print(f"Parsed tool calls found in choice {i}: {parsed_tools}")
                    # If tool calls are parsed successfully from content:
                    # - Set the 'tool_calls' field in the response.
                    # - Set 'content' to None (standard OpenAI behavior when returning tool calls).
                    # - Set finish_reason to 'tool_calls' if it wasn't already.
                    choice_message_dict["tool_calls"] = parsed_tools
                    choice_message_dict["content"] = None # Standard practice
                    if finish_reason != "tool_calls":
                        print(f"Original finish_reason was '{finish_reason}', setting to 'tool_calls' due to parsed content.")
                        finish_reason = "tool_calls" 
                elif message.tool_calls:
                     # If the downstream API *already* provided structured tool_calls, use them directly
                     print(f"Using pre-structured tool calls from downstream choice {i}: {message.tool_calls}")
                     # Convert ToolCall objects to dictionaries if needed
                     choice_message_dict["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
                     # Content might already be None, but ensure it is
                     choice_message_dict["content"] = message.content # Keep original content if provided alongside structured tools, though usually it's None

                # Append the processed choice to the list
                final_choices.append({
                    "message": choice_message_dict,
                    "index": i,
                    "finish_reason": finish_reason,
                    # Include logprobs if present
                    "logprobs": choice.logprobs.model_dump() if choice.logprobs else None
                })
            
            response_payload["choices"] = final_choices

            print(f"Returning processed response: {response_payload}") # Log final response
            return response_payload # Return the structured dictionary

    except APIStatusError as e:
        print(f"Downstream API Error: Status={e.status_code} Response={e.response}")
        raise HTTPException(status_code=e.status_code, detail=f"Downstream API error: {e.message}")
    except APIConnectionError as e:
        print(f"Downstream Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to downstream API: {OPENAI_BASE_URL}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OpenAI Proxy API with Tool Call Parsing is running. Send POST requests to /chat/completions or /v1/chat/completions"}

# --- Run the application ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000)) # Allow configuring port via environment variable
    host = os.getenv("HOST", "0.0.0.0") # Allow configuring host
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

# --- Dependencies ---
# You'll need to install:
# pip install fastapi uvicorn openai python-dotenv requests httpx
# Optional for monitoring:
# pip install lunary
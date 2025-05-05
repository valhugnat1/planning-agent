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
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1") # Default if not set
OPENAI_API_KEY = os.getenv("FIREWORKS_API_KEY") # Use FIREWORKS_API_KEY or OPENAI_API_KEY


if not OPENAI_API_KEY:
    # Fallback to standard OPENAI_API_KEY if FIREWORKS_API_KEY is not set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: Neither FIREWORKS_API_KEY nor OPENAI_API_KEY environment variable is set.")
    # You might want to raise an error or exit here in a real application
    # raise ValueError("An API key environment variable (FIREWORKS_API_KEY or OPENAI_API_KEY) is required.")
    client = None
else:
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

# --- Helper Function for Tool Call Parsing (Used in Non-Streaming and Streaming) ---
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
            # Handle cases where arguments might already be a string or need dumping
            if isinstance(tool_call_data["arguments"], str):
                 # Attempt to parse and re-dump to ensure valid JSON string format
                 try:
                     arguments_obj = json.loads(tool_call_data["arguments"])
                     arguments_str = json.dumps(arguments_obj)
                 except json.JSONDecodeError:
                     print(f"Warning: Arguments field contains non-JSON string: {tool_call_data['arguments']}")
                     # Keep the original string if it's not valid JSON
                     arguments_str = tool_call_data["arguments"]
            else:
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
            # This generator handles parsing tool calls from content within the stream
            def stream_generator():
                content_buffer = "" # Buffer for accumulating content across chunks
                current_tool_call_index = 0 # To assign index to streamed tool calls
                role_sent = False # Track if role has been sent in the stream
                streamed_tool_calls_exist = False # Track if we yielded any tool calls

                try:
                    for chunk in response:
                        # print(f"Raw Chunk: {chunk.model_dump_json()}") # Debugging
                        if not chunk.choices:
                            # Pass through non-choice chunks (e.g., usage info from some providers)
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        finish_reason = choice.finish_reason
                        logprobs = choice.logprobs # Capture logprobs if present

                        current_chunk_role = delta.role if delta else None
                        current_chunk_content = delta.content if delta else None
                        current_chunk_tool_calls = delta.tool_calls if delta else None

                        # Determine role for the delta chunk(s) we are about to yield
                        role_to_yield = None
                        if current_chunk_role and not role_sent:
                            role_to_yield = current_chunk_role
                            role_sent = True
                        elif not role_sent and (current_chunk_content or current_chunk_tool_calls):
                            # If role wasn't in the first delta, but content/tools are, assume 'assistant'
                            role_to_yield = "assistant"
                            role_sent = True

                        # --- Step 1: Handle incoming structured tool calls from downstream ---
                        if current_chunk_tool_calls:
                            # If the downstream API streams tool calls correctly, forward them
                            print(f"Forwarding structured tool call delta: {current_chunk_tool_calls}")
                            # We might need to adjust indices or IDs if mixing parsed/native calls.
                            # For now, assume downstream is the source of truth if it provides tool_calls delta.
                            # Reconstruct the chunk payload to potentially add the role if needed
                            structured_tool_chunk_dict = chunk.model_dump(exclude={'choices'})
                            structured_tool_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {
                                    "role": role_to_yield, # Add role if not sent yet
                                    "content": None, # Content is usually None when tool calls are present
                                    "tool_calls": [tc.model_dump(exclude_none=True) for tc in current_chunk_tool_calls]
                                },
                                "finish_reason": None, # Finish reason comes later
                                "logprobs": None # Logprobs usually don't accompany tool call deltas
                            }]
                            yield f"data: {json.dumps(structured_tool_chunk_dict)}\n\n"

                            # Update tool call index based on the received chunk
                            last_tool_call = current_chunk_tool_calls[-1]
                            if last_tool_call.index is not None:
                                current_tool_call_index = last_tool_call.index + 1
                            streamed_tool_calls_exist = True # Mark that we have seen tool calls
                            # If content also exists in this chunk, buffer it for later processing
                            if current_chunk_content:
                                content_buffer += current_chunk_content
                            continue # Move to next chunk after handling structured tool calls

                        # --- Step 2: Handle content and buffer it ---
                        if current_chunk_content:
                            content_buffer += current_chunk_content
                            # print(f"Buffer updated: '{content_buffer}'") # Debugging

                        # --- Step 3: Parse complete tool calls from buffer ---
                        processed_upto_index = 0 # Index in buffer up to which we have processed/yielded
                        while True:
                            # Search for the next complete tool call *after* the already processed part
                            match = re.search(r"<tool_call>(.*?)</tool_call>", content_buffer[processed_upto_index:], re.DOTALL)
                            if not match:
                                break # No more complete tool calls in the remaining buffer

                            # Calculate absolute indices in the original buffer
                            match_start_abs = processed_upto_index + match.start()
                            match_end_abs = processed_upto_index + match.end()

                            # Extract content *before* the tag (prefix) that hasn't been yielded yet
                            prefix = content_buffer[processed_upto_index:match_start_abs]

                            # Yield the prefix content chunk
                            if prefix:
                                print(f"Yielding prefix content: '{prefix}'")
                                prefix_chunk_dict = chunk.model_dump(exclude={'choices'}) # Base structure
                                prefix_chunk_dict["choices"] = [{
                                    "index": choice.index,
                                    "delta": {"role": role_to_yield, "content": prefix},
                                    "finish_reason": None,
                                    "logprobs": None
                                }]
                                yield f"data: {json.dumps(prefix_chunk_dict)}\n\n"
                                role_to_yield = None # Role is sent only once

                            # Attempt to parse the found tool call JSON
                            tool_call_json_str = match.group(1).strip()
                            try:
                                tool_call_data = json.loads(tool_call_json_str)
                                if "name" in tool_call_data and "arguments" in tool_call_data:
                                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

                                    # Ensure arguments are a JSON string
                                    if isinstance(tool_call_data["arguments"], str):
                                         try:
                                             arguments_obj = json.loads(tool_call_data["arguments"])
                                             arguments_str = json.dumps(arguments_obj)
                                         except json.JSONDecodeError:
                                             arguments_str = tool_call_data["arguments"] # Keep original if not valid JSON
                                    else:
                                         arguments_str = json.dumps(tool_call_data["arguments"])

                                    print(f"Parsed tool call from content (Index {current_tool_call_index}): ID {tool_call_id}, Name {tool_call_data['name']}")

                                    # --- Yield the parsed tool call structure in OpenAI format ---
                                    # Chunk 1: Tool call index, id, type, function name, empty args
                                    tool_chunk_dict_name = chunk.model_dump(exclude={'choices'})
                                    tool_chunk_dict_name["choices"] = [{
                                        "index": choice.index,
                                        "delta": {
                                            "role": role_to_yield, # Include role only on first delta yielded
                                            "content": None, # Content is None when tool_calls are present
                                            "tool_calls": [{
                                                "index": current_tool_call_index,
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {"name": tool_call_data["name"], "arguments": ""}
                                            }]
                                        },
                                        "finish_reason": None, "logprobs": None
                                    }]
                                    yield f"data: {json.dumps(tool_chunk_dict_name)}\n\n"
                                    role_to_yield = None # Role sent

                                    # Chunk 2: Tool call index and arguments delta
                                    # Stream arguments incrementally if they are long?
                                    # For simplicity, yield all args at once.
                                    # A more advanced impl could chunk the args string.
                                    if arguments_str: # Only yield args chunk if args exist
                                        tool_chunk_dict_args = chunk.model_dump(exclude={'choices'})
                                        tool_chunk_dict_args["choices"] = [{
                                            "index": choice.index,
                                            "delta": {
                                                "tool_calls": [{
                                                    "index": current_tool_call_index,
                                                    "function": {"arguments": arguments_str}
                                                }]
                                            },
                                            "finish_reason": None, "logprobs": None
                                        }]
                                        yield f"data: {json.dumps(tool_chunk_dict_args)}\n\n"

                                    current_tool_call_index += 1 # Increment for the next tool call
                                    streamed_tool_calls_exist = True # Mark that we yielded a tool call

                                else:
                                    print(f"Warning: Skipping invalid tool call structure in stream: {tool_call_json_str}")
                                    # Treat as content - leave it in the buffer by not advancing processed_upto_index past it

                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse JSON from tool call in stream: {tool_call_json_str}. Error: {e}")
                                # Treat as content - leave it in the buffer

                            # Update buffer processed index *past the tag* regardless of parse success/failure
                            # If we treat parse failures as content, we should yield them later.
                            # Let's assume invalid JSON means it wasn't a tool call, so keep it for content yield.
                            # If parsing was successful, update index past the tag.
                            # If parsing failed, do NOT update processed_upto_index here, let it be yielded as content later.
                            if 'tool_call_data' in locals() and "name" in tool_call_data and "arguments" in tool_call_data:
                                processed_upto_index = match_end_abs
                            else:
                                # If parsing failed or structure invalid, stop tag processing for this cycle
                                break


                        # --- Step 4: Yield remaining content in the buffer ---
                        remaining_content = content_buffer[processed_upto_index:]
                        if remaining_content:
                            print(f"Yielding remaining content: '{remaining_content}'")
                            remaining_content_chunk_dict = chunk.model_dump(exclude={'choices'})
                            remaining_content_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {"role": role_to_yield, "content": remaining_content},
                                "finish_reason": None,
                                "logprobs": None
                            }]
                            yield f"data: {json.dumps(remaining_content_chunk_dict)}\n\n"
                            role_to_yield = None # Role sent

                        # Update buffer: remove the processed/yielded part
                        content_buffer = content_buffer[processed_upto_index:]
                        # print(f"Buffer after processing chunk: '{content_buffer}'") # Debugging


                        # --- Step 5: Handle Finish Reason ---
                        if finish_reason:
                            print(f"Received finish_reason: {finish_reason}")
                            # Check buffer for incomplete tags? Optional.
                            # if re.search(r"<tool_call>(?!.*</tool_call>)", content_buffer, re.DOTALL):
                            #    print("Warning: Stream ended with potentially incomplete <tool_call> tag in buffer.")

                            final_chunk_dict = chunk.model_dump(exclude={'choices'})
                            # Determine the correct final reason
                            final_reason = finish_reason
                            if streamed_tool_calls_exist and finish_reason == 'stop':
                                # If we streamed tool calls AND the original reason was 'stop',
                                # OpenAI standard is to report 'tool_calls'
                                print(f"Overriding finish_reason from '{finish_reason}' to 'tool_calls' because tool calls were streamed.")
                                final_reason = 'tool_calls'
                            elif finish_reason == 'tool_calls':
                                streamed_tool_calls_exist = True # Ensure flag is set if downstream reports it

                            # Construct final chunk payload
                            final_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {}, # Empty delta in the final chunk
                                "finish_reason": final_reason,
                                "logprobs": logprobs # Include logprobs if they came with the final chunk
                            }]
                            yield f"data: {json.dumps(final_chunk_dict)}\n\n"
                            break # End the stream processing loop

                except APIConnectionError as e:
                    print(f"Error during streaming (Connection): {e}")
                    error_payload = {"error": {"message": f"Proxy connection error during stream: {e}", "type": "proxy_error", "code": 503}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except APIStatusError as e:
                    print(f"Error during streaming (API Status): {e}")
                    error_payload = {"error": {"message": f"Downstream API error during stream: {e.message}", "type": "downstream_api_error", "code": e.status_code}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except Exception as e:
                    print(f"Error during streaming (General): {e}")
                    import traceback
                    traceback.print_exc()
                    error_payload = {"error": {"message": f"Internal proxy error during stream: {e}", "type": "proxy_error", "code": 500}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                finally:
                    print("Stream finished. Yielding [DONE].")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # --- Handle Non-Streaming Response ---
        else:
            # (Non-streaming logic remains the same as in your original code)
            print(f"Received non-streaming response from downstream: {response}") # Log received response

            # Prepare the response structure based on OpenAI's format
            response_payload = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created or int(time.time()), # Use response.created if available
                "model": response.model,
                "choices": [],
                "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
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
                    print(f"Parsed tool calls found in non-streaming choice {i}: {parsed_tools}")
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
                     choice_message_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
                     # Content might already be None, but ensure it is if tool calls exist
                     if choice_message_dict["tool_calls"]:
                         choice_message_dict["content"] = message.content # Keep original content ONLY if provided alongside structured tools

                # Append the processed choice to the list
                final_choices.append({
                    "message": choice_message_dict,
                    "index": i,
                    "finish_reason": finish_reason,
                    # Include logprobs if present
                    "logprobs": choice.logprobs.model_dump(exclude_none=True) if choice.logprobs else None
                })

            response_payload["choices"] = final_choices

            print(f"Returning processed non-streaming response: {response_payload}") # Log final response
            return response_payload # Return the structured dictionary

    except APIStatusError as e:
        print(f"Downstream API Error: Status={e.status_code} Response={e.response}")
        raise HTTPException(status_code=e.status_code, detail=f"Downstream API error: {e.message or e.response.text}")
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
    # Enable reload for development if needed: uvicorn.run("main:app", host=host, port=port, reload=True)
    uvicorn.run(app, host=host, port=port)

# --- Dependencies ---
# You'll need to install:
# pip install fastapi uvicorn openai python-dotenv requests httpx pydantic
# Optional for monitoring:
# pip install lunary

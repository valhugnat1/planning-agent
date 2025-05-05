import time
import os
import re
import json
import uuid
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI, APIConnectionError, APIStatusError # Added specific errors
# lunary is optional for monitoring, ensure it's installed if you uncomment its use
# import lunary
from dotenv import load_dotenv # To load environment variables from .env file

# Load environment variables from .env file (optional, good for development)
load_dotenv()

app = FastAPI(title="OpenAI Proxy API with Tool & Think Tag Parsing")

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
    # Note: reasoning_content is added dynamically to the response, not part of the request model

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

# --- Helper Function for Parsing Tool Calls and Thinking Tags ---
def parse_and_clean_content(raw_content: Optional[str]) -> Dict[str, Any]:
    """
    Parses <tool_call> and <think> tags from raw content.
    Returns a dictionary containing:
        - cleaned_content: Content with tags removed.
        - parsed_tool_calls: List of formatted tool calls.
        - reasoning_content: Concatenated content from <think> tags.
    """
    if not raw_content:
        return {
            "cleaned_content": None,
            "parsed_tool_calls": None,
            "reasoning_content": None
        }

    parsed_tool_calls = []
    reasoning_parts = []
    cleaned_parts = []
    last_end = 0

    # Combined regex to find either tag
    # Use non-capturing groups (?:...) for the alternatives
    tag_regex = re.compile(r"(?:<tool_call>(.*?)</tool_call>)|(?:<think>(.*?)</think>)", re.DOTALL)

    for match in tag_regex.finditer(raw_content):
        start, end = match.span()
        # Add content before the current tag
        cleaned_parts.append(raw_content[last_end:start])

        tool_call_content = match.group(1)
        think_content = match.group(2)

        if tool_call_content is not None:
            # Process <tool_call>
            try:
                tool_call_json_str = tool_call_content.strip()
                tool_call_data = json.loads(tool_call_json_str)

                if "name" in tool_call_data and "arguments" in tool_call_data:
                    # Ensure arguments are a JSON string
                    if isinstance(tool_call_data["arguments"], str):
                        try:
                            arguments_obj = json.loads(tool_call_data["arguments"])
                            arguments_str = json.dumps(arguments_obj)
                        except json.JSONDecodeError:
                            arguments_str = tool_call_data["arguments"] # Keep original if not valid JSON
                    else:
                        arguments_str = json.dumps(tool_call_data["arguments"])

                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                    parsed_tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call_data["name"],
                            "arguments": arguments_str,
                        }
                    })
                else:
                    print(f"Warning: Skipping invalid tool call structure: {tool_call_json_str}")
                    # If invalid, treat the raw tag as text content
                    cleaned_parts.append(raw_content[start:end])
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from tool call: {tool_call_json_str}. Error: {e}")
                # Treat the raw tag as text content if JSON parsing fails
                cleaned_parts.append(raw_content[start:end])

        elif think_content is not None:
            # Process <think>
            reasoning_parts.append(think_content.strip())
            # Do not add the <think> tag itself to cleaned_parts

        last_end = end

    # Add any remaining content after the last tag
    cleaned_parts.append(raw_content[last_end:])

    # Join cleaned parts and reasoning parts
    final_cleaned_content = "".join(cleaned_parts) if cleaned_parts else None
    # Strip leading/trailing whitespace that might result from tag removal
    final_cleaned_content = final_cleaned_content.strip() if final_cleaned_content else None
    final_reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

    return {
        "cleaned_content": final_cleaned_content if final_cleaned_content else None, # Return None if empty after stripping
        "parsed_tool_calls": parsed_tool_calls if parsed_tool_calls else None,
        "reasoning_content": final_reasoning_content
    }


# --- API Endpoints ---
@app.post("/v1/chat/completions") # Match OpenAI path convention
@app.post("/chat/completions")    # Keep original path for compatibility
async def chat_completions(request: ChatCompletionRequest):
    if not client:
         raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check configuration.")

    # Convert Pydantic models to dictionaries for OpenAI SDK
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
            # This generator handles parsing tool calls and think tags from content within the stream
            def stream_generator():
                content_buffer = "" # Buffer for accumulating content across chunks
                current_tool_call_index = 0 # To assign index to streamed tool calls
                role_sent = False # Track if role has been sent in the stream
                streamed_tool_calls_exist = False # Track if we yielded any tool calls
                # Combined regex for streaming parsing
                tag_regex_stream = re.compile(r"(<tool_call>.*?</tool_call>)|(<think>.*?</think>)", re.DOTALL)

                try:
                    for chunk in response:
                        # print(f"Raw Chunk: {chunk.model_dump_json()}") # Debugging
                        if not chunk.choices:
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        finish_reason = choice.finish_reason
                        logprobs = choice.logprobs

                        current_chunk_role = delta.role if delta else None
                        current_chunk_content = delta.content if delta else None
                        current_chunk_tool_calls = delta.tool_calls if delta else None # Native tool calls

                        # Determine role for the *first* delta chunk yielded from this raw chunk
                        role_to_yield = None
                        if current_chunk_role and not role_sent:
                            role_to_yield = current_chunk_role
                            role_sent = True
                        elif not role_sent and (current_chunk_content or current_chunk_tool_calls):
                            role_to_yield = "assistant"
                            role_sent = True

                        # --- Step 1: Handle native structured tool calls ---
                        if current_chunk_tool_calls:
                            print(f"Forwarding structured tool call delta: {current_chunk_tool_calls}")
                            structured_tool_chunk_dict = chunk.model_dump(exclude={'choices'})
                            structured_tool_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {
                                    "role": role_to_yield,
                                    "content": None,
                                    "tool_calls": [tc.model_dump(exclude_none=True) for tc in current_chunk_tool_calls]
                                },
                                "finish_reason": None, "logprobs": None
                            }]
                            yield f"data: {json.dumps(structured_tool_chunk_dict)}\n\n"
                            role_to_yield = None # Role sent

                            last_tool_call = current_chunk_tool_calls[-1]
                            if last_tool_call.index is not None:
                                current_tool_call_index = last_tool_call.index + 1
                            streamed_tool_calls_exist = True
                            if current_chunk_content: # Buffer any accompanying content
                                content_buffer += current_chunk_content
                            continue # Process next chunk

                        # --- Step 2: Buffer incoming content ---
                        if current_chunk_content:
                            content_buffer += current_chunk_content

                        # --- Step 3: Process buffer for tags and content ---
                        processed_upto_index = 0
                        while True:
                            # Find the *first* occurrence of either tag in the unprocessed buffer part
                            match = tag_regex_stream.search(content_buffer, pos=processed_upto_index)
                            if not match:
                                # No more tags found in the current buffer
                                break

                            match_start, match_end = match.span()
                            tag_content = match.group(0) # Full tag <tag>content</tag>
                            tool_call_match = match.group(1) # Content of tool_call or None
                            think_match = match.group(2) # Content of think or None

                            # Yield content *before* the found tag
                            prefix = content_buffer[processed_upto_index:match_start]
                            if prefix:
                                print(f"Yielding prefix content delta: '{prefix}'")
                                prefix_chunk_dict = chunk.model_dump(exclude={'choices'})
                                prefix_chunk_dict["choices"] = [{
                                    "index": choice.index,
                                    "delta": {"role": role_to_yield, "content": prefix}, # DELTA CONTENT
                                    "finish_reason": None, "logprobs": None
                                }]
                                yield f"data: {json.dumps(prefix_chunk_dict)}\n\n"
                                role_to_yield = None # Role sent

                            # Process the found tag
                            if tool_call_match:
                                # Attempt to parse the <tool_call> content
                                tool_call_inner_content = re.search(r"<tool_call>(.*?)</tool_call>", tag_content, re.DOTALL).group(1)
                                try:
                                    tool_call_data = json.loads(tool_call_inner_content.strip())
                                    if "name" in tool_call_data and "arguments" in tool_call_data:
                                        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                                        if isinstance(tool_call_data["arguments"], str):
                                             try: arguments_obj = json.loads(tool_call_data["arguments"]); arguments_str = json.dumps(arguments_obj)
                                             except json.JSONDecodeError: arguments_str = tool_call_data["arguments"]
                                        else: arguments_str = json.dumps(tool_call_data["arguments"])

                                        print(f"Yielding parsed tool call (Index {current_tool_call_index}): ID {tool_call_id}, Name {tool_call_data['name']}")
                                        # Yield Name Chunk
                                        tool_chunk_name = chunk.model_dump(exclude={'choices'})
                                        tool_chunk_name["choices"] = [{"index": choice.index,"delta": {"role": role_to_yield,"content": None,"tool_calls": [{"index": current_tool_call_index,"id": tool_call_id,"type": "function","function": {"name": tool_call_data["name"], "arguments": ""}}]},"finish_reason": None, "logprobs": None}]
                                        yield f"data: {json.dumps(tool_chunk_name)}\n\n"
                                        role_to_yield = None # Role sent
                                        # Yield Arguments Chunk
                                        if arguments_str:
                                            tool_chunk_args = chunk.model_dump(exclude={'choices'})
                                            tool_chunk_args["choices"] = [{"index": choice.index,"delta": {"tool_calls": [{"index": current_tool_call_index,"function": {"arguments": arguments_str}}]},"finish_reason": None, "logprobs": None}]
                                            yield f"data: {json.dumps(tool_chunk_args)}\n\n"

                                        current_tool_call_index += 1
                                        streamed_tool_calls_exist = True
                                        processed_upto_index = match_end # Advance past processed tag
                                    else: # Invalid structure
                                        print(f"Warning: Skipping invalid tool call structure in stream: {tool_call_inner_content}")
                                        # Don't advance processed_upto_index, let it be yielded as content later
                                        break # Stop tag processing for this chunk cycle
                                except json.JSONDecodeError as e: # Invalid JSON
                                    print(f"Warning: Failed to parse JSON from tool call in stream: {tool_call_inner_content}. Error: {e}")
                                    # Don't advance processed_upto_index, let it be yielded as content later
                                    break # Stop tag processing for this chunk cycle

                            elif think_match:
                                # Extract thinking content
                                think_inner_content = re.search(r"<think>(.*?)</think>", tag_content, re.DOTALL).group(1).strip()
                                if think_inner_content:
                                    print(f"Yielding reasoning content delta: '{think_inner_content}'")
                                    # Yield custom chunk with reasoning_content
                                    reasoning_chunk_dict = chunk.model_dump(exclude={'choices'})
                                    reasoning_chunk_dict["choices"] = [{
                                        "index": choice.index,
                                        "delta": {"role": role_to_yield, "reasoning_content": think_inner_content}, # CUSTOM FIELD
                                        "finish_reason": None, "logprobs": None
                                    }]
                                    yield f"data: {json.dumps(reasoning_chunk_dict)}\n\n"
                                    role_to_yield = None # Role sent
                                processed_upto_index = match_end # Advance past processed tag

                        # --- Step 4: Yield remaining content in the buffer ---
                        remaining_content = content_buffer[processed_upto_index:]
                        if remaining_content:
                            print(f"Yielding remaining content delta: '{remaining_content}'")
                            remaining_content_chunk_dict = chunk.model_dump(exclude={'choices'})
                            remaining_content_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {"role": role_to_yield, "content": remaining_content}, # DELTA CONTENT
                                "finish_reason": None,
                                "logprobs": None
                            }]
                            yield f"data: {json.dumps(remaining_content_chunk_dict)}\n\n"
                            role_to_yield = None # Role sent

                        # Update buffer: remove the processed/yielded part
                        content_buffer = content_buffer[processed_upto_index:]

                        # --- Step 5: Handle Finish Reason ---
                        if finish_reason:
                            print(f"Received finish_reason: {finish_reason}")
                            if content_buffer:
                                print(f"Warning: Stream finished with unprocessed content in buffer: '{content_buffer}'") # Should ideally be empty

                            final_chunk_dict = chunk.model_dump(exclude={'choices'})
                            final_reason = finish_reason
                            if streamed_tool_calls_exist and finish_reason == 'stop':
                                print(f"Overriding finish_reason from '{finish_reason}' to 'tool_calls'.")
                                final_reason = 'tool_calls'
                            elif finish_reason == 'tool_calls':
                                streamed_tool_calls_exist = True

                            final_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {},
                                "finish_reason": final_reason,
                                "logprobs": logprobs
                            }]
                            yield f"data: {json.dumps(final_chunk_dict)}\n\n"
                            break # End stream processing

                except Exception as e: # Catch potential errors during streaming
                    print(f"Error during streaming: {e}")
                    import traceback
                    traceback.print_exc()
                    error_payload = {"error": {"message": f"Internal proxy error during stream: {e}", "type": "proxy_error", "code": 500}}
                    try:
                        yield f"data: {json.dumps(error_payload)}\n\n"
                    except Exception as yield_e:
                        print(f"Error yielding error message: {yield_e}")
                finally:
                    print("Stream finished. Yielding [DONE].")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # --- Handle Non-Streaming Response ---
        else:
            print(f"Received non-streaming response from downstream: {response}")
            response_payload = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created or int(time.time()),
                "model": response.model,
                "choices": [],
                "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
                "system_fingerprint": response.system_fingerprint,
            }

            final_choices = []
            for i, choice in enumerate(response.choices):
                message = choice.message
                original_content = message.content
                original_finish_reason = choice.finish_reason

                # Parse content for tags and clean it
                parsed_data = parse_and_clean_content(original_content)
                cleaned_content = parsed_data["cleaned_content"]
                parsed_tools = parsed_data["parsed_tool_calls"]
                reasoning_content = parsed_data["reasoning_content"]

                # Initialize choice message structure
                choice_message_dict = {
                    "role": message.role or "assistant",
                    "content": cleaned_content, # Use cleaned content
                    "tool_calls": None, # Initialize
                    # Add reasoning_content if it exists
                    **({"reasoning_content": reasoning_content} if reasoning_content else {})
                }

                # Determine final tool calls and finish reason
                final_tool_calls = None
                final_finish_reason = original_finish_reason

                if parsed_tools:
                    # Use tool calls parsed from content
                    print(f"Using tool calls parsed from content in non-streaming choice {i}: {parsed_tools}")
                    final_tool_calls = parsed_tools
                    if original_finish_reason != "tool_calls":
                        print(f"Overriding finish_reason from '{original_finish_reason}' to 'tool_calls' due to parsed content.")
                        final_finish_reason = "tool_calls"
                elif message.tool_calls:
                    # Use pre-structured tool calls from downstream if no tags were parsed
                    print(f"Using pre-structured tool calls from downstream choice {i}: {message.tool_calls}")
                    final_tool_calls = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
                    # Ensure content is None if structured tool calls exist and no other content remains after cleaning
                    if final_tool_calls and not cleaned_content:
                         choice_message_dict["content"] = None


                choice_message_dict["tool_calls"] = final_tool_calls

                # Append the processed choice
                final_choices.append({
                    "message": choice_message_dict,
                    "index": i,
                    "finish_reason": final_finish_reason,
                    "logprobs": choice.logprobs.model_dump(exclude_none=True) if choice.logprobs else None
                })

            response_payload["choices"] = final_choices
            print(f"Returning processed non-streaming response: {response_payload}")
            return response_payload

    except APIStatusError as e:
        print(f"Downstream API Error: Status={e.status_code} Response={e.response}")
        raise HTTPException(status_code=e.status_code, detail=f"Downstream API error: {e.message or e.response.text}")
    except APIConnectionError as e:
        print(f"Downstream Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to downstream API: {OPENAI_BASE_URL}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OpenAI Proxy API with Tool & Think Tag Parsing is running. Send POST requests to /chat/completions or /v1/chat/completions"}

# --- Run the application ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000)) # Allow configuring port via environment variable
    host = os.getenv("HOST", "0.0.0.0") # Allow configuring host
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

# --- Dependencies ---
# pip install fastapi uvicorn openai python-dotenv requests httpx pydantic
# Optional: pip install lunary

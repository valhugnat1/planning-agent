import time, os, re, json, uuid
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, APIConnectionError, APIStatusError
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI(title="OpenAI Proxy API with Tool & Think Tag Parsing")

# API Configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1")
OPENAI_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        print(f"OpenAI client initialized pointing to: {OPENAI_BASE_URL}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")

# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

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
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

def parse_and_clean_content(raw_content: Optional[str]) -> Dict[str, Any]:
    """Parse <tool_call> and <think> tags from raw content."""
    if not raw_content:
        return {"cleaned_content": None, "parsed_tool_calls": None, "reasoning_content": None}

    parsed_tool_calls = []
    reasoning_parts = []
    cleaned_parts = []
    last_end = 0

    tag_regex = re.compile(r"(?:<tool_call>(.*?)</tool_call>)|(?:<think>(.*?)</think>)", re.DOTALL)

    for match in tag_regex.finditer(raw_content):
        start, end = match.span()
        cleaned_parts.append(raw_content[last_end:start])  # Content before tag

        tool_call_content = match.group(1)
        think_content = match.group(2)

        if tool_call_content:  # Process <tool_call>
            try:
                tool_call_data = json.loads(tool_call_content.strip())
                if "name" in tool_call_data and "arguments" in tool_call_data:
                    # Process arguments
                    if isinstance(tool_call_data["arguments"], str):
                        try: 
                            arguments_obj = json.loads(tool_call_data["arguments"])
                            arguments_str = json.dumps(arguments_obj)
                        except json.JSONDecodeError: 
                            arguments_str = tool_call_data["arguments"]
                    else: 
                        arguments_str = json.dumps(tool_call_data["arguments"])
                    
                    # Create tool call
                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                    parsed_tool_calls.append({
                        "id": tool_call_id, 
                        "type": "function", 
                        "function": {
                            "name": tool_call_data["name"], 
                            "arguments": arguments_str
                        }
                    })
                else:
                    cleaned_parts.append(raw_content[start:end])  # Invalid structure, treat as content
            except json.JSONDecodeError:
                cleaned_parts.append(raw_content[start:end])  # Invalid JSON, treat as content
        elif think_content:  # Process <think>
            reasoning_parts.append(think_content.strip())

        last_end = end

    cleaned_parts.append(raw_content[last_end:])  # Content after last tag
    final_cleaned_content = "".join(cleaned_parts).strip() or None
    final_reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

    return {
        "cleaned_content": final_cleaned_content,
        "parsed_tool_calls": parsed_tool_calls if parsed_tool_calls else None,
        "reasoning_content": final_reasoning_content
    }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check configuration.")

    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    kwargs = request.model_dump(exclude_none=True)
    kwargs["messages"] = messages

    try:
        response = client.chat.completions.create(**kwargs)

        # Handle Streaming Response
        if request.stream:
            def stream_generator():
                # State Variables
                content_buffer = ""           # Holds raw content from current chunk
                current_tool_call_index = 0   # Index for yielded tool calls
                role_sent = False             # Track if initial role delta was sent
                streamed_tool_calls_exist = False  # Track if any tool calls yielded
                in_think_block = False        # Inside <think>...</think>
                in_tool_call_block = False    # Inside <tool_call>...</tool_call>
                tool_call_buffer = ""         # Buffer for content between tool call tags

                try:
                    for chunk in response:
                        # Process Chunk Metadata
                        if not chunk.choices:
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        finish_reason = choice.finish_reason

                        current_chunk_role = delta.role if delta else None
                        current_chunk_content = delta.content if delta else None
                        current_chunk_tool_calls = delta.tool_calls if delta else None

                        # Determine first delta role
                        role_to_yield_on_first_delta = None
                        if current_chunk_role and not role_sent:
                            role_to_yield_on_first_delta = current_chunk_role
                        elif not role_sent and (current_chunk_content or current_chunk_tool_calls):
                            role_to_yield_on_first_delta = "assistant"

                        # Handle Native Tool Calls from downstream
                        if current_chunk_tool_calls:
                            # Reset state for native tool calls
                            in_think_block = in_tool_call_block = False
                            tool_call_buffer = ""

                            structured_tool_chunk_dict = chunk.model_dump(exclude={'choices'})
                            structured_tool_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": {
                                    "role": role_to_yield_on_first_delta,
                                    "content": None,
                                    "tool_calls": [tc.model_dump(exclude_none=True) for tc in current_chunk_tool_calls]
                                },
                                "finish_reason": None, 
                                "logprobs": None
                            }]
                            yield f"data: {json.dumps(structured_tool_chunk_dict)}\n\n"
                            if role_to_yield_on_first_delta: 
                                role_sent = True

                            last_tool_call = current_chunk_tool_calls[-1]
                            if last_tool_call.index is not None:
                                current_tool_call_index = last_tool_call.index + 1
                            streamed_tool_calls_exist = True
                            if current_chunk_content:  # Buffer any accompanying content
                                content_buffer += current_chunk_content
                            continue

                        # Buffer Incoming Content
                        if current_chunk_content:
                            content_buffer += current_chunk_content

                        # Process Buffer Segment by Segment
                        processed_upto_buffer_idx = 0
                        while processed_upto_buffer_idx < len(content_buffer):
                            # Find the earliest relevant tag in the remaining buffer
                            first_match_pos = float('inf')
                            action = "content"  # Default action
                            tag_len = 0        # Length of the tag found

                            # Determine next action based on current state and tags found
                            if not in_think_block and not in_tool_call_block:
                                think_open_pos = content_buffer.find("<think>", processed_upto_buffer_idx)
                                tool_open_pos = content_buffer.find("<tool_call>", processed_upto_buffer_idx)

                                if think_open_pos != -1 and think_open_pos < first_match_pos:
                                    first_match_pos = think_open_pos
                                    action = "think_open"
                                    tag_len = len("<think>")
                                if tool_open_pos != -1 and tool_open_pos < first_match_pos:
                                    first_match_pos = tool_open_pos
                                    action = "tool_call_open"
                                    tag_len = len("<tool_call>")
                            elif in_think_block:
                                think_close_pos = content_buffer.find("</think>", processed_upto_buffer_idx)
                                if think_close_pos != -1:
                                    first_match_pos = think_close_pos
                                    action = "think_close"
                                    tag_len = len("</think>")
                            elif in_tool_call_block:
                                tool_close_pos = content_buffer.find("</tool_call>", processed_upto_buffer_idx)
                                if tool_close_pos != -1:
                                    first_match_pos = tool_close_pos
                                    action = "tool_call_close"
                                    tag_len = len("</tool_call>")

                            # Process content segment before the tag (or all remaining content)
                            segment_end = first_match_pos if first_match_pos != float('inf') else len(content_buffer)
                            content_segment = content_buffer[processed_upto_buffer_idx:segment_end]

                            if content_segment:
                                if in_tool_call_block:
                                    # Append to tool call buffer, don't yield yet
                                    tool_call_buffer += content_segment
                                else:
                                    delta_payload = {"content": ""}
                                    # Yield content (potentially also as reasoning)
                                    if in_think_block:
                                        delta_payload["reasoning_content"] = content_segment
                                    else: 
                                        delta_payload = {"content": content_segment}

                                    yield_choice = {
                                        "index": choice.index,
                                        "delta": delta_payload,
                                        "finish_reason": None, 
                                        "logprobs": None
                                    }
                                    if role_to_yield_on_first_delta and not role_sent:
                                        yield_choice["delta"]["role"] = role_to_yield_on_first_delta
                                        role_sent = True

                                    yield_chunk_dict = chunk.model_dump(exclude={'choices'})
                                    yield_chunk_dict["choices"] = [yield_choice]
                                    yield f"data: {json.dumps(yield_chunk_dict)}\n\n"

                            processed_upto_buffer_idx = segment_end  # Move buffer index past the processed segment

                            # Handle the tag action
                            if action == "think_open":
                                in_think_block = True
                                processed_upto_buffer_idx += tag_len
                            elif action == "think_close":
                                in_think_block = False
                                processed_upto_buffer_idx += tag_len
                            elif action == "tool_call_open":
                                in_tool_call_block = True
                                tool_call_buffer = ""  # Reset buffer
                                processed_upto_buffer_idx += tag_len
                            elif action == "tool_call_close":
                                in_tool_call_block = False
                                processed_upto_buffer_idx += tag_len

                                # Attempt to parse and yield the tool call
                                try:
                                    tool_call_data = json.loads(tool_call_buffer.strip())
                                    if "name" in tool_call_data and "arguments" in tool_call_data:
                                        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                                        # Process arguments
                                        if isinstance(tool_call_data["arguments"], str):
                                            try: 
                                                arguments_obj = json.loads(tool_call_data["arguments"])
                                                arguments_str = json.dumps(arguments_obj)
                                            except json.JSONDecodeError: 
                                                arguments_str = tool_call_data["arguments"]
                                        else: 
                                            arguments_str = json.dumps(tool_call_data["arguments"])

                                        # Yield Name Chunk
                                        tool_chunk_name = chunk.model_dump(exclude={'choices'})
                                        tool_chunk_name_delta = {
                                            "role": role_to_yield_on_first_delta if not role_sent else None, 
                                            "content": None, 
                                            "tool_calls": [{
                                                "index": current_tool_call_index,
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call_data["name"], 
                                                    "arguments": ""
                                                }
                                            }]
                                        }
                                        # Remove None values from delta
                                        tool_chunk_name_delta = {k:v for k,v in tool_chunk_name_delta.items() if v is not None}
                                        tool_chunk_name["choices"] = [{
                                            "index": choice.index,
                                            "delta": tool_chunk_name_delta,
                                            "finish_reason": None, 
                                            "logprobs": None
                                        }]
                                        yield f"data: {json.dumps(tool_chunk_name)}\n\n"
                                        if role_to_yield_on_first_delta and not role_sent: 
                                            role_sent = True

                                        # Yield Arguments Chunk
                                        if arguments_str:  # Only yield if arguments exist
                                            tool_chunk_args = chunk.model_dump(exclude={'choices'})
                                            tool_chunk_args["choices"] = [{
                                                "index": choice.index,
                                                "delta": {
                                                    "tool_calls": [{
                                                        "index": current_tool_call_index,
                                                        "function": {"arguments": arguments_str}
                                                    }]
                                                },
                                                "finish_reason": None, 
                                                "logprobs": None
                                            }]
                                            yield f"data: {json.dumps(tool_chunk_args)}\n\n"

                                        current_tool_call_index += 1
                                        streamed_tool_calls_exist = True
                                    else:  # Invalid structure
                                        raise ValueError("Parsed tool call missing 'name' or 'arguments'")
                                except (json.JSONDecodeError, ValueError) as e:
                                    # If parsing fails or structure invalid, yield the raw content
                                    raw_tool_content = f"<tool_call>{tool_call_buffer}</tool_call>"
                                    delta_payload = {"content": raw_tool_content}
                                    yield_choice = {
                                        "index": choice.index,
                                        "delta": delta_payload,
                                        "finish_reason": None, 
                                        "logprobs": None
                                    }
                                    if role_to_yield_on_first_delta and not role_sent:
                                        yield_choice["delta"]["role"] = role_to_yield_on_first_delta
                                        role_sent = True
                                    yield_chunk_dict = chunk.model_dump(exclude={'choices'})
                                    yield_chunk_dict["choices"] = [yield_choice]
                                    yield f"data: {json.dumps(yield_chunk_dict)}\n\n"

                                tool_call_buffer = ""  # Clear buffer after processing

                        # Update main content buffer
                        content_buffer = content_buffer[processed_upto_buffer_idx:]

                        # Handle Finish Reason
                        if finish_reason:
                            # Handle incomplete states
                            if in_tool_call_block and tool_call_buffer:
                                # Yield remaining tool_call_buffer as raw content if incomplete
                                raw_tool_content = f"<tool_call>{tool_call_buffer}"
                                yield_choice = {
                                    "index": choice.index,
                                    "delta": {"content": raw_tool_content},
                                    "finish_reason": None, 
                                    "logprobs": None
                                }
                                if role_to_yield_on_first_delta and not role_sent:
                                    yield_choice["delta"]["role"] = role_to_yield_on_first_delta
                                    role_sent = True
                                yield_chunk_dict = chunk.model_dump(exclude={'choices'})
                                yield_chunk_dict["choices"] = [yield_choice]
                                yield f"data: {json.dumps(yield_chunk_dict)}\n\n"

                            # Determine final reason and yield final chunk
                            final_chunk_dict = chunk.model_dump(exclude={'choices'})
                            final_reason = finish_reason
                            if streamed_tool_calls_exist and finish_reason == 'stop':
                                final_reason = 'tool_calls'

                            final_delta = {}
                            if role_to_yield_on_first_delta and not role_sent:
                                final_delta["role"] = role_to_yield_on_first_delta

                            final_chunk_dict["choices"] = [{
                                "index": choice.index,
                                "delta": final_delta,
                                "finish_reason": final_reason,
                                "logprobs": choice.logprobs
                            }]
                            yield f"data: {json.dumps(final_chunk_dict)}\n\n"
                            break  # End stream processing

                except Exception as e:
                    # Handle streaming errors
                    import traceback
                    traceback.print_exc()
                    error_payload = {
                        "error": {
                            "message": f"Internal proxy error during stream: {e}", 
                            "type": "proxy_error", 
                            "code": 500
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                finally:
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # Handle Non-Streaming Response
        else:
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

                parsed_data = parse_and_clean_content(original_content)
                cleaned_content = parsed_data["cleaned_content"]
                parsed_tools = parsed_data["parsed_tool_calls"]
                reasoning_content = parsed_data["reasoning_content"]

                choice_message_dict = {
                    "role": message.role or "assistant",
                    "content": cleaned_content,
                    "tool_calls": None
                }
                
                if reasoning_content:
                    choice_message_dict["reasoning_content"] = reasoning_content

                final_tool_calls = None
                final_finish_reason = original_finish_reason

                if parsed_tools:
                    final_tool_calls = parsed_tools
                    # Content should be None if only tool calls remain after cleaning
                    if final_tool_calls and not cleaned_content:
                        choice_message_dict["content"] = None
                    if original_finish_reason != "tool_calls": 
                        final_finish_reason = "tool_calls"
                elif message.tool_calls:
                    final_tool_calls = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
                    # Ensure content is None if structured tool calls exist and no other content remains after cleaning
                    if final_tool_calls and not cleaned_content:
                        choice_message_dict["content"] = None

                choice_message_dict["tool_calls"] = final_tool_calls

                final_choices.append({
                    "message": choice_message_dict,
                    "index": i,
                    "finish_reason": final_finish_reason,
                    "logprobs": choice.logprobs.model_dump(exclude_none=True) if choice.logprobs else None
                })

            response_payload["choices"] = final_choices
            return response_payload

    except APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Downstream API error: {e.message or e.response.text}")
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to downstream API: {OPENAI_BASE_URL}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OpenAI Proxy API with Tool & Think Tag Parsing is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
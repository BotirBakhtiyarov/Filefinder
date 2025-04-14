import json

def get_chat_payload(model, prompt, files=[]):
    """Generate the request payload based on the selected AI model."""
    if model == "Regular":
        return {
            "inputs": {},
            "query": prompt,
            "response_mode": "streaming",
            "conversation_id": "",
            "user": "abc-123",
            "files": files
        }
    elif model == "ChatGPT":
        return {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
    elif model == "DeepSeek":
        return {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
    elif model == "Ollama":
        return {
            "model": "llama2",
            "prompt": prompt,
            "stream": True
        }
    else:
        raise ValueError(f"Unknown AI model: {model}")

def parse_streaming_response(model, response):
    """Parse the streaming response based on the selected AI model."""
    if model == "Regular":
        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                data = json.loads(line[6:].decode("utf-8"))
                if "answer" in data:
                    yield data["answer"]
    elif model in ["ChatGPT", "DeepSeek"]:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    elif model == "Ollama":
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
    else:
        raise ValueError(f"Unknown AI model: {model}")
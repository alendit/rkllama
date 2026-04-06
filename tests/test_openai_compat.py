import importlib.util
import json
from pathlib import Path
import sys
import types


FORMAT_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "rkllama" / "api" / "format_utils.py"
)

spec = importlib.util.spec_from_file_location("rkllama_format_utils", FORMAT_UTILS_PATH)
format_utils = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(format_utils)

normalize_openai_format_spec = format_utils.normalize_openai_format_spec
openai_to_ollama_chat_request = format_utils.openai_to_ollama_chat_request
openai_to_ollama_generate_request = format_utils.openai_to_ollama_generate_request
ollama_embedding_to_openai_v1_embeddingns = (
    format_utils.ollama_embedding_to_openai_v1_embeddingns
)
responses_input_to_messages = format_utils.responses_input_to_messages


SERVER_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "rkllama" / "api" / "server_utils.py"
)


def _load_server_utils_module(monkeypatch):
    rkllama_pkg = types.ModuleType("rkllama")
    rkllama_pkg.__path__ = []  # type: ignore[attr-defined]

    api_pkg = types.ModuleType("rkllama.api")
    api_pkg.__path__ = []  # type: ignore[attr-defined]

    config_module = types.ModuleType("rkllama.config")
    config_module.is_debug_mode = lambda: False
    config_module.get_path = lambda name: "/tmp"
    rkllama_pkg.config = config_module

    variables_module = types.ModuleType("rkllama.api.variables")
    model_utils_module = types.ModuleType("rkllama.api.model_utils")
    model_utils_module.get_property_modelfile = lambda *args, **kwargs: None

    format_utils_module = types.ModuleType("rkllama.api.format_utils")
    format_utils_module.create_format_instruction = lambda *args, **kwargs: None
    format_utils_module.validate_format_response = lambda *args, **kwargs: None
    format_utils_module.get_tool_calls = lambda *args, **kwargs: []
    format_utils_module.handle_ollama_response = lambda *args, **kwargs: {}
    format_utils_module.handle_ollama_embedding_response = lambda *args, **kwargs: {}
    format_utils_module.get_base64_image_from_pil = lambda *args, **kwargs: None
    format_utils_module.get_url_image_from_pil = lambda *args, **kwargs: None
    format_utils_module.responses_input_to_messages = lambda *args, **kwargs: []

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = object

    monkeypatch.setitem(sys.modules, "rkllama", rkllama_pkg)
    monkeypatch.setitem(sys.modules, "rkllama.api", api_pkg)
    monkeypatch.setitem(sys.modules, "rkllama.config", config_module)
    monkeypatch.setitem(sys.modules, "rkllama.api.variables", variables_module)
    monkeypatch.setitem(sys.modules, "rkllama.api.model_utils", model_utils_module)
    monkeypatch.setitem(sys.modules, "rkllama.api.format_utils", format_utils_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    spec = importlib.util.spec_from_file_location(
        "rkllama.api.server_utils", SERVER_UTILS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_server_utils_imports_cleanly(monkeypatch):
    module = _load_server_utils_module(monkeypatch)

    assert hasattr(module, "ResponsesEndpointHandler")


def test_responses_handler_stringifies_mixed_content(monkeypatch):
    module = _load_server_utils_module(monkeypatch)

    assert module.ResponsesEndpointHandler._stringify_content(
        [{"text": "hello"}, {"content": "world"}, 123]
    ) == "hello\nworld\n123"


def test_normalize_openai_format_spec_json_object():
    assert normalize_openai_format_spec({"type": "json_object"}) == "json"


def test_normalize_openai_format_spec_json_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    assert normalize_openai_format_spec({"type": "json_schema", "schema": schema}) == schema


def test_openai_to_ollama_chat_request_maps_response_format():
    payload = {
        "model": "qwen3:0.6b",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }

    result = openai_to_ollama_chat_request(payload)

    assert result["format"] == "json"
    assert "response_format" not in result


def test_openai_to_ollama_generate_request_supports_images():
    payload = {
        "model": "qwen3:0.6b",
        "prompt": "describe this",
        "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    }

    result = openai_to_ollama_generate_request(payload)

    assert result["images"] == ["data:image/png;base64,AAAA"]


def test_embedding_response_preserves_each_vector():
    payload = {
        "model": "qwen3-embedding:0.6b",
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "prompt_eval_count": 11,
    }

    response = ollama_embedding_to_openai_v1_embeddingns(payload)

    assert response["object"] == "list"
    assert response["model"] == "qwen3-embedding:0.6b"
    assert response["usage"] == {"prompt_tokens": 11, "total_tokens": 11}
    assert response["data"] == [
        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
        {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
    ]


def test_responses_input_to_messages_translates_tool_history():
    messages = responses_input_to_messages(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "think"}],
                "encrypted_content": "think",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": {"city": "Tokyo"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            },
            {
                "type": "message",
                "role": "developer",
                "content": "be brief",
            },
        ],
        instructions="Use short answers",
    )

    assert messages[0] == {"role": "system", "content": "Use short answers"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["thinking"] == "think"
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": json.dumps({"city": "Tokyo"}),
    }
    assert messages[2] == {
        "role": "tool",
        "content": "sunny",
        "tool_call_id": "call_1",
    }
    assert messages[3] == {"role": "system", "content": "be brief"}

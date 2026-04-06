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

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise NotImplementedError

    transformers_module.AutoTokenizer = DummyAutoTokenizer

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


def test_server_utils_tokenizer_falls_back_to_slow(monkeypatch):
    module = _load_server_utils_module(monkeypatch)
    calls = []

    class FakeTokenizer:
        pass

    def fake_from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        if len(calls) == 1:
            raise AttributeError("'NoneType' object has no attribute 'endswith'")
        return FakeTokenizer()

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tokenizer = module._load_tokenizer_with_fallback("phi3")

    assert isinstance(tokenizer, FakeTokenizer)
    assert calls == [
        ("phi3", {"trust_remote_code": True}),
        ("phi3", {"trust_remote_code": True, "use_fast": False}),
    ]


def test_server_utils_prefers_tokenizer_modelfile_value(monkeypatch, tmp_path):
    module = _load_server_utils_module(monkeypatch)
    model_root = tmp_path / "models"
    model_root.mkdir()
    local_tokenizer_path = model_root / "phi3:mini" / "tokenizer"

    monkeypatch.setattr(module.rkllama.config, "get_path", lambda name: str(model_root))

    def fake_get_property(model_name, key, models_path):
        assert model_name == "phi3:mini"
        assert models_path == str(model_root)
        if key == "TOKENIZER":
            return '"microsoft/Phi-3-mini-4k-instruct"'
        if key == "HUGGINGFACE_PATH":
            return '"GatekeeperZA/Phi-3-mini-4k-instruct-RKLLM-v1.2.3"'
        return None

    calls = []

    class FakeTokenizer:
        def save_pretrained(self, path):
            calls.append(("save_pretrained", path))

    monkeypatch.setattr(module, "get_property_modelfile", fake_get_property)
    monkeypatch.setattr(module.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(
        module,
        "_load_tokenizer_with_fallback",
        lambda source: calls.append(("load", source)) or FakeTokenizer(),
    )

    tokenizer = module.EndpointHandler.get_tokenizer("phi3:mini")

    assert isinstance(tokenizer, FakeTokenizer)
    assert calls == [
        ("load", "microsoft/Phi-3-mini-4k-instruct"),
        ("save_pretrained", str(local_tokenizer_path)),
    ]


def test_responses_handler_emits_completed_event_after_stream_error(monkeypatch):
    module = _load_server_utils_module(monkeypatch)

    class FakeChatResponse:
        def __init__(self):
            self.response = self._iter_chunks()

        @staticmethod
        def _iter_chunks():
            yield json.dumps(
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "done": False,
                }
            ).encode("utf-8")
            raise RuntimeError("upstream stream crashed")

    monkeypatch.setattr(
        module.ChatEndpointHandler,
        "handle_request",
        classmethod(lambda cls, **kwargs: FakeChatResponse()),
    )
    monkeypatch.setattr(module, "stream_with_context", lambda value: value)

    response = module.ResponsesEndpointHandler.handle_request(
        model_name="qwen3:0.6b",
        input_data="hi",
        stream=True,
        request_data={},
    )

    chunks = list(response.response)
    payloads = [json.loads(chunk.split("data: ", 1)[1]) for chunk in chunks]

    assert payloads[0]["type"] == "response.created"
    assert payloads[1]["type"] == "response.output_text.delta"
    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["status"] == "incomplete"
    assert payloads[-1]["error"] == {
        "type": "server_error",
        "message": "upstream stream crashed",
    }


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

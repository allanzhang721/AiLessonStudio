import unittest

from explain_it.app import IMAGE_PROVIDERS, TEXT_PROVIDERS
from pipeline.clients import build_image_client, chat_completion
from pipeline.image_pipeline import _find_wanx_image_url


class _Message:
    content = "deepseek ok"


class _Choice:
    message = _Message()


class _Completion:
    choices = [_Choice()]


class _Completions:
    def create(self, **kwargs):
        return _Completion()


class _Chat:
    completions = _Completions()


class _Responses:
    def create(self, **kwargs):
        raise AssertionError("DeepSeek must not use the OpenAI Responses endpoint")


class _DeepSeekClient:
    responses = _Responses()
    chat = _Chat()


class ProviderSelectionTests(unittest.TestCase):
    def test_current_provider_choices_are_exposed(self):
        self.assertIn("OpenAI", TEXT_PROVIDERS)
        self.assertIn("DeepSeek", TEXT_PROVIDERS)
        self.assertIn("OpenAI", IMAGE_PROVIDERS)
        self.assertIn("Alibaba Wan", IMAGE_PROVIDERS)
        self.assertIn("deepseek-v4-flash", TEXT_PROVIDERS["DeepSeek"]["models"].values())
        self.assertIn("wan2.7-image", IMAGE_PROVIDERS["Alibaba Wan"]["models"].values())

    def test_wan_client_uses_explicit_session_key(self):
        client = build_image_client("wanx", api_key="visitor-key")
        self.assertEqual(client["api_key"], "visitor-key")
        self.assertEqual(client["provider"], "wanx")

    def test_deepseek_routes_to_chat_completions(self):
        result = chat_completion(_DeepSeekClient(), "deepseek-v4-flash", "hello")
        self.assertEqual(result, "deepseek ok")

    def test_wan_response_url_extraction(self):
        payload = {"output": {"choices": [{"message": {"content": [{"image": "https://example.test/frame.png"}]}}]}}
        self.assertEqual(_find_wanx_image_url(payload), "https://example.test/frame.png")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
import wave
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from explain_it.app import _waiting_tip
from pipeline.lesson_service import build_concept_map
from pipeline.video_pipeline import (
    build_narration_script,
    calculate_scene_durations,
    images_to_video,
    synthesize_clean_voiceover,
)


class _SpeechResponse:
    def stream_to_file(self, path: str) -> None:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 2400)


class _SpeechEndpoint:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _SpeechResponse()


class _Audio:
    def __init__(self) -> None:
        self.speech = _SpeechEndpoint()


class _TTSClient:
    def __init__(self) -> None:
        self.audio = _Audio()


class VideoQualityTests(unittest.TestCase):
    def test_waiting_tips_are_short_stable_and_stage_aware(self):
        first = _waiting_tip("draft", 10, "Physics")
        self.assertEqual(first, _waiting_tip("draft", 10, "Physics"))
        self.assertNotEqual(first, _waiting_tip("gate", 10, "Physics"))
        self.assertLess(len(first), 180)
    def test_concept_map_classifies_learning_directions(self):
        concept_map = build_concept_map({
            "title": "Forces",
            "related_topics": [
                {"topic": "Vectors", "relationship": "prerequisite", "why_useful": "Resolve directions."},
                {"topic": "Momentum", "relationship": "next step", "why_useful": "Connect force and time."},
            ],
        }, "Physics")
        self.assertEqual([node["direction"] for node in concept_map["nodes"]], ["in", "out"])
        self.assertTrue(all(node["challenge"] and node["sources"] for node in concept_map["nodes"]))
    def _plan(self):
        return {
            "captions": [
                "A net force changes an object's motion.",
                "For the same mass, a larger net force creates greater acceleration.",
            ]
        }

    def test_narration_is_natural_and_scene_timing_matches_audio(self):
        plan = self._plan()
        script = build_narration_script(plan)
        self.assertTrue(script.startswith("First,"))
        self.assertNotIn("Step 1", script)
        durations = calculate_scene_durations(plan, total_seconds=8.0)
        self.assertEqual(len(durations), 2)
        self.assertTrue(all(value >= 3.2 for value in durations))
        self.assertGreaterEqual(sum(durations), 8.6)

    def test_voiceover_uses_controllable_lossless_tts(self):
        client = _TTSClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = synthesize_clean_voiceover(client, self._plan(), Path(temp_dir), voice="cedar")
            self.assertIsNotNone(audio_path)
            self.assertTrue(audio_path.exists())
        args = client.audio.speech.kwargs
        self.assertEqual(args["model"], "gpt-4o-mini-tts")
        self.assertEqual(args["voice"], "cedar")
        self.assertEqual(args["response_format"], "wav")
        self.assertIn("teacher", args["instructions"])

    def test_renderer_outputs_h264_motion_video_at_requested_canvas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames = []
            audio_path = root / "narration.wav"
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00\x00" * 4800)
            for index, color in enumerate(("#2563EB", "#0F766E"), start=1):
                image = Image.new("RGB", (480, 320), color)
                ImageDraw.Draw(image).text((30, 30), f"Lesson frame {index}", fill="white")
                path = root / f"frame_{index}.png"
                image.save(path)
                frames.append(path)
            output = images_to_video(
                frames,
                root / "lesson.mp4",
                audio_path=audio_path,
                scene_durations=[0.6, 0.6],
                resolution=(320, 180),
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            reader = imageio.get_reader(str(output))
            try:
                metadata = reader.get_meta_data()
                self.assertEqual(tuple(metadata["size"]), (320, 180))
                self.assertAlmostEqual(float(metadata["fps"]), 30.0, delta=0.2)
            finally:
                reader.close()

    def test_streamlit_opens_directly_in_api_mode(self):
        app = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        self.assertFalse(app.exception)
        labels = [item.label for item in app.selectbox]
        self.assertIn("Text provider", labels)
        self.assertNotIn("Image provider", labels)
        self.assertIn("Teaching voice", labels)
        self.assertIn("Add cited web research", [item.label for item in app.toggle])
        self.assertFalse(any("Demo" in item.label for item in app.button))
        self.assertEqual(len(app.get("progress")), 6)
        self.assertEqual(len(app.dataframe), 0)
        self.assertTrue(any("checker tests teach us" in str(item.value).lower() for item in app.markdown))
        api_key_labels = [item.label for item in app.text_input if "API key" in item.label]
        self.assertEqual(api_key_labels, ["Text API key"])
        app.selectbox[0].select("DeepSeek").run()
        self.assertFalse(app.exception)
        api_key_labels = [item.label for item in app.text_input if "API key" in item.label]
        self.assertEqual(api_key_labels, ["Text API key"])
        self.assertFalse(any("Image provider" == item.label for item in app.selectbox))
        self.assertFalse(any("Teaching voice" == item.label for item in app.selectbox))
        self.assertTrue(any("text-only" in str(item.value).lower() for item in app.warning))
        add_image = next(item for item in app.toggle if item.label == "Add image API (optional)")
        add_image.set_value(True).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Image provider" == item.label for item in app.selectbox))
        api_key_labels = [item.label for item in app.text_input if "API key" in item.label]
        self.assertEqual(api_key_labels, ["Text API key", "Image API key"])
        self.assertTrue(any("Teaching voice" == item.label for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()

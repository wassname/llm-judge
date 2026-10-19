"""Keep creator precision policy and model-specific reasoning explicit. -- PI/OpenAI"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ModelOutput, get_model
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from openai.types.chat import ChatCompletion

from llm_judge import judge
import test_runtime as fixtures


def endpoint(provider, quantization, **changes):
    return {"provider_name": provider, "tag": provider.lower() + "/test", "quantization": quantization,
            "supported_parameters": ["structured_outputs", "response_format", "max_tokens", "temperature", "reasoning"],
            "max_completion_tokens": 131072, "service_tier": None, **changes}


class ProviderPolicy(unittest.TestCase):
    def test_creator_unknown_is_allowed_but_other_creators_and_hosts_are_not_exempt(self):
        creator = endpoint("DeepSeek", "unknown")
        host = endpoint("DeepInfra", "fp8")
        selected = judge.select_endpoint([host, creator], creator="DeepSeek", parameters={"structured_outputs"}, max_tokens=64000)
        self.assertIs(selected, creator)
        self.assertNotIn("quantizations", judge.provider_options(creator, "DeepSeek"))
        for provider in ("DeepInfra", "Fireworks", "Groq", "OpenAI"):
            for precision in ("unknown", "int8", "int4", "fp4", "fp6", "mxfp4"):
                with self.subTest(provider=provider, precision=precision), self.assertRaises(ValueError):
                    judge.provider_options(endpoint(provider, precision), "DeepSeek")
        for precision in judge.FLOAT_PRECISIONS:
            options = judge.provider_options(endpoint("DeepInfra", precision), "DeepSeek")
            self.assertEqual(options["quantizations"], [precision])
            self.assertFalse(options["allow_fallbacks"])
            self.assertTrue(options["require_parameters"])

    def test_preference_does_not_override_precision_parameters_or_token_capacity(self):
        eligible = endpoint("DeepInfra", "bf16")
        candidates = [endpoint("DeepSeek", "unknown", max_completion_tokens=16000),
                      endpoint("DeepSeek", "unknown", supported_parameters=[]),
                      endpoint("Fireworks", "unknown"), endpoint("OtherHost", "fp32"), eligible]
        self.assertIs(judge.select_endpoint(candidates, creator="DeepSeek", parameters={"structured_outputs"}, max_tokens=64000), eligible)
        with self.assertRaisesRegex(ValueError, "No endpoint"):
            judge.select_endpoint(candidates[:3], creator="DeepSeek", parameters={"structured_outputs"}, max_tokens=64000)


class ModelExecution(unittest.TestCase):
    def test_jev_artifacts_contain_actual_input_not_unused_generation_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = judge.Trace(tmp)
            definition = judge.task([fixtures.PAIR], 'jev', 1, 1, trace, fixtures.VOCAB, judge.CRITERION)
            request = json.loads(definition.dataset[0].input)
            self.assertEqual(request, judge.jev_request(fixtures.PAIR, 'AB', False, fixtures.VOCAB, judge.CRITERION))
            self.assertIsNone(definition.config.extra_body)
            self.assertIsNone(definition.metadata['provider'])
            options = {'provider': {'only': ['fireworks'], 'allow_fallbacks': False, 'require_parameters': True}}
            generated = judge.task([fixtures.PAIR], 'gen', 1, 1, trace, fixtures.VOCAB, judge.CRITERION,
                                   options=options, model=get_model('mockllm/model'))
            self.assertEqual(generated.metadata['provider'], options['provider'])

    def test_corrupt_startup_cannot_leave_old_success_or_aggregate_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixtures.InspectPipeline().options(tmp)
            args.output.mkdir()
            (args.output / 'status.json').write_text('{"status":"success"}')
            (args.output / 'results.jsonl').write_text('{"old":true}\n')
            (args.output / 'events.jsonl').write_text('not valid json\n')
            with self.assertRaises(json.JSONDecodeError):
                judge.run(args)
            self.assertEqual(json.loads((args.output / 'status.json').read_text())['status'], 'failed')
            self.assertFalse((args.output / 'results.jsonl').exists())
            self.assertEqual(len(list(args.output.glob('results.*.previous.jsonl'))), 1)
            self.assertEqual((args.output / 'events.jsonl').read_text(), 'not valid json\n')

    def test_paid_compare_requires_explicit_pairs_and_empty_input_is_clear(self):
        environment = os.environ.copy()
        environment.pop('OPENROUTER_API_KEY', None)
        process = subprocess.run([sys.executable, '-m', 'llm_judge', 'compare'], capture_output=True, text=True, env=environment)
        self.assertEqual(process.returncode, 2)
        self.assertIn('--pairs', process.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No pairs'):
                judge.compare([], output=tmp)
            with self.assertRaisesRegex(ValueError, 'generative draws only'):
                judge.compare([fixtures.PAIR], passes=5, output=tmp)

    def test_real_adapter_uses_selected_endpoint_and_replays_without_catalog_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline", "INSPECT_CACHE_DIR": tmp + "/cache"}):
            args = fixtures.InspectPipeline().options(tmp)
            args.mode, args.passes, args.parallel = "gen", 1, 1
            chosen = endpoint("DeepInfra", "fp8", tag="deepinfra/fp8", supported_parameters=[
                "structured_outputs", "response_format", "max_tokens", "temperature"])
            response = httpx.Response(200, request=httpx.Request("GET", "https://openrouter.ai/catalog"),
                                      json={"data": {"endpoints": [endpoint("Fireworks", "unknown"), chosen]}})
            calls = []
            async def completion(api, request, config):
                calls.append(request)
                prompt = request["messages"][0]["content"]
                return ChatCompletion.model_validate({"id": "fixture", "created": 0, "object": "chat.completion",
                    "model": judge.MODEL_ID, "provider": "DeepInfra",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(fixtures.rating(prompt))}}]})
            with patch.object(judge.httpx, "get", return_value=response) as catalog, patch.object(OpenRouterAPI, "_generate_completion", new=completion):
                judge.run(args)
                self.assertEqual(len(calls), 4)
                self.assertEqual(catalog.call_count, 1)
                judge.run(args)
                self.assertEqual(len(calls), 4)
                self.assertEqual(catalog.call_count, 1)
            for request in calls:
                self.assertEqual(request["max_tokens"], 64000)
                self.assertNotIn("reasoning", request["extra_body"])
                self.assertNotIn("min_p", request["extra_body"])
                self.assertEqual(request["extra_body"]["provider"]["only"], ["deepinfra/fp8"])
            self.assertEqual(json.loads((args.output / "endpoint.json").read_text())["endpoint"], chosen)
            self.assertIn('"followup_reasoning": true', (args.output / 'run.md').read_text())

    def test_rejected_quote_cache_is_evicted_through_real_openrouter_adapter(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline", "INSPECT_CACHE_DIR": tmp + "/cache"}):
            args = fixtures.InspectPipeline().options(tmp)
            args.mode, args.passes, args.parallel = "gen", 1, 1
            chosen = endpoint("DeepInfra", "fp8", tag="deepinfra/fp8")
            catalog = httpx.Response(200, request=httpx.Request("GET", "https://openrouter.ai/catalog"), json={"data": {"endpoints": [chosen]}})
            invalid, calls = True, []
            async def completion(api, request, config):
                calls.append(request)
                result = fixtures.rating(request['messages'][0]['content'])
                if invalid:
                    result['response_A']['quotes'] = ['invented span']
                return ChatCompletion.model_validate({'id': 'fixture', 'created': 0, 'object': 'chat.completion',
                    'model': judge.MODEL_ID, 'provider': 'DeepInfra',
                    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(result)}}]})
            with patch.object(judge.httpx, 'get', return_value=catalog), patch.object(OpenRouterAPI, '_generate_completion', new=completion):
                with self.assertRaises(ValueError):
                    judge.run(args)
                failed_calls = len(calls)
                invalid = False
                result = judge.run(args)
                self.assertEqual(result[0]['gen']['B_minus_A_preference'], -1)
                self.assertEqual(len(calls) - failed_calls, 4)
                successful_calls = len(calls)
                judge.run(args)
                self.assertEqual(len(calls), successful_calls)
            rejected = [r for r in judge.records(args.output/'events.jsonl') if r['kind']=='invalid_generation_cache']
            self.assertEqual(sum(r['removed'] for r in rejected), 2)

    def test_always_reasoning_followup_keeps_full_allowance_and_returned_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"INSPECT_CACHE_DIR": tmp + "/cache"}):
            args = fixtures.InspectPipeline().options(tmp)
            args.mode, args.passes, args.parallel = "gen", 1, 1
            args.max_tokens, args.followup_reasoning = 96000, True
            calls = []
            async def mock(messages, tools, tool_choice, config):
                calls.append((messages, config))
                if len(messages) == 1:
                    return ModelOutput(model="fixture/gen", choices=[ChatCompletionChoice(
                        message=ChatMessageAssistant(content=[ContentReasoning(reasoning="Saved incomplete reasoning.")]), stop_reason="max_tokens")])
                self.assertNotIn("reasoning", config.extra_body)
                self.assertEqual(config.max_tokens, 96000)
                self.assertTrue(any(isinstance(p, ContentReasoning) and p.reasoning == "Saved incomplete reasoning." for p in messages[1].content))
                return ModelOutput.from_content("fixture/gen", json.dumps(fixtures.rating(messages[0].text)))
            judge.run(args, model=get_model("mockllm/model", custom_outputs=mock))
            self.assertEqual(len(calls), 8)
            self.assertIn("Saved incomplete reasoning.", (args.output / "run.md").read_text())


if __name__ == "__main__":
    unittest.main()

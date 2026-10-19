"""Maintain runtime safeguards through production Inspect and HTTP paths. -- PI/OpenAI"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ModelOutput, get_model

from llm_judge import trace as runtime
from llm_judge import pairs as evaluation
from importlib.resources import files
VOCAB = json.loads(files('llm_judge').joinpath('vocabulary.json').read_text())
PAIR = {'id': 'capital', 'context': 'Name the capital of France.', 'response_a': 'Paris.', 'response_b': 'Lyon.'}


def body(request):
    good_a = request['state']['answer_A'] == 'Paris.'
    answers = {}
    for name, q in request['questions'].items():
        if q['type'] == 'noul':
            answers[name] = {'type': 'noul', 'noul': .1}
        else:
            chosen = ('A' if good_a else 'B') if name == 'preference' else ('0' if good_a else '4') if name == 'change' else 'none'
            labels = list(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]
            answers[name] = {'type': q['type'], 'probabilities': {k: float(k == chosen) for k in labels},
                             **({'choice': chosen} if q['type'] == 'choice' else {'score': float(chosen)})}
    return {'model': 'fixture/jev', 'answers': answers, 'usage': {'cost': .001}}


def rating(prompt):
    a = prompt.split('Response A:\n', 1)[1].split('\n\nResponse B:', 1)[0]
    b = prompt.split('Response B:\n', 1)[1]
    def response(text):
        return {'quotes': [text], 'quality': 3 if text == 'Paris.' else 0,
                'refusal': False, 'incoherence': False, 'repetition': False, 'off_topic': False}
    return {'explanation': 'Paris answers the capital question correctly.', 'response_A': response(a), 'response_B': response(b),
            'preference': 'A' if a == 'Paris.' else 'B', 'judge_note': 'No apparent setup issue.'}


class DurableDecisions(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_resume_rubric_change_and_vent_copy(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'offline'}):
            trace = runtime.Trace(Path(tmp))
            count = 0
            async def reply(request):
                nonlocal count
                count += 1
                await asyncio.sleep(0)
                return httpx.Response(200, json=body(json.loads(request.content)))
            transport = httpx.MockTransport(reply)
            request = evaluation.jev_request(PAIR, 'AB', False, VOCAB, 'First criterion')
            copied = request['questions']['vent']['instructions']['questions_to_review']
            self.assertEqual(copied, {k:v for k,v in request['questions'].items() if k != 'vent'})
            self.assertEqual(copied['preference']['instructions'], 'First criterion')
            decide = runtime.Decisions(trace, evaluation.validate, transport=transport)
            await asyncio.gather(decide(request), decide(request))
            self.assertEqual(count, 1)
            resumed = runtime.Decisions(runtime.Trace(Path(tmp)), evaluation.validate, transport=transport)
            await resumed(request)
            self.assertEqual(count, 1)
            changed = evaluation.jev_request(PAIR, 'AB', False, VOCAB, 'Changed criterion')
            await resumed(changed)
            self.assertEqual(count, 2)
            self.assertIn('decision_cache_hit', (Path(tmp)/'events.jsonl').read_text())

    async def test_retry_and_permanent_failures_retained(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'offline'}):
            trace = runtime.Trace(Path(tmp))
            request = evaluation.jev_request(PAIR, 'AB', False, VOCAB, 'criterion')
            calls = []
            async def no_wait(delay):
                calls.append(delay)
            n = 0
            def reply(sent):
                nonlocal n
                n += 1
                return httpx.Response(429, text='retry body') if n == 1 else httpx.Response(200, json=body(request))
            await runtime.Decisions(trace, evaluation.validate, transport=httpx.MockTransport(reply), sleep=no_wait)(request)
            self.assertEqual(n, 2)
            self.assertEqual(len(calls), 1)
            self.assertIn('retry body', (Path(tmp)/'events.jsonl').read_text())
            changed = request | {'state': request['state'] | {'task': 'changed'}}
            decide = runtime.Decisions(trace, evaluation.validate, transport=httpx.MockTransport(lambda r:httpx.Response(400, text='permanent body')), sleep=no_wait)
            with self.assertRaises(httpx.HTTPStatusError):
                await decide(changed)
            self.assertIn('permanent body', (Path(tmp)/'events.jsonl').read_text())
            self.assertEqual(len(runtime.records(Path(tmp)/'decisions_cache.jsonl')), 1)

    async def test_torn_tail_preserved_bad_full_record_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.jsonl'
            path.write_bytes(b'{"ok":1}\n{"incomplete":')
            self.assertEqual(runtime.records(path), [{'ok':1}])
            self.assertEqual(path.with_suffix('.jsonl.partial').read_bytes(), b'{"incomplete":')
            self.assertEqual(path.read_bytes(), b'{"ok":1}\n')
            path.write_text('not json\n')
            with self.assertRaises(json.JSONDecodeError):
                runtime.records(path)


    async def test_parallel_bound_invalid_vectors_and_served_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            trace = runtime.Trace(Path(tmp))
            active, peak = 0, 0
            async def reply(sent):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.01)
                active -= 1
                return httpx.Response(200, json=body(json.loads(sent.content)))
            decide = runtime.Decisions(trace, evaluation.validate, parallel=2, transport=httpx.MockTransport(reply))
            requests = [evaluation.jev_request(PAIR | {'context':f'Question {i}'}, 'AB', False, VOCAB, 'criterion') for i in range(6)]
            await asyncio.gather(*(decide(q) for q in requests))
            self.assertEqual(peak, 2)
            first = requests[0]
            invalid = body(first)
            invalid['answers']['change']['score'] = 4
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                evaluation.validate(invalid, first)
            invalid = body(first)
            invalid['answers']['vent']['probabilities']['none'] = .97
            with self.assertRaisesRegex(ValueError, 'mass'):
                evaluation.validate(invalid, first)
            invalid['answers']['vent']['probabilities']['none'] = .99
            evaluation.validate(invalid, first)
            with self.assertRaisesRegex(ValueError, 'mixed served'):
                runtime.Trace(Path(tmp)).model('jev', 'changed/served-model')


class InspectPipeline(unittest.TestCase):
    def options(self, tmp, **changes):
        path = Path(tmp)/'pairs.jsonl'
        path.write_text(json.dumps(PAIR)+'\n')
        return argparse.Namespace(pairs=path, mode='both', passes=2, parallel=2, output=Path(tmp)/'run',
                                  criterion=evaluation.gen.CRITERION, dev=False, dev_budget=.05, **changes)

    def test_full_pipeline_epochs_cache_replay_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline', 'INSPECT_CACHE_DIR':tmp+'/cache'}):
            generated, typed = [], []
            async def mock(messages, tools, tool_choice, config):
                generated.append((messages, config))
                return ModelOutput.from_content('fixture/gen', json.dumps(rating(messages[0].text)))
            def reply(sent):
                typed.append(sent)
                return httpx.Response(200, json=body(json.loads(sent.content)))
            args = self.options(tmp)
            model = get_model('mockllm/model', custom_outputs=mock)
            result = evaluation.run(args, transport=httpx.MockTransport(reply), model=model)
            self.assertEqual(len(generated), 8)
            self.assertEqual(len(typed), 4)
            self.assertEqual(result[0]['gen']['B_minus_A_preference'], -1)
            self.assertEqual(result[0]['jev']['B_minus_A_preference'], -1)
            self.assertEqual(result[0]['jev']['response_order_discrepancy'], 0)
            evaluation.run(args, transport=httpx.MockTransport(reply), model=model)
            self.assertEqual(len(generated), 8, 'gen replay must make zero calls')
            self.assertEqual(len(typed), 4, 'Jev replay must make zero calls')
            samples = runtime.records(args.output/'samples.jsonl')
            self.assertEqual(len(samples), 16)
            self.assertEqual({s['epoch'] for s in samples}, {1,2})
            text = (args.output/'run.md').read_text()
            self.assertIn('First pair: complete sample', text)
            self.assertIn('response_schema', text)
            self.assertIn('judge_note', text)
            self.assertTrue(list((args.output/'logs').glob('*.eval')))
            args.criterion = 'A changed criterion forces fresh measurement.'
            evaluation.run(args, transport=httpx.MockTransport(reply), model=model)
            self.assertEqual(len(generated), 16)
            self.assertEqual(len(typed), 8)

    def test_dev_budget_stop_does_not_discard_valid_scores(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            count = 0
            def reply(sent):
                nonlocal count
                count += 1
                request = json.loads(sent.content)
                if 'answer_A' in request['state']:
                    result = body(request)
                else:
                    answers = {}
                    for name, q in request['questions'].items():
                        labels = list(q['criteria'])
                        answers[name] = {'type':'choice', 'choice':labels[0], 'probabilities':{k:float(k == labels[0]) for k in labels}}
                    result = {'answers':answers, 'model':'fixture/jev', 'usage':{'cost':.051}}
                return httpx.Response(200, json=result)
            args = self.options(tmp)
            args.mode, args.passes, args.parallel, args.dev = 'jev', 1, 1, True
            result = evaluation.run(args, transport=httpx.MockTransport(reply))
            self.assertEqual(count, 5)
            self.assertEqual(result[0]['jev']['B_minus_A_preference'], -1)
            self.assertEqual(json.loads((args.output/'status.json').read_text())['status'], 'success')
            samples = runtime.records(args.output/'samples.jsonl')
            diagnostic = [s['metadata']['dev'] for s in samples if 'dev' in s['metadata']][0]
            self.assertEqual(diagnostic['status'], 'budget_exhausted')
            self.assertEqual(diagnostic['cost'], .051)
            self.assertEqual(diagnostic['calls'], 1)
            self.assertIn('partial_state', diagnostic)
            self.assertIn('First pair: raw event', (args.output/'run.md').read_text())

    def test_reasoning_recovery_and_invalid_quote_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'INSPECT_CACHE_DIR':tmp+'/cache'}):
            calls = []
            async def mock(messages, tools, tool_choice, config):
                calls.append((messages.copy(), config))
                if len(messages) == 1:
                    return ModelOutput(model='fixture/gen', choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=[ContentReasoning(reasoning='Compared the capital answers.')]), stop_reason='max_tokens')])
                answer = rating(messages[0].text)
                if len(messages) == 3:
                    answer['response_A']['quotes'] = ['invented quote']
                return ModelOutput.from_content('fixture/gen', json.dumps(answer))
            args = self.options(tmp)
            args.mode, args.passes, args.parallel = 'gen', 1, 1
            result = evaluation.run(args, model=get_model('mockllm/model', custom_outputs=mock))
            self.assertEqual(len(calls), 12)
            for messages, config in calls:
                if len(messages) > 1:
                    self.assertEqual(config.extra_body['reasoning'], {'enabled':False})
                    self.assertTrue(any(isinstance(part, ContentReasoning) and part.reasoning == 'Compared the capital answers.' for part in messages[1].content))
            samples = runtime.records(args.output/'samples.jsonl')
            self.assertTrue(all(s['metadata']['follow_up_used'] for s in samples))
            trace = (args.output/'run.md').read_text()
            self.assertIn('invented quote', trace)
            self.assertIn('Validation failed', trace)
            self.assertIn('Compared the capital answers.', trace)
            self.assertEqual(result[0]['gen']['B_minus_A_quality'], -3)

    def test_provider_routing_raw_trace_and_reasoning_replay(self):
        from inspect_ai.model import ChatMessageUser, GenerateConfig
        from inspect_ai.model._openai import OpenAIResponseError
        from inspect_ai.model._providers.openrouter import openrouter_reasoning_details_to_reasoning
        from openai import BadRequestError
        from openai.types.chat import ChatCompletion
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            api = get_model(evaluation.gen.MODEL).api
            request = httpx.Request('POST', 'https://openrouter.ai/api/v1/chat/completions')
            def bad(code):
                body = {'error':{'code':code, 'message':'fixture'}}
                return BadRequestError('fixture', response=httpx.Response(400, request=request, json=body), body=body)
            with self.assertRaises(OpenAIResponseError):
                api.handle_bad_request(bad(503))
            self.assertIsInstance(api.handle_bad_request(bad(400)), BadRequestError)
            sent = []
            async def stub(request, config):
                sent.append(request)
                return ChatCompletion.model_validate({'id':'fixture', 'created':0, 'object':'chat.completion',
                    'model':evaluation.gen.MODEL_ID, 'provider':'DeepInfra',
                    'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{}'}}]})
            api._generate_completion = stub
            details = [{'type':'reasoning.text','text':'Preserve this reasoning.','format':'unknown','index':0}]
            trace = runtime.Trace(Path(tmp))
            evaluation.gen.TRACE = trace
            try:
                output, raw = asyncio.run(api.generate(input=[ChatMessageUser(content='Compare.'),
                    ChatMessageAssistant(content=[openrouter_reasoning_details_to_reasoning(details)]),
                    ChatMessageUser(content='Return final JSON.')], tools=[], tool_choice=None,
                    config=GenerateConfig(extra_body=evaluation.gen.OPTIONS, response_schema=evaluation.gen.schema(False))))
            finally:
                evaluation.gen.TRACE = None
            self.assertEqual(sent[0]['messages'][1]['reasoning_details'], details)
            self.assertEqual(sent[0]['extra_body']['provider'], evaluation.gen.PROVIDER)
            self.assertTrue(sent[0]['response_format']['json_schema']['strict'])
            self.assertEqual(output.metadata['provider'], 'DeepInfra')
            events = runtime.records(Path(tmp)/'events.jsonl')
            self.assertEqual([e['kind'] for e in events], ['generative_request','generative_response'])
            self.assertEqual(events[-1]['raw']['response']['provider'], 'DeepInfra')
            self.assertNotIn('Authorization', (Path(tmp)/'events.jsonl').read_text())

    def test_unresolved_output_fails_with_complete_failed_sample(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'INSPECT_CACHE_DIR':tmp+'/cache'}):
            async def mock(messages, tools, tool_choice, config):
                return ModelOutput.from_content('fixture/gen', '{"invalid":true}')
            args = self.options(tmp)
            args.mode, args.passes, args.parallel = 'gen', 1, 1
            (args.output).mkdir(parents=True)
            (args.output/'results.jsonl').write_text('{"stale":true}\n')
            with self.assertRaises(ValueError):
                evaluation.run(args, model=get_model('mockllm/model', custom_outputs=mock))
            self.assertFalse((args.output/'results.jsonl').exists())
            self.assertEqual(json.loads((args.output/'status.json').read_text())['status'], 'failed')
            self.assertEqual(len(list(args.output.glob('results.*.previous.jsonl'))), 1)
            self.assertIn('Failed run', (args.output/'run.md').read_text())
            samples = runtime.records(args.output/'samples.jsonl')
            self.assertTrue(any(s['error'] for s in samples))
            self.assertIn('invalid', (args.output/'run.md').read_text())


if __name__ == '__main__':
    unittest.main()

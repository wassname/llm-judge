"""First-sample feedback and complete-trace review are normal defaults. -- PI/OpenAI"""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from llm_judge import judge
import test_questions as label_fixtures
import test_runtime as pair_fixtures


def budget_stop(request):
    answers = {}
    for name, question in request['questions'].items():
        labels = list(question['criteria'])
        answers[name] = {'type': 'choice', 'choice': labels[0], 'probabilities': {k: float(k == labels[0]) for k in labels}}
    return {'answers': answers, 'model': 'fixture/jev', 'usage': {'cost': .051}}


class FirstPair(unittest.TestCase):
    def test_default_cli_interviews_first_pair_only_and_keeps_valid_scores(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'offline'}):
            source, output = Path(tmp) / 'pairs.jsonl', Path(tmp) / 'run'
            other = pair_fixtures.PAIR | {'id': 'other', 'context': 'Second sample; same answer alternatives.'}
            source.write_text('\n'.join(json.dumps(p) for p in (pair_fixtures.PAIR, other)))
            requests, feedback = [], []
            def reply(sent):
                request = json.loads(sent.content)
                requests.append(request)
                if 'answer_A' in request['state']:
                    body = pair_fixtures.body(request)
                else:
                    feedback.append(request)
                    body = budget_stop(request)
                return httpx.Response(200, json=body)
            original = judge.run
            def execute(args):
                self.assertTrue(args.dev)
                return original(args, transport=httpx.MockTransport(reply))
            argv = ['llm-judge', 'compare', '--pairs', str(source), '--parallel', '1', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(judge, 'run', side_effect=execute):
                judge.main()
            self.assertEqual(len(requests), 9)
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback[0]['state']['judged_setup']['state']['request'], pair_fixtures.PAIR['context'])
            rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
            self.assertEqual([r['jev']['B_minus_A_preference'] for r in rows], [-1, -1])
            samples = judge.records(output / 'samples.jsonl')
            interviews = [s['metadata']['dev'] for s in samples if 'dev' in s['metadata']]
            self.assertEqual(len(interviews), 1)
            self.assertEqual(interviews[0]['status'], 'budget_exhausted')
            self.assertTrue(interviews[0]['unscored'])
            self.assertEqual((output / 'run.md').read_text().rstrip().splitlines()[-1], judge.READ_LOG_INSTRUCTION)

    def test_handled_failure_also_ends_with_the_full_log_review_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                with judge.Trace(tmp):
                    raise ValueError('fixture failure')
            self.assertEqual(json.loads((Path(tmp) / 'status.json').read_text())['status'], 'failed')
            self.assertEqual((Path(tmp) / 'run.md').read_text().rstrip().splitlines()[-1], judge.READ_LOG_INSTRUCTION)


class FirstState(unittest.IsolatedAsyncioTestCase):
    async def test_plain_callback_requires_explicit_optout_before_any_paid_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            async def decide(request):
                calls.append(request)
                return label_fixtures.reply_body(request)
            questions = {'ok': {'type': 'noul', 'instructions': 'Is this sufficient?'}}
            with judge.Trace(tmp) as trace:
                with self.assertRaisesRegex(TypeError, 'charged'):
                    await judge.label({'data': 'fixture'}, questions, trace=trace, decide=decide)
                with self.assertRaisesRegex(ValueError, 'finite and positive'):
                    await judge.label({'data': 'fixture'}, questions, trace=trace, decide=decide, feedback_budget=0)
                self.assertEqual(calls, [])
                self.assertEqual(trace.label_count, 0)
                result = await judge.label({'data': 'fixture'}, questions, trace=trace, decide=decide, feedback=False)
                self.assertEqual(result['means']['ok']['noul'], .9)
                self.assertEqual(len(calls), 2)

    async def test_default_label_interviews_only_first_state_and_logs_it_unclipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, feedback = [], []
            state = {'data': 'BEGIN_FULL_SAMPLE ' + 'sample material ' * 1000 + ' END_FULL_SAMPLE'}
            questions = {'ok': {'type': 'noul', 'instructions': 'Is the supplied material sufficient?'}}
            def reply(sent):
                request = json.loads(sent.content)
                calls.append(request)
                if 'ok' in request['questions']:
                    body = label_fixtures.reply_body(request)
                else:
                    feedback.append(request)
                    body = budget_stop(request)
                return httpx.Response(200, json=body)
            with judge.Trace(tmp) as trace:
                decide = judge.Decisions(trace, judge.validate, transport=httpx.MockTransport(reply), api_key='offline')
                first = await judge.label(state, questions, trace=trace, decide=decide)
                replay = await judge.label(state, questions, trace=trace, decide=decide)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(feedback), 1)
            self.assertEqual(first['feedback']['status'], 'budget_exhausted')
            self.assertEqual(first['means'], replay['means'])
            self.assertNotIn('feedback', first['means'])
            log = (Path(tmp) / 'run.md').read_text()
            self.assertIn(state['data'], log)
            self.assertIn('partial_state', log)
            self.assertEqual(log.rstrip().splitlines()[-1], judge.READ_LOG_INSTRUCTION)


if __name__ == '__main__':
    unittest.main()

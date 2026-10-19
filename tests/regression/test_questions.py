"""Exercise composable labeling and optional controls through the real transport. -- PI/OpenAI"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from llm_judge import Trace, label
from llm_judge.diagnostics import control_requests, controls
from llm_judge.jev import validate
from llm_judge.trace import Decisions, records


def reply_body(request, cost=.001):
    answers = {}
    for name, question in request['questions'].items():
        if question['type'] == 'noul':
            answers[name] = {'type':'noul','noul':.9}
        else:
            options = list(question['criteria']) if question['type'] == 'choice' else [str(i) for i in range(len(question['criteria']))]
            winner = 'none' if name == 'vent' else 'tie' if 'tie' in options else str(len(options)//2) if question['type'] == 'score' else options[0]
            answers[name] = {'type':question['type'], 'probabilities':{k:float(k == winner) for k in options},
                             **({'choice':winner} if question['type'] == 'choice' else {'score':float(winner)})}
    return {'model':'fixture/jev', 'answers':answers, 'usage':{'cost':cost}}


class Questions(unittest.IsolatedAsyncioTestCase):
    async def test_nested_inputs_full_vent_and_ordered_score_levels(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            sent = []
            def reply(request):
                data = json.loads(request.content)
                sent.append(data)
                return httpx.Response(200,json=reply_body(data))
            state = {'task':'Evaluate provided material', 'story':{'dialogue':['A','B']}}
            questions = {'tree':{'type':'choice', 'instructions':{'question':'Which category?', 'context':{'origin':'supplied'}},
                                 'criteria':{'A':{'children':['x','y']},'B':{'children':['z']}}},
                         'level':{'type':'score','instructions':'How much?', 'criteria':['none','some','much']},
                         'flag':{'type':'noul','instructions':'Is the evidence sufficient?'}}
            untouched = json.loads(json.dumps(questions))
            with Trace(tmp) as trace:
                decide = Decisions(trace, validate, transport=httpx.MockTransport(reply))
                result = await label(state, questions, trace=trace, decide=decide)
                replay = await label(state, questions, trace=trace, decide=decide)
            self.assertEqual(len(sent),2)
            self.assertEqual(questions, untouched)
            self.assertEqual(sent[0]['state'], state)
            self.assertEqual(list(sent[0]['questions']['tree']['criteria']), ['A','B'])
            self.assertEqual(list(sent[1]['questions']['tree']['criteria']), ['B','A'])
            for request in sent:
                q = request['questions']
                self.assertEqual(q['vent']['instructions']['questions_to_review'],{k:v for k,v in q.items() if k != 'vent'})
                self.assertEqual(len(q['vent']['criteria']),219)
                self.assertEqual(q['level']['criteria'], ['none','some','much'])
            self.assertEqual(result['means']['level']['score'], 1)
            self.assertEqual(result['means']['flag']['noul'], .9)
            self.assertNotIn('vent',result['means'])
            self.assertEqual(len(result['unscored_vent']),2)
            self.assertEqual(replay['means'],result['means'])
            self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['status'],'success')

    async def test_controls_are_fresh_and_isolate_rubric_visibility(self):
        requests = list(control_requests())
        self.assertEqual(len(requests),6)
        for i in range(0,6,2):
            hidden, visible = requests[i][2], requests[i+1][2]
            self.assertEqual(hidden['state'],visible['state'])
            self.assertEqual({k:v for k,v in hidden['questions'].items() if k != 'vent'},
                             {k:v for k,v in visible['questions'].items() if k != 'vent'})
            self.assertNotIn('questions_to_review',hidden['questions']['vent']['instructions'])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            calls = []
            def reply(request):
                calls.append(request)
                return httpx.Response(200,json=reply_body(json.loads(request.content)))
            with Trace(tmp) as trace:
                decide = Decisions(trace, validate, transport=httpx.MockTransport(reply))
                result = await controls(trace=trace,budget=1,repeats=2,decide=decide)
            self.assertEqual(result['calls'],12)
            self.assertEqual(len(calls),12,'identical repeats must not come from cache or dedup')
            self.assertEqual(calls[0].content,calls[1].content)
            self.assertTrue(result['unscored'])

    async def test_actual_charge_excludes_cache_and_duplicate_waiters(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            calls = []
            async def reply(sent):
                calls.append(sent)
                await asyncio.sleep(0)
                return httpx.Response(200,json=reply_body(json.loads(sent.content),cost=.02))
            with Trace(tmp) as trace:
                decide = Decisions(trace,validate,transport=httpx.MockTransport(reply))
                request = next(control_requests())[2]
                results = await asyncio.gather(decide.charged(request),decide.charged(request))
                self.assertEqual(sum(charge for body,charge in results),.02)
                body, charge = await decide.charged(request)
                self.assertEqual(charge,0)
                self.assertEqual(body['usage']['cost'],.02,'raw provider usage must remain untouched')
                self.assertEqual(len(calls),1)

    async def test_control_budget_and_context_failure_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY':'offline'}):
            def reply(request):
                return httpx.Response(200,json=reply_body(json.loads(request.content),cost=.02))
            with Trace(tmp) as trace:
                decide = Decisions(trace, validate, transport=httpx.MockTransport(reply))
                result = await controls(trace=trace,budget=.01,decide=decide)
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertEqual(len(result['rows']),1)
            self.assertEqual(result['cost'],.02)
            try:
                with Trace(tmp):
                    raise ValueError('fixture failure')
            except ValueError:
                pass
            self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['status'],'failed')
            self.assertIn('fixture failure',(Path(tmp)/'events.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()

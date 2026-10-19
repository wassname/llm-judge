"""Regression tests for the opt-in composer through production code. -- PI/OpenAI"""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from llm_judge import jev as judge_jev
from llm_judge.feedback import compose_feedback


SENTENCE = 'the reference is missing'
WORDS = SENTENCE.split() + ['wrong', 'other']


def reply(request, *, first_wrong=False):
    state = request['state']
    draft = state['candidate'] if 'candidate' in state else state['draft']
    answers = {}
    for name, q in request['questions'].items():
        if q['type'] == 'choice':
            options = list(q['criteria'])
            expected = SENTENCE.split()[len(draft.split())] if SENTENCE.startswith(draft) and len(draft.split()) < 4 else 'other'
            next_word = 'wrong' if first_wrong and not state['recent_moves'] else expected
            label = next_word if next_word in options else options[0]
            answers[name] = {'type': 'choice', 'choice': label, 'confidence': 1,
                            'probabilities': {w: float(w == label) for w in options}}
        elif q['type'] == 'score':
            answers[name] = {'type': 'score', 'score': 3, 'confidence': 1,
                             'probabilities': {str(i): float(i == 3) for i in range(len(q['criteria']))}}
        else:
            if name == 'finish':
                p = float(draft == SENTENCE)
            elif name == 'undo':
                p = float(draft == 'wrong')
            else:
                text = q['instructions'].split('makes the draft ', 1)[1].split('. Would', 1)[0]
                candidate = ast.literal_eval(text)
                valid = SENTENCE.startswith(candidate) and len(candidate) > len(draft)
                p = 0.9 if valid else 0.01
                if first_wrong and not state['recent_moves'] and candidate == 'wrong':
                    p = 0.99
            answers[name] = {'type': 'noul', 'noul': p}
    return {'model': 'typesafe/jev-1.13-20260917', 'answers': answers, 'usage': {'cost': 0.0001}}


class TalkTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_sentence_uses_separate_phrase_checks(self):
        requests = []
        async def decide(request):
            requests.append(request)
            body = reply(request)
            judge_jev.validate(body, request)
            return body
        result = await compose_feedback({'judged_setup': {}, 'returned_answers': {}}, WORDS, decide, model=judge_jev.JEV)
        self.assertEqual(result['text'], SENTENCE)
        self.assertEqual(result['status'], 'model_stopped')
        self.assertTrue(result['unscored'])
        self.assertEqual(len(requests), 10)
        self.assertTrue(all(q['type'] == 'noul' for q in requests[1]['questions'].values()))

    async def test_backtracking_rejects_the_previous_bad_prefix(self):
        seen = []
        async def decide(request):
            seen.append(request)
            return reply(request, first_wrong=True)
        result = await compose_feedback({'judged_setup': {}, 'returned_answers': {}}, WORDS, decide, model=judge_jev.JEV)
        self.assertEqual(result['status'], 'model_stopped')
        self.assertEqual(result['text'], SENTENCE)
        self.assertIn('undo', [r['move'] for r in result['rounds']])
        after_undo = next(i for i, r in enumerate(seen) if any('Removed' in move for move in r['state']['recent_moves']))
        for request in seen[after_undo:]:
            for name, question in request['questions'].items():
                if name.startswith('append_') and not request['state']['draft']:
                    self.assertNotIn("makes the draft 'wrong", question['instructions'])

    async def test_round_limit_is_not_a_completed_explanation(self):
        async def decide(request):
            return reply(request)
        result = await compose_feedback({}, WORDS, decide, model=judge_jev.JEV, max_rounds=1)
        self.assertEqual(result['status'], 'round_limit')
        self.assertEqual(result['text'], 'the')
        self.assertIn('Unfinished', result['note'])



if __name__ == "__main__":
    unittest.main()

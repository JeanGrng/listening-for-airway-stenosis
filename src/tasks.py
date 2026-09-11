"""The 16 recording task groups, their display names and their families.

`b2ai_canonical.TASK_GROUPS` is the authority on which groups exist (it derives
them from the raw B2AI task names). This module only adds presentation: a human
readable label and the coarse family used in the task-importance figure.
"""

from __future__ import annotations

# group name -> label used in figures and tables
PRETTY = {
    'picture-description':               'Picture description',
    'story-recall':                      'Story recall',
    'rainbow-passage':                   'Rainbow passage',
    'diadochokinesis-pataka':            'DDK /pataka/',
    'diadochokinesis-ta':                'DDK /ta/',
    'diadochokinesis-ka':                'DDK /ka/',
    'diadochokinesis-pa':                'DDK /pa/',
    'diadochokinesis-buttercup':         'DDK buttercup',
    'prolonged-vowel':                   'Prolonged vowel',
    'maximum-phonation-time':            'Max phonation time',
    'loudness':                          'Loudness',
    'glides-low-to-high':                'Glide low->high',
    'glides-high-to-low':                'Glide high->low',
    'respiration-and-cough-breath':      'Breath',
    'respiration-and-cough-cough':       'Cough',
    'respiration-and-cough-fivebreaths': 'Five breaths',
}

# group name -> family
FAMILY = {
    'picture-description':               'Complex speech',
    'story-recall':                      'Complex speech',
    'rainbow-passage':                   'Complex speech',
    'diadochokinesis-pataka':            'Complex articulation',
    'diadochokinesis-ta':                'Simple DDK',
    'diadochokinesis-ka':                'Simple DDK',
    'diadochokinesis-pa':                'Simple DDK',
    'diadochokinesis-buttercup':         'Simple DDK',
    'prolonged-vowel':                   'Sustained acoustic',
    'maximum-phonation-time':            'Sustained acoustic',
    'loudness':                          'Sustained acoustic',
    'glides-low-to-high':                'Sustained acoustic',
    'glides-high-to-low':                'Sustained acoustic',
    'respiration-and-cough-breath':      'Respiration',
    'respiration-and-cough-cough':       'Respiration',
    'respiration-and-cough-fivebreaths': 'Respiration',
}

# one line per family, for the README and for readers unfamiliar with the protocol
FAMILY_DESCRIPTION = {
    'Complex speech':       'connected speech: reading a standard passage, describing a picture, retelling a story',
    'Complex articulation': 'the /pataka/ diadochokinetic sequence, alternating three places of articulation',
    'Simple DDK':           'repetition of a single syllable as fast as possible',
    'Sustained acoustic':   'sustained phonation: held vowels, pitch glides, loudness and maximum phonation time',
    'Respiration':          'breathing and coughing, without phonation',
}

FAMILY_ORDER = ['Complex speech', 'Complex articulation', 'Simple DDK',
                'Sustained acoustic', 'Respiration']

assert set(PRETTY) == set(FAMILY), "PRETTY and FAMILY must cover the same groups"
assert len(PRETTY) == 16, f"expected 16 task groups, got {len(PRETTY)}"

"""scripts/release_cards.py: the wording each published card needs beyond the shared template. No network call."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_cards', ROOT / 'scripts' / 'release_cards.py')
cards = importlib.util.module_from_spec(SPEC)
sys.modules['release_cards'] = cards
SPEC.loader.exec_module(cards)

VALUES = {'TARGET_LANGUAGE': 'Thai', 'TARGET_CONDITION': 'with the preamble', 'TARGET_SAFE_GLOBAL': '94.9',
          'TARGET_CONFIRMATION_SENTENCE': 'That gap did not pass our test, so every repair result here is exploratory.',
          'MODEL_REPO_ID': 'team/tiny-aya-earth-langsafe-steer-1p0', 'REPO_URL': 'https://github.com/team/langsafe',
          'MODEL_NAME': 'tiny-aya-earth-langsafe-x'}


def test_a_card_without_a_confirmed_target_gap_says_it_is_exploratory():
    text = ('the rate with the preamble was 89.2% for Earth against 94.9% for Global. This checkpoint is one attempt.\n'
            'These results apply to with the preamble, which has a confirmed gap for the target language. '
            'In a preamble condition without one, this checkpoint is exploratory.')
    out = cards.exploratory(text, VALUES)
    assert 'which has a confirmed gap' not in out
    assert 'Thai, the target language, has no confirmed gap in either preamble condition' in out
    assert 'for Global. That gap did not pass our test' in out
    assert cards.exploratory(out, VALUES) == out    # applying it twice changes nothing


def test_the_steering_card_describes_a_direction_not_weights():
    text = ('---\nbase_model: CohereLabs/tiny-aya-earth\nbase_model_relation: merge\nlibrary_name: transformers\n---\n'
            'A research checkpoint derived from Tiny Aya Earth by weight arithmetic, built to test per-language safety '
            'repair. Arm: **Steering**, knob strength = 1.0, precision FP16.\n'
            '## Use the model\nweights example\n## Reproducibility\n| Provenance | `lexsi_provenance.json` in this repository |\n')
    out = cards.steering(text, VALUES)
    assert 'base_model_relation' not in out and 'library_name' not in out
    assert 'A refusal direction for Tiny Aya Earth, added at run time' in out and 'No weights are changed.' in out
    assert '## Use the direction' in out and 'weights example' not in out
    assert 'snapshot_download("team/tiny-aya-earth-langsafe-steer-1p0")' in out
    assert '`meta.json` in this repository' in out


def test_the_gguf_card_points_at_its_full_precision_source():
    text = ('---\nbase_model: CohereLabs/tiny-aya-earth\nbase_model_relation: merge\nlibrary_name: transformers\n---\n'
            '# tiny-aya-earth-langsafe-x\nArm: **Circuit merge**, knob lambda = 0.5, precision FP16.\n'
            '## Use the model\nweights example\n## Reproducibility\n| Provenance | `lexsi_provenance.json` in this repository |\n')
    out = cards.gguf(text, VALUES, 'x-q4_k_m.gguf', 'team/full')
    assert 'base_model: team/full\nbase_model_relation: quantized\nlibrary_name: gguf' in out
    assert 'precision Q4_K_M (GGUF), quantised from FP16.' in out
    assert 'llama-server -m x-q4_k_m.gguf' in out and 'weights example' not in out
    assert '[team/full](https://huggingface.co/team/full)' in out


def test_the_missed_conditions_come_from_the_gates():
    report = {'thresholds': {'arm/default': {'harmful_gap_closed': {'value': 0.1, 'pass': False},
                                            'over_refusal_increase': {'pass': False}, 'capability_drop': {'pass': True}}}}
    line = cards.limitations(report, 'arm', {'TARGET_LANGUAGE': 'Thai', 'MAX_OVER_REFUSAL_RISE': '+2.1',
                                             'MAX_CAPABILITY_DROP': '+0.5'})
    assert 'closed 10% of the Thai gap' in line and '+2.1 points against a cap of 2' in line
    assert 'capability' not in line
    assert cards.limitations({'thresholds': {}}, 'arm', {}) == ''


def test_a_card_with_a_placeholder_left_is_not_written():
    with pytest.raises(SystemExit, match='MISSING'):
        cards.card('# {{MODEL_NAME}}\n{{MISSING}}\n', {'MODEL_NAME': 'x'}, {})

"""Synthetic partition checks; no benchmark inputs, evaluators or model calls."""
from copy import deepcopy

import pytest

from phiagent.harness.benchmark_partition import (
    build_partition,canonical_sha256,materialize_partition,validate_partition,
)


def records(n):
    return [{'id':f'synthetic-{i}','payload_sha256':canonical_sha256({'payload':i}),
             'public_task_sha256':canonical_sha256({'public':i})} for i in range(n)]


@pytest.mark.parametrize('n,counts',[(30,{'train':6,'validation':6,'test':18}),
                                    (160,{'train':32,'validation':32,'test':96})])
def test_exact_ratios_and_full_disjoint_population(n,counts):
    value=build_partition(records(n),benchmark='synthetic',seed=20260928)
    assert value['counts']==counts
    validate_partition(value)
    data=materialize_partition(records(n),value)
    assert not data['test'] and len(data['train'])==counts['train']
    assert len(data['development'])==counts['validation']
    assert len(materialize_partition(records(n),value,include_test=True)['test'])==counts['test']


def test_split_is_order_independent_and_outcome_independent():
    source=records(30);expected=build_partition(source,benchmark='synthetic',seed=1)
    changed=[{**r,'score':i,'hard':bool(i%2)} for i,r in enumerate(reversed(source))]
    assert build_partition(changed,benchmark='synthetic',seed=1)==expected


def test_overlap_missing_population_and_repeat_leakage_fail():
    good=build_partition(records(30),benchmark='synthetic',seed=1)
    bad=deepcopy(good);bad['splits']['test'][0]=bad['splits']['train'][0]
    with pytest.raises(ValueError,match='Overlapping'):validate_partition(bad)
    bad=deepcopy(good);bad['all_repetitions_follow_unit_split']=False
    with pytest.raises(ValueError,match='Repeated'):validate_partition(bad)
    with pytest.raises(ValueError,match='Source items'):
        materialize_partition(records(29),good)


def test_duplicate_public_tasks_must_be_grouped_not_leaked():
    source=records(30);source[1]['public_task_sha256']=source[0]['public_task_sha256']
    with pytest.raises(ValueError,match='grouped'):build_partition(source,benchmark='synthetic',seed=1)
    with pytest.raises(ValueError,match='divisible'):build_partition(records(31),benchmark='synthetic',seed=1)


@pytest.mark.parametrize('n,counts', [
    (89, {'train': 18, 'validation': 18, 'test': 53}),
    (14, {'train': 3, 'validation': 3, 'test': 8}),
    (30, {'train': 6, 'validation': 6, 'test': 18}),
])
def test_explicit_rounding_keeps_the_entire_independent_population(n, counts):
    value = build_partition(records(n), benchmark='synthetic', seed=20260928,
                            rounding='largest_remainder', independent_unit='image')
    assert value['schema'] == 'phiagent-20-20-60/v2'
    assert value['counts'] == counts
    assert value['independent_unit'] == 'image'
    assert sum(counts.values()) == n
    validate_partition(value)
    assert build_partition(list(reversed(records(n))), benchmark='synthetic', seed=20260928,
                           rounding='largest_remainder', independent_unit='image') == value


def test_rounding_policy_cannot_be_changed_after_freeze():
    good = build_partition(records(89), benchmark='synthetic', seed=1,
                           rounding='largest_remainder')
    bad = deepcopy(good)
    bad['rounding_tie_order'] = ['test', 'validation', 'train']
    with pytest.raises(ValueError, match='rounding'):
        validate_partition(bad)
    bad = deepcopy(good)
    bad['counts'] = {'train': 17, 'validation': 17, 'test': 55}
    with pytest.raises(ValueError, match='Partition sizes'):
        validate_partition(bad)
    with pytest.raises(ValueError, match='Unsupported'):
        build_partition(records(89), benchmark='synthetic', seed=1, rounding='chosen_after_scores')

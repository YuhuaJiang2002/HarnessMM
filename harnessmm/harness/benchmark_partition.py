"""Outcome-independent 20/20/60 partitions for independent benchmark units."""
from collections.abc import Mapping, Sequence
import hashlib
import json

SPLITS = ('train', 'validation', 'test')


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=False).encode()).hexdigest()


def _partition_counts(population: int, rounding: str):
    if type(population) is not int or population <= 0:
        raise ValueError('Invalid independent population')
    if rounding == 'exact':
        if population % 5:
            raise ValueError('Exact 20/20/60 requires a nonempty population divisible by five')
        return dict(zip(SPLITS, (population//5, population//5, 3*population//5)))
    if rounding != 'largest_remainder':
        raise ValueError('Unsupported outcome-independent rounding policy')
    if population < 5:
        raise ValueError('At least five independent units are required for three nonempty splits')
    numerators = (population, population, 3*population)
    counts = [value//5 for value in numerators]
    order = sorted(range(3), key=lambda index: (-(numerators[index] % 5), index))
    for index in order[:population-sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def build_partition(records: Sequence[Mapping], *, benchmark: str, seed: int,
                    rounding: str = 'exact', independent_unit: str | None = None):
    if not benchmark or isinstance(seed,bool) or not isinstance(seed,int):
        raise ValueError('A named benchmark and integer split seed are required')
    counts = _partition_counts(len(records), rounding)
    if independent_unit is not None and (
            not isinstance(independent_unit, str) or not independent_unit.strip()):
        raise ValueError('Independent unit must be a nonempty name')
    identifiers=[r['id'] for r in records]
    if any(not isinstance(i,str) or not i for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Independent benchmark unit IDs must be unique nonempty strings')
    public_hashes=[r['public_task_sha256'] for r in records]
    for row in records:
        for field in ('payload_sha256','public_task_sha256'):
            value=row[field]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Malformed record hash: '+field)
    if len(set(public_hashes))!=len(public_hashes):
        raise ValueError('Duplicate public tasks require explicit grouped partitioning before use')
    ranked=sorted(identifiers,key=lambda i:(hashlib.sha256(
        f'{benchmark}\0{seed}\0{i}'.encode()).hexdigest(),i))
    train_end = counts['train']
    validation_end = train_end + counts['validation']
    split_ids={'train':ranked[:train_end],'validation':ranked[train_end:validation_end],
               'test':ranked[validation_end:]}
    source=sorted(({key:r[key] for key in ('id','payload_sha256','public_task_sha256')}
                   for r in records),key=lambda r:r['id'])
    value={'schema':'phiagent-20-20-60/v1','benchmark':benchmark,'split_seed':seed,
        'independent_unit':independent_unit or ('problem' if benchmark=='aime2025' else 'scenario'),
        'population':len(records),'ratios':{'train':0.2,'validation':0.2,'test':0.6},
        'counts':{k:len(v) for k,v in split_ids.items()},'splits':split_ids,
        'records':source,'population_sha256':canonical_sha256(source),
        'selection_inputs':['benchmark','split_seed','independent_unit_id'],
        'score_or_outcome_used':False,'all_repetitions_follow_unit_split':True,
        'validation_use':'Selection of frozen training-derived candidates only; no validation trace feedback to optimizer',
        'test_use':'Only after candidate and all baseline policies are frozen',
        'prior_exposure':'This public population appeared in earlier archived runs; this is a newly enforced partition, not previously unseen data.'}
    if rounding != 'exact':
        value.update(schema='phiagent-20-20-60/v2', rounding_policy=rounding,
                     rounding_tie_order=list(SPLITS))
    validate_partition(value)
    return value


def validate_partition(manifest):
    if manifest['schema'] not in ('phiagent-20-20-60/v1','phiagent-20-20-60/v2') or manifest['score_or_outcome_used']:
        raise ValueError('Unsupported or outcome-selected partition')
    population=manifest['population']
    rounding = 'exact'
    if manifest['schema'] == 'phiagent-20-20-60/v2':
        rounding = manifest.get('rounding_policy')
        if rounding != 'largest_remainder' or manifest.get('rounding_tie_order') != list(SPLITS):
            raise ValueError('Invalid frozen rounding policy')
    expected = _partition_counts(population, rounding)
    if manifest['counts']!=expected:raise ValueError('Partition sizes violate frozen 20/20/60 policy')
    seen=set()
    for name in SPLITS:
        ids=manifest['splits'][name]
        if len(ids)!=expected[name] or len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('Overlapping, duplicated or incomplete partition')
        seen.update(ids)
    records=manifest['records']
    if seen!={r['id'] for r in records} or len(records)!=population:
        raise ValueError('Partition does not preserve the complete population')
    if manifest['population_sha256']!=canonical_sha256(records):
        raise ValueError('Population hash changed')
    if len({r['public_task_sha256'] for r in records})!=population:
        raise ValueError('Duplicate public tasks cross the independent population')
    if not manifest['all_repetitions_follow_unit_split']:
        raise ValueError('Repeated runs cannot cross split boundaries')


def materialize_partition(items, manifest, *, include_test=False):
    validate_partition(manifest)
    by_id={r['id']:r for r in items}
    if len(by_id)!=len(items) or set(by_id)!={r['id'] for r in manifest['records']}:
        raise ValueError('Source items do not match the frozen population')
    return {'train':[by_id[i] for i in manifest['splits']['train']],
            'development':[by_id[i] for i in manifest['splits']['validation']],
            'test':[by_id[i] for i in manifest['splits']['test']] if include_test else []}

"""Lossless public-context dictionary for the native CoT continuation author."""
from copy import deepcopy
import hashlib
import json

INSTRUCTION = (' Public evidence uses a lossless dictionary: a {"$public_text":SHA256} '
               'object denotes the exact string in public_text_dictionary. Expand references '
               'mentally wherever they occur, including earlier receipts, latest observation '
               'and native reasoning. All chronological receipts, tool inputs, descriptions, '
               'bindings and task conditions remain present. Repeated bytes are stored once; '
               'repeated occurrences do not prove that an operation is newly necessary. '
               'Reuse discovered contact IDs and conversation IDs. Do not reread an unchanged '
               'contact directory when the required identity is already publicly established.')


def compact(state):
    """Intern repeated long strings, without omitting or summarizing any evidence."""
    if not isinstance(state, dict) or 'public_text_dictionary' in state:
        raise ValueError('Require one original public author state')
    counts = {}

    def count(value):
        if isinstance(value, str) and len(value) >= 160:
            counts[value] = counts.get(value, 0) + 1
        elif isinstance(value, dict):
            if set(value) == {'$public_text'}:
                raise ValueError('Reference collision in original public data')
            for item in value.values():
                count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)

    count(state)
    dictionary = {}

    def encode(value):
        if isinstance(value, str) and counts.get(value, 0) > 1:
            key = hashlib.sha256(value.encode()).hexdigest()
            if key in dictionary and dictionary[key] != value:
                raise ValueError('Public evidence digest collision')
            dictionary[key] = value
            return {'$public_text': key}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return deepcopy(value)

    result = encode(state)
    result['public_text_dictionary'] = dictionary
    if restore(result) != state:
        raise ValueError('Public context round trip changed original evidence')
    return result


def restore(state):
    dictionary = state['public_text_dictionary']
    for key, value in dictionary.items():
        if not isinstance(value, str) or hashlib.sha256(value.encode()).hexdigest() != key:
            raise ValueError('Public text reference does not match original bytes')

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'$public_text'}:
                return dictionary[value['$public_text']]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return deepcopy(value)

    return decode({key: value for key, value in state.items() if key != 'public_text_dictionary'})


def compact_request(body):
    result = deepcopy(body)
    original = json.loads(result['messages'][1]['content'])
    encoded = compact(original)
    text = json.dumps(encoded, ensure_ascii=False, separators=(',', ':'))
    # Small dictionaries can cost more. Keep the original full state in that case.
    if len(text.encode()) < len(result['messages'][1]['content'].encode()):
        result['messages'][1]['content'] = text
    return result

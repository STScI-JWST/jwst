"""Test the Registry"""
import sys
from copy import deepcopy
from tempfile import NamedTemporaryFile

from jwst.associations.lib.keyvalue_registry import KeyValueRegistry
from jwst.associations.registry import import_from_file

import pytest


def test_registry_match(full_pool_rules):
    """Test the match method"""
    pool, rules, pool_fname = full_pool_rules

    assert len(rules.schemas) > 0
    matches = rules.match(pool[1])
    assert isinstance(matches, tuple)
    asns = matches[0]
    reprocess_list = matches[1]
    assert isinstance(asns, list)
    assert isinstance(reprocess_list, list)
    assert len(asns) >= 1


# Tests below for keyvalue_registry

def test_dict_like():
    """Test the basic to ensure similar to a dict"""
    data = {'a': 1, 'b': 2}
    kvr = KeyValueRegistry(data)
    assert data.keys() == kvr.keys()
    assert set(data.values()) == set(kvr.values())

    assert kvr.get('a') == 1
    assert kvr.get('c', 3) == 3

    keys, values = zip(*kvr.items())
    assert set(data.keys()) == set(keys)
    assert set(data.values()) == set(values)

    kvr_copy = kvr.copy()
    assert set(kvr_copy) == set(kvr)

    assert kvr.pop('a') == 1
    assert kvr.pop('a', 3) == 3
    assert kvr.popitem() == ('b', 2)
    with pytest.raises(KeyError):
        kvr.popitem()

    kvr = KeyValueRegistry()
    kvr.update(data)
    assert data.keys() == kvr.keys()
    assert set(data.values()) == set(kvr.values())

    kvr.clear()
    assert len(kvr) == 0


def test_default():
    kvr = KeyValueRegistry(default={'a': 1})
    assert kvr.default == 'a'
    assert kvr[None] == 1


def test_tuple():
    data = ('a', 1)
    kvr = KeyValueRegistry(data)
    assert kvr['a'] == 1

    kvr = KeyValueRegistry(default=data)
    assert kvr['a'] == 1
    assert kvr[None] == 1


def test_fn():
    def fn():
        return 1

    kvr = KeyValueRegistry(fn)
    assert kvr[fn.__name__] == fn

    kvr = KeyValueRegistry(default=fn)
    assert kvr[fn.__name__] == fn
    assert kvr[None] == fn


def test_decorator():
    kvr = KeyValueRegistry()

    @kvr
    def fn():
        return 1

    assert kvr[fn.__name__] == fn
    assert kvr[fn.__name__]() == fn()

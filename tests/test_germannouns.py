import pytest

from germannouns import create_ngram

def test_create_ngram():
    assert create_ngram ('foo', 2) == [('<S>', 'f'), 
                                   ('f', 'o'),
                                   ('o', 'o'),
                                   ('o', '<E>')]
    assert create_ngram ('foo', 3) == [('<S>', 'f', 'o'),
                                       ('f', 'o', 'o'),
                                       ('o', 'o', '<E>')]
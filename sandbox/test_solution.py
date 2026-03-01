import pytest
from solution import word_break

def test_word_break():
    assert word_break("leetcode", ["leet", "code"]) == True
    assert word_break("applepenapple", ["apple", "pen"]) == True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False

def test_empty_string():
    assert word_break("", []) == True
    assert word_break("", ["test"]) == True

def test_single_word():
    assert word_break("test", ["test"]) == True
    assert word_break("test", ["other"]) == False

def test_long_string():
    long_string = "a" * 1000
    assert word_break(long_string, ["a"]) == True
import pytest
import quantlop as ql


def test_positional_args_init():
    pw = ql.PauliWord(0.53, "ZZIXY")
    assert pw.num_qubits == 5
    assert pw.coeff == 0.53
    assert pw.string == "ZZIXY"


def test_keyword_args_init():
    pw = ql.PauliWord(coeff=0.53, string="ZZIXY")
    assert pw.num_qubits == 5
    assert pw.coeff == 0.53
    assert pw.string == "ZZIXY"


def test_num_qubits_read_only():
    pw = ql.PauliWord(coeff=0.53, string="ZZIXY")
    with pytest.raises(AttributeError, match="property 'num_qubits' of 'PauliWord' object has no setter"):
        pw.num_qubits = 4


def test_coeff_read_only():
    pw = ql.PauliWord(coeff=0.53, string="ZZIXY")
    with pytest.raises(AttributeError, match="property 'coeff' of 'PauliWord' object has no setter"):
        pw.coeff = 1.29


def test_string_read_only():
    pw = ql.PauliWord(coeff=0.53, string="ZZIXY")
    with pytest.raises(AttributeError, match="property 'string' of 'PauliWord' object has no setter"):
        pw.string = "YYYZI"

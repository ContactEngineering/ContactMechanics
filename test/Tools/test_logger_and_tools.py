"""
Tests of the logger and of small helper functions.
"""

import numpy as np
import pytest

from ContactMechanics.Tools.Logger import Logger, quiet, read_convergence_log
from ContactMechanics.Tools.common import (compare_containers, compute_wavevectors, evaluate_gradient,
                                           get_q_from_lambda, mean_err)


def test_logger_file(tmp_path):
    fn = tmp_path / 'log.out'
    logger = Logger(str(fn))
    assert logger.has_logfile()
    logger.pr('message')
    for i in range(3):
        logger.st(['name', 'it', 'value', 'vector'], ['a', i, 0.5 * i, np.array([1., 2.])])
    logger.flush()
    lines = fn.read_text().splitlines()
    assert lines[0].endswith('message')
    # Header with one column per vector component
    assert '4:vector(x)' in lines[1] and '5:vector(y)' in lines[1]
    assert len(lines) == 5


def test_read_convergence_log(tmp_path):
    fn = tmp_path / 'convergence.out'
    logger = Logger(str(fn))
    for i in range(3):
        logger.st(['it', 'energy'], [i, 0.5 * i])
    logger.flush()
    df = read_convergence_log(str(fn))
    np.testing.assert_allclose(df['energy'], [0, 0.5, 1.0])


def test_logger_output_frequency(tmp_path):
    fn = tmp_path / 'log.out'
    logger = Logger(str(fn), outevery=3, sepevery=100)
    for i in range(9):
        logger.st(['it'], [i])
    logger.st(['it'], [99], force_print=True)
    values = [line.split()[0] for line in fn.read_text().splitlines() if not line.startswith('#')]
    assert values == ['2', '5', '8', '99']


def test_logger_one_file_per_iteration(tmp_path):
    pattern = str(tmp_path / 'log.{}.out')
    (tmp_path / 'log.1.out').write_text('existing file')
    logger = Logger(pattern)
    logger.pr('first iteration')
    logger.iteration_finished()
    logger.pr('second iteration')
    logger.flush()
    assert 'first iteration' in (tmp_path / 'log.1.out').read_text()
    assert 'second iteration' in (tmp_path / 'log.2.out').read_text()
    # Existing files are backed up
    assert (tmp_path / 'log.1.out.0.bak').read_text() == 'existing file'


def test_quiet_logger_does_not_accumulate_output():
    nb_buffered = len(quiet.buffer)
    for i in range(100):
        quiet.st(['it'], [i])
        quiet.pr('message')
    assert not quiet.has_logfile()
    assert len(quiet.buffer) == nb_buffered


def test_logger_warn(capsys):
    logger = Logger(None)
    logger.warn('something')
    assert 'Warning: something' in capsys.readouterr().out
    assert logger.get_logfile() is None


def test_compare_containers():
    assert compare_containers([1, 2, 3], (1, 2., 3.))
    assert not compare_containers([1, 2], (1, 2, 3))
    assert not compare_containers([1, 2], (1, 3))
    assert not compare_containers([1, 2], 5)


def test_evaluate_gradient():
    x = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(evaluate_gradient(lambda x: np.sum(x ** 2), x, 1e-4), 2 * x, rtol=1e-6)


def test_mean_err():
    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    assert mean_err(a, b) == 1
    assert mean_err(a, np.ones((4, 3)), rfft=True) == 1
    with pytest.raises(Exception):
        mean_err(a, np.ones((2, 2)))


def test_wavevectors():
    qx, = compute_wavevectors(8, 2., 1)
    np.testing.assert_allclose(qx, 2 * np.pi * np.fft.fftfreq(8, 0.25))
    qx, qy = compute_wavevectors((8, 4), (2., 1.), 2)
    assert len(qx) == 8 and len(qy) == 4
    np.testing.assert_allclose(get_q_from_lambda(0.5, 2.), (np.pi, 4 * np.pi))
    assert get_q_from_lambda(0, 2.)[1] == np.inf

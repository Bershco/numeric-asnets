"""Tests for robust MDPSim problem-name resolution."""
import unittest

from asnets.utils.mdpsim_utils import PDDLLoadError, parse_problem_args


class FakeMDPSim:
    def __init__(self, problems):
        self._problems = problems

    @staticmethod
    def parse_file(_path):
        return True

    def get_problems(self):
        return self._problems


class ParseProblemArgsTest(unittest.TestCase):
    def test_exact_match_takes_precedence(self):
        expected = object()
        module = FakeMDPSim({'ZTRAVEL-1': expected, 'ztravel-1': object()})
        self.assertIs(
            parse_problem_args(module, ['domain.pddl'], 'ZTRAVEL-1'),
            expected,
        )

    def test_unique_case_insensitive_match(self):
        expected = object()
        module = FakeMDPSim({'ztravel-1': expected})
        self.assertIs(
            parse_problem_args(module, ['domain.pddl'], 'ZTRAVEL-1'),
            expected,
        )

    def test_missing_name_fails_clearly(self):
        module = FakeMDPSim({'ztravel-1': object()})
        with self.assertRaisesRegex(PDDLLoadError,
                                    'Could not find problem missing'):
            parse_problem_args(module, ['domain.pddl'], 'missing')

    def test_ambiguous_case_insensitive_name_fails_clearly(self):
        module = FakeMDPSim({
            'ztravel-1': object(),
            'Ztravel-1': object(),
        })
        with self.assertRaisesRegex(PDDLLoadError,
                                    'ambiguous under case-insensitive'):
            parse_problem_args(module, ['domain.pddl'], 'ZTRAVEL-1')


if __name__ == '__main__':
    unittest.main()

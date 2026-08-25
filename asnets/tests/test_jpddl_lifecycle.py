import unittest
from unittest import mock

from asnets.interfaces import jpddl_interface


class JpddlLifecycleTests(unittest.TestCase):
    def test_stopped_jvm_is_not_restarted_in_process(self):
        with mock.patch.object(
                jpddl_interface.jpype, "isJVMStarted", return_value=False), \
                mock.patch.object(jpddl_interface.jpype, "startJVM") as start:
            with self.assertRaisesRegex(RuntimeError, "cannot restart"):
                jpddl_interface.ensure_jvm()
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

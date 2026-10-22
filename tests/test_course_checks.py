from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import check_course


class CourseSourceChecks(unittest.TestCase):
    def test_new_root_guides_have_their_local_links_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "CURRICULUM.md"):
                (root / name).write_text("", encoding="utf-8")
            (root / "SETUP.md").write_text(
                "[CPU profile](requirements/linux-cpu.txt)", encoding="utf-8",
            )
            with patch.object(check_course, "ROOT", root):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"SETUP\.md links to requirements/linux-cpu\.txt",
                ):
                    check_course.check_sources([])


if __name__ == "__main__":
    unittest.main()

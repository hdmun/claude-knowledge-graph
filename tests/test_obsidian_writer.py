from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claude_knowledge_graph import obsidian_writer as ow


class ObsidianWriterTests(unittest.TestCase):
    def test_build_session_filename_map_sequences_per_project_and_date(self) -> None:
        qa_list = [
            {
                "timestamp": "2026-03-20T09:00:00",
                "session_id": "s1",
                "project_slug": "project-a",
                "source_platform": "codex",
                "qwen_result": {"title": "Alpha"},
            },
            {
                "timestamp": "2026-03-20T11:00:00",
                "session_id": "s2",
                "project_slug": "project-a",
                "source_platform": "codex",
                "qwen_result": {"title": "Beta"},
            },
            {
                "timestamp": "2026-03-21T08:00:00",
                "session_id": "s3",
                "project_slug": "project-a",
                "source_platform": "codex",
                "qwen_result": {"title": "Gamma"},
            },
            {
                "timestamp": "2026-03-20T10:00:00",
                "session_id": "s4",
                "project_slug": "project-b",
                "source_platform": "codex",
                "qwen_result": {"title": "Delta"},
            },
        ]

        filename_map = ow.build_session_filename_map(qa_list)

        self.assertEqual(filename_map[ow.session_key(qa_list[0])], "2026-03-20_01_Alpha")
        self.assertEqual(filename_map[ow.session_key(qa_list[1])], "2026-03-20_02_Beta")
        self.assertEqual(filename_map[ow.session_key(qa_list[2])], "2026-03-21_01_Gamma")
        self.assertEqual(filename_map[ow.session_key(qa_list[3])], "2026-03-20_01_Delta")

    def test_existing_session_rename_map_and_link_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            projects_dir = root / "projects"
            daily_dir = root / "daily"
            concepts_dir = root / "concepts"
            moc_path = root / "_MOC.md"
            session_dir = projects_dir / "project-a" / "sessions"
            session_dir.mkdir(parents=True)
            daily_dir.mkdir()
            concepts_dir.mkdir()

            old_session = session_dir / "2026-03-20_Alpha.md"
            old_session.write_text(
                """---
title: "Alpha"
date: 2026-03-20
time: "09:00"
session_id: s1
source_platform: codex
project_slug: "project-a"
---

# Alpha
"""
            )
            daily_file = daily_dir / "2026-03-20.md"
            daily_file.write_text("[[projects/project-a/sessions/2026-03-20_Alpha|2026-03-20_Alpha]]")
            concept_file = concepts_dir / "Alpha.md"
            concept_file.write_text("- [[projects/project-a/sessions/2026-03-20_Alpha]]")
            moc_path.write_text("- [[projects/project-a/sessions/2026-03-20_Alpha]]")

            qa = {
                "timestamp": "2026-03-20T09:00:00",
                "session_id": "s1",
                "project_slug": "project-a",
                "source_platform": "codex",
                "qwen_result": {"title": "Alpha"},
            }
            filename_map = ow.build_session_filename_map([qa])

            old_projects_dir = ow.PROJECTS_DIR
            old_daily_dir = ow.DAILY_DIR
            old_concepts_dir = ow.CONCEPTS_DIR
            old_moc_path = ow.MOC_PATH
            try:
                ow.PROJECTS_DIR = projects_dir
                ow.DAILY_DIR = daily_dir
                ow.CONCEPTS_DIR = concepts_dir
                ow.MOC_PATH = moc_path

                rename_map = ow.build_existing_session_rename_map([qa], filename_map)
                self.assertEqual(
                    rename_map,
                    {
                        "projects/project-a/sessions/2026-03-20_Alpha": "projects/project-a/sessions/2026-03-20_01_Alpha"
                    },
                )

                ow.rename_existing_session_files(rename_map)
                ow.rewrite_session_links(rename_map)
            finally:
                ow.PROJECTS_DIR = old_projects_dir
                ow.DAILY_DIR = old_daily_dir
                ow.CONCEPTS_DIR = old_concepts_dir
                ow.MOC_PATH = old_moc_path

            self.assertFalse(old_session.exists())
            self.assertTrue((session_dir / "2026-03-20_01_Alpha.md").exists())
            self.assertIn("2026-03-20_01_Alpha", daily_file.read_text())
            self.assertIn("2026-03-20_01_Alpha", concept_file.read_text())
            self.assertIn("2026-03-20_01_Alpha", moc_path.read_text())

    def test_existing_rename_skips_sequences_reserved_by_current_qas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            projects_dir = root / "projects"
            session_dir = projects_dir / "project-a" / "sessions"
            session_dir.mkdir(parents=True)

            (session_dir / "2026-03-20_Legacy.md").write_text(
                """---
title: "Legacy"
date: 2026-03-20
time: "12:00"
session_id: old
source_platform: codex
project_slug: "project-a"
---
"""
            )

            qa_list = [
                {
                    "timestamp": "2026-03-20T09:00:00",
                    "session_id": "new",
                    "project_slug": "project-a",
                    "source_platform": "codex",
                    "qwen_result": {"title": "Fresh"},
                }
            ]
            filename_map = ow.build_session_filename_map(qa_list)

            old_projects_dir = ow.PROJECTS_DIR
            try:
                ow.PROJECTS_DIR = projects_dir
                rename_map = ow.build_existing_session_rename_map(qa_list, filename_map)
            finally:
                ow.PROJECTS_DIR = old_projects_dir

            self.assertEqual(
                rename_map,
                {
                    "projects/project-a/sessions/2026-03-20_Legacy": "projects/project-a/sessions/2026-03-20_02_Legacy"
                },
            )


if __name__ == "__main__":
    unittest.main()

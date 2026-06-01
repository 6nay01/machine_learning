import io
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path

from compute_test_true_cardinality import (
    count_total_queries,
    create_schema,
    import_table,
    load_completed_ids,
    stream_true_cardinalities,
)


class ImportTableStreamModeTest(unittest.TestCase):
    def test_import_table_supports_tar_stream_file_object(self) -> None:
        csv_bytes = b"1,10,20,,,,\n2,11,21,,,,2\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "mini.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="cast_info.csv")
                info.size = len(csv_bytes)
                archive.addfile(info, io.BytesIO(csv_bytes))

            conn = sqlite3.connect(":memory:")
            try:
                create_schema(conn)
                with tarfile.open(tar_path, "r|gz") as archive:
                    member = next(m for m in archive if m.name == "cast_info.csv")
                    file_obj = archive.extractfile(member)
                    self.assertIsNotNone(file_obj)
                    import_table(conn, "cast_info", file_obj)

                rows = conn.execute(
                    "SELECT person_id, movie_id, role_id FROM cast_info ORDER BY movie_id"
                ).fetchall()
                self.assertEqual(rows, [(10, 20, None), (11, 21, 2)])
            finally:
                conn.close()

    def test_import_table_supports_quoted_commas_and_newlines(self) -> None:
        csv_bytes = (
            b'1,10,20,,"note with, comma\nand newline",,3\n'
            b'2,11,21,,,,2\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "mini.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="cast_info.csv")
                info.size = len(csv_bytes)
                archive.addfile(info, io.BytesIO(csv_bytes))

            conn = sqlite3.connect(":memory:")
            try:
                create_schema(conn)
                with tarfile.open(tar_path, "r|gz") as archive:
                    member = next(m for m in archive if m.name == "cast_info.csv")
                    file_obj = archive.extractfile(member)
                    self.assertIsNotNone(file_obj)
                    import_table(conn, "cast_info", file_obj)

                rows = conn.execute(
                    "SELECT person_id, movie_id, role_id FROM cast_info ORDER BY movie_id"
                ).fetchall()
                self.assertEqual(rows, [(10, 20, 3), (11, 21, 2)])
            finally:
                conn.close()

    def test_import_table_supports_backslash_escaped_quotes(self) -> None:
        csv_bytes = (
            b'1,10,20,,"(segment \\\\\\"Pesnya, ili Kak\\\\\\")",,3\n'
            b'2,11,21,,,,2\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "mini.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo(name="cast_info.csv")
                info.size = len(csv_bytes)
                archive.addfile(info, io.BytesIO(csv_bytes))

            conn = sqlite3.connect(":memory:")
            try:
                create_schema(conn)
                with tarfile.open(tar_path, "r|gz") as archive:
                    member = next(m for m in archive if m.name == "cast_info.csv")
                    file_obj = archive.extractfile(member)
                    self.assertIsNotNone(file_obj)
                    import_table(conn, "cast_info", file_obj)

                rows = conn.execute(
                    "SELECT person_id, movie_id, role_id FROM cast_info ORDER BY movie_id"
                ).fetchall()
                self.assertEqual(rows, [(10, 20, 3), (11, 21, 2)])
            finally:
                conn.close()


class ResumeWriteTest(unittest.TestCase):
    def test_stream_true_cardinalities_appends_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "mini.sqlite"
            test_csv = tmp / "test.csv"
            output_csv = tmp / "out.csv"

            test_csv.write_text(
                "Id,Tables,Join Conditions,Predicates\n"
                '1,cast_info ci,,"ci.person_id,=,10"\n'
                '2,cast_info ci,,"ci.role_id,=,2"\n',
                encoding="utf-8",
            )

            conn = sqlite3.connect(db_path)
            try:
                create_schema(conn)
                conn.executemany(
                    "INSERT INTO cast_info (person_id, movie_id, role_id) VALUES (?, ?, ?)",
                    [(10, 100, 1), (11, 101, 2), (12, 102, 2)],
                )
                conn.commit()

                stream_true_cardinalities(conn, test_csv, output_csv, limit=1)
                self.assertEqual(load_completed_ids(output_csv), {"1"})
                self.assertEqual(count_total_queries(test_csv, None), 2)

                stream_true_cardinalities(conn, test_csv, output_csv, limit=None)
            finally:
                conn.close()

            rows = output_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], "Id,Tables,Join Conditions,Predicates,Cardinality")
            self.assertTrue(rows[1].startswith("1,"))
            self.assertTrue(rows[2].startswith("2,"))


if __name__ == "__main__":
    unittest.main()

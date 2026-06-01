from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path


SQLITE_BATCH_SIZE = 50_000

FULL_TABLE_COLUMNS: dict[str, list[str]] = {
    "title": [
        "id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "imdb_id",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "series_years",
        "md5sum",
    ],
    "cast_info": [
        "id",
        "person_id",
        "movie_id",
        "person_role_id",
        "note",
        "nr_order",
        "role_id",
    ],
    "movie_companies": [
        "id",
        "movie_id",
        "company_id",
        "company_type_id",
        "note",
    ],
    "movie_info": [
        "id",
        "movie_id",
        "info_type_id",
        "info",
        "note",
    ],
    "movie_info_idx": [
        "id",
        "movie_id",
        "info_type_id",
        "info",
        "note",
    ],
    "movie_keyword": [
        "id",
        "movie_id",
        "keyword_id",
    ],
}

TABLE_COLUMN_TYPES: dict[str, dict[str, str]] = {
    "title": {
        "id": "INTEGER",
        "kind_id": "INTEGER",
        "production_year": "INTEGER",
    },
    "cast_info": {
        "person_id": "INTEGER",
        "movie_id": "INTEGER",
        "role_id": "INTEGER",
    },
    "movie_companies": {
        "movie_id": "INTEGER",
        "company_id": "INTEGER",
        "company_type_id": "INTEGER",
    },
    "movie_info": {
        "movie_id": "INTEGER",
        "info_type_id": "INTEGER",
    },
    "movie_info_idx": {
        "movie_id": "INTEGER",
        "info_type_id": "INTEGER",
    },
    "movie_keyword": {
        "movie_id": "INTEGER",
        "keyword_id": "INTEGER",
    },
}

TARGET_COLUMNS: dict[str, list[str]] = {
    table: list(columns.keys()) for table, columns in TABLE_COLUMN_TYPES.items()
}

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_title_id ON title(id)",
    "CREATE INDEX IF NOT EXISTS idx_title_kind_year_id ON title(kind_id, production_year, id)",
    "CREATE INDEX IF NOT EXISTS idx_cast_info_person_role_movie ON cast_info(person_id, role_id, movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_cast_info_movie ON cast_info(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_companies_company_type_movie ON movie_companies(company_id, company_type_id, movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_companies_movie ON movie_companies(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_info_type_movie ON movie_info(info_type_id, movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_movie ON movie_info(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_idx_info_type_movie ON movie_info_idx(info_type_id, movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_info_idx_movie ON movie_info_idx(movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_keyword_keyword_movie ON movie_keyword(keyword_id, movie_id)",
    "CREATE INDEX IF NOT EXISTS idx_movie_keyword_movie ON movie_keyword(movie_id)",
]

TABLE_ALIASES = {
    "t": "title",
    "ci": "cast_info",
    "mc": "movie_companies",
    "mi": "movie_info",
    "mi_idx": "movie_info_idx",
    "mk": "movie_keyword",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import JOB IMDb CSVs into SQLite and compute true cardinalities for test.csv."
    )
    parser.add_argument(
        "--imdb-tgz",
        type=Path,
        default=Path("datasets/imdb.tgz"),
        help="Path to the JOB IMDb tar.gz archive.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("test.csv"),
        help="Path to the test query CSV.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("datasets/imdb_job_minimal.sqlite"),
        help="SQLite database to create or reuse.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("test_with_true_cardinality.csv"),
        help="Output CSV with the original test rows plus true Cardinality.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild the SQLite database before importing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only compute the first N test queries for spot checks.",
    )
    return parser.parse_args()


def configure_sqlite(conn: sqlite3.Connection) -> None:
    pragmas = [
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
        "PRAGMA temp_store = MEMORY",
        "PRAGMA cache_size = -200000",
        "PRAGMA foreign_keys = OFF",
    ]
    for pragma in pragmas:
        conn.execute(pragma)


def create_schema(conn: sqlite3.Connection) -> None:
    for table_name, columns in TABLE_COLUMN_TYPES.items():
        col_defs = ", ".join(f"{name} {col_type}" for name, col_type in columns.items())
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")
    conn.commit()


def table_has_rows(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
    return row is not None


def ensure_empty_database(db_path: Path, force_rebuild: bool) -> None:
    if force_rebuild and db_path.exists():
        db_path.unlink()


def import_archive(conn: sqlite3.Connection, imdb_tgz: Path) -> None:
    if not imdb_tgz.exists():
        raise FileNotFoundError(f"未找到压缩包: {imdb_tgz}")

    if all(table_has_rows(conn, table) for table in TARGET_COLUMNS):
        print("[import] 目标表已完整存在，跳过压缩包扫描")
        return

    imported_any = False
    with tarfile.open(imdb_tgz, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue

            name = Path(member.name).name
            if not name.endswith(".csv"):
                continue

            table_name = name[:-4]
            if table_name not in TARGET_COLUMNS:
                continue

            if table_has_rows(conn, table_name):
                print(f"[import] 跳过已存在表: {table_name}")
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise RuntimeError(f"无法读取压缩成员: {member.name}")

            print(f"[import] 开始导入 {table_name}")
            import_table(conn, table_name, file_obj)
            imported_any = True

    missing_tables = [table for table in TARGET_COLUMNS if not table_has_rows(conn, table)]
    if missing_tables:
        raise RuntimeError(f"以下目标表未成功导入: {', '.join(missing_tables)}")
    if imported_any:
        print("[import] 所有目标表导入完成")


def import_table(conn: sqlite3.Connection, table_name: str, file_obj: io.BufferedReader) -> None:
    source_columns = FULL_TABLE_COLUMNS[table_name]
    target_columns = TARGET_COLUMNS[table_name]
    index_map = [source_columns.index(column) for column in target_columns]
    placeholders = ", ".join("?" for _ in target_columns)
    insert_sql = (
        f"INSERT INTO {table_name} ({', '.join(target_columns)}) VALUES ({placeholders})"
    )

    text_stream = TarTextStream(file_obj)
    reader = csv.reader(text_stream, escapechar="\\")
    batch: list[tuple[int | None, ...]] = []
    total_rows = 0

    for row in reader:
        selected = tuple(to_int_or_none(row[idx]) for idx in index_map)
        batch.append(selected)
        if len(batch) >= SQLITE_BATCH_SIZE:
            conn.executemany(insert_sql, batch)
            conn.commit()
            total_rows += len(batch)
            print(f"[import] {table_name}: 已写入 {total_rows} 行")
            batch.clear()

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        total_rows += len(batch)
        print(f"[import] {table_name}: 已写入 {total_rows} 行")


class TarTextStream(io.TextIOBase):
    def __init__(self, file_obj: io.BufferedReader, encoding: str = "utf-8") -> None:
        self._buffer = io.BufferedReader(file_obj)
        self._encoding = encoding

    def readable(self) -> bool:
        return True

    def __iter__(self) -> "TarTextStream":
        return self

    def __next__(self) -> str:
        line = self.readline()
        if line == "":
            raise StopIteration
        return line

    def readline(self, size: int = -1) -> str:
        raw = self._buffer.readline(size)
        if not raw:
            return ""
        return raw.decode(self._encoding)


def to_int_or_none(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def create_indexes(conn: sqlite3.Connection) -> None:
    print("[index] 开始创建索引")
    for statement in INDEX_STATEMENTS:
        conn.execute(statement)
    conn.commit()
    print("[index] 索引创建完成")


def parse_tables(tables_str: str) -> list[str]:
    return [part.strip() for part in tables_str.split(",") if part.strip()]


def parse_join_conditions(join_str: str) -> list[str]:
    if not join_str.strip():
        return []
    return [part.strip() for part in join_str.split(",") if part.strip()]


def parse_predicates(predicate_str: str) -> list[tuple[str, str, int]]:
    parts = [part.strip() for part in predicate_str.split(",") if part.strip()]
    if len(parts) % 3 != 0:
        raise ValueError(f"谓词格式不合法: {predicate_str}")
    predicates = []
    for idx in range(0, len(parts), 3):
        column, operator, value = parts[idx : idx + 3]
        if operator not in {"=", "<", ">", "<=", ">="}:
            raise ValueError(f"不支持的操作符: {operator}")
        predicates.append((column, operator, int(value)))
    return predicates


def build_count_query(row: dict[str, str]) -> tuple[str, list[int]]:
    from_clause = ", ".join(parse_tables(row["Tables"]))
    where_clauses = []
    params: list[int] = []

    for join_condition in parse_join_conditions(row["Join Conditions"]):
        where_clauses.append(join_condition)

    for column, operator, value in parse_predicates(row["Predicates"]):
        where_clauses.append(f"{column} {operator} ?")
        params.append(value)

    query = f"SELECT COUNT(*) FROM {from_clause}"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    return query, params


def count_total_queries(test_csv: Path, limit: int | None) -> int:
    with test_csv.open(newline="", encoding="utf-8") as handle:
        total = max(sum(1 for _ in handle) - 1, 0)
    if limit is not None:
        return min(total, limit)
    return total


def load_completed_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()

    completed_ids: set[str] = set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return set()
        if "Id" not in reader.fieldnames:
            raise RuntimeError(f"续跑文件缺少 Id 列: {output_csv}")
        for row in reader:
            row_id = row.get("Id", "").strip()
            if row_id:
                completed_ids.add(row_id)
    return completed_ids


def open_output_writer(
    output_csv: Path, fieldnames: list[str]
) -> tuple[io.TextIOWrapper, csv.DictWriter]:
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    handle = output_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def stream_true_cardinalities(
    conn: sqlite3.Connection,
    test_csv: Path,
    output_csv: Path,
    limit: int | None,
) -> None:
    completed_ids = load_completed_ids(output_csv)
    total_queries = count_total_queries(test_csv, limit)
    print(f"[resume] 已完成 {len(completed_ids)} / {total_queries}")

    fieldnames = ["Id", "Tables", "Join Conditions", "Predicates", "Cardinality"]
    start_time = time.time()
    processed = 0
    written = 0

    output_handle, writer = open_output_writer(output_csv, fieldnames)
    try:
        with test_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                if limit is not None and idx > limit:
                    break

                row_id = row["Id"].strip()
                if row_id in completed_ids:
                    processed += 1
                    print(
                        f"[query] 已跳过 {processed} / {total_queries} "
                        f"(Id={row_id}, 已存在)"
                    )
                    continue

                sql, params = build_count_query(row)
                query_start = time.time()
                count = conn.execute(sql, params).fetchone()[0]
                query_elapsed = time.time() - query_start

                result_row: dict[str, str | int] = {
                    "Id": row["Id"],
                    "Tables": row["Tables"],
                    "Join Conditions": row["Join Conditions"],
                    "Predicates": row["Predicates"],
                    "Cardinality": int(count),
                }
                writer.writerow(result_row)
                output_handle.flush()

                processed += 1
                written += 1
                completed_ids.add(row_id)
                total_elapsed = time.time() - start_time
                print(
                    f"[query] 已完成 {processed} / {total_queries} "
                    f"(Id={row_id}, count={count}, 单条={query_elapsed:.3f}s, 累计={total_elapsed:.1f}s)"
                )
    finally:
        output_handle.close()

    if written == 0 and total_queries > 0 and len(completed_ids) == 0:
        raise RuntimeError("没有写出任何结果，请检查输入或续跑文件状态")


def main() -> None:
    args = parse_args()
    ensure_empty_database(args.db_path, args.force_rebuild)

    conn = sqlite3.connect(args.db_path)
    try:
        configure_sqlite(conn)
        create_schema(conn)
        import_archive(conn, args.imdb_tgz)
        create_indexes(conn)
        print("[query] 开始执行真实计数查询")
        stream_true_cardinalities(conn, args.test_csv, args.output_csv, args.limit)
    finally:
        conn.close()

    print(f"[done] 已写出结果文件: {args.output_csv}")


if __name__ == "__main__":
    main()

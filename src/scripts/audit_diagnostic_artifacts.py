
import csv
import hashlib
import json
import pickle

from pathlib import Path


DIAG_DIR = Path(
    "src/results/diagnostics"
)

OUT_PATH = Path(
    "src/results/final/"
    "diagnostic_artifact_inventory.json"
)


def _sha256(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _describe_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.reader(
            f
        )

        rows = list(
            reader
        )

    header = (
        rows[
            0
        ]
        if rows
        else []
    )

    return {
        "kind": "csv",
        "rows_excluding_header": max(
            0,
            len(rows) - 1,
        ),
        "columns": header,
    }


def _describe_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(
            f
        )

    result = {
        "kind": "json",
        "python_type": (
            type(
                obj
            ).__name__
        ),
    }

    if isinstance(
        obj,
        dict,
    ):
        result[
            "top_level_keys"
        ] = sorted(
            str(
                key
            )
            for key
            in obj.keys()
        )

    elif isinstance(
        obj,
        list,
    ):
        result[
            "length"
        ] = len(
            obj
        )

        if (
            obj
            and isinstance(
                obj[
                    0
                ],
                dict,
            )
        ):
            result[
                "first_record_keys"
            ] = sorted(
                str(
                    key
                )
                for key
                in obj[
                    0
                ].keys()
            )

    return result


def _describe_pickle(path):
    with open(
        path,
        "rb",
    ) as f:
        obj = pickle.load(
            f
        )

    result = {
        "kind": "pickle",
        "python_type": (
            type(
                obj
            ).__name__
        ),
    }

    if isinstance(
        obj,
        list,
    ):
        result[
            "length"
        ] = len(
            obj
        )

        if obj:
            first = obj[
                0
            ]

            result[
                "first_item_type"
            ] = type(
                first
            ).__name__

            if isinstance(
                first,
                dict,
            ):
                result[
                    "first_record_keys"
                ] = sorted(
                    str(
                        key
                    )
                    for key
                    in first.keys()
                )

    elif isinstance(
        obj,
        dict,
    ):
        result[
            "top_level_keys"
        ] = sorted(
            str(
                key
            )
            for key
            in obj.keys()
        )

    return result


def _describe(path):
    suffix = (
        path.suffix.lower()
    )

    if suffix == ".csv":
        return _describe_csv(
            path
        )

    if suffix == ".json":
        return _describe_json(
            path
        )

    if suffix == ".pkl":
        return _describe_pickle(
            path
        )

    return {
        "kind": "other"
    }


def main():
    if not DIAG_DIR.exists():
        raise FileNotFoundError(
            DIAG_DIR
        )

    files = sorted(
        path
        for path
        in DIAG_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in (
                ".pkl",
                ".json",
                ".csv",
            )
        )
    )

    if not files:
        raise RuntimeError(
            "No diagnostic artifacts "
            "were found."
        )

    inventory = {}

    print(
        "DIAGNOSTIC ARTIFACT "
        "FREEZE / INVENTORY"
    )

    print()

    print(
        f"Directory: {DIAG_DIR}"
    )

    print(
        f"Files found: {len(files)}"
    )

    print()

    for index, path in enumerate(
        files,
        start=1,
    ):
        description = (
            _describe(
                path
            )
        )

        digest = (
            _sha256(
                path
            )
        )

        inventory[
            str(
                path
            )
        ] = {
            "sha256": digest,
            **description,
        }

        print(
            "=" * 110
        )

        print(
            f"{index}. {path.name}"
        )

        print(
            "=" * 110
        )

        print(
            f"SHA-256:"
        )

        print(
            digest
        )

        print(
            f"Type: "
            f"{description['kind']}"
        )

        if (
            "length"
            in description
        ):
            print(
                f"Length: "
                f"{description['length']}"
            )

        if (
            "rows_excluding_header"
            in description
        ):
            print(
                f"Rows: "
                f"{description['rows_excluding_header']}"
            )

        if (
            "columns"
            in description
        ):
            print(
                "Columns:"
            )

            print(
                description[
                    "columns"
                ]
            )

        if (
            "top_level_keys"
            in description
        ):
            print(
                "Top-level keys:"
            )

            print(
                description[
                    "top_level_keys"
                ]
            )

        if (
            "first_record_keys"
            in description
        ):
            print(
                "First-record keys:"
            )

            print(
                description[
                    "first_record_keys"
                ]
            )

        print()

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(
            OUT_PATH
        )
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            inventory,
            f,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(
        OUT_PATH
    )

    print(
        "=" * 110
    )

    print(
        "INVENTORY COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Frozen inventory:"
    )

    print(
        OUT_PATH
    )


if __name__ == "__main__":
    main()

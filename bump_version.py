#!/usr/bin/env -S python3 -u
import argparse
import re
import sys
from pathlib import Path

_version_pattern = re.compile(
    r"__version__ *= *['\"]+([0-9]+)\.([0-9]+)\.([0-9]+)['\"]+"
)


def bump_version(major_by=0, middle_by=0, minor_by=1):
    project_dir = Path(__file__).parent
    version_script = project_dir / "utensor_cgen" / "_version.py"
    lines = []
    with version_script.open("r") as fid:
        for line in fid.readlines():
            match = _version_pattern.match(line.strip())
            if match:
                major = int(match.group(1)) + major_by
                middle = int(match.group(2)) + middle_by
                minor = int(match.group(3)) + minor_by
                line = f'__version__ = "{major}.{middle}.{minor}"\n'
            lines.append(line)
    with version_script.open("w") as fid:
        fid.write("".join(lines))

    ns = {}
    with version_script.open("r") as fid:
        exec(fid.read(), ns, ns)
    print(f"bumped version: {ns['__version__']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # setup the parser here
    parser.add_argument(
        "--major-by",
        type=int,
        default=0,
        help="bump major version number by given number (default: %(default)s)",
    )
    parser.add_argument(
        "--middle-by",
        type=int,
        default=0,
        help="bump middle version number by given number (default: %(default)s)",
    )
    parser.add_argument(
        "--minor-by",
        type=int,
        default=0,
        help="bump minor version number by given number (default: %(default)s)",
    )
    kwargs = vars(parser.parse_args())
    sys.exit(bump_version(**kwargs))

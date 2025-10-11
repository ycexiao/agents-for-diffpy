import argparse

from agenets_for_diffpy.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="agenets-for-diffpy",
        description=(
            "Agents that pick refinement configuration for diffpy\n\n"
            "For more information, visit: "
            "https://github.com/ycexiao/agenets-for-diffpy/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"agenets-for-diffpy {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()

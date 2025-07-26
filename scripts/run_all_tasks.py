import subprocess

from fittingroom.utils import get_default_constant


def main():
    tasks = get_default_constant("TASKS")

    for task in tasks:
        subprocess.run(
            [
                "python",
                "run.py",
                "--task",
                task,
            ]
        )


if __name__ == "__main__":
    print()
    main()
    print()

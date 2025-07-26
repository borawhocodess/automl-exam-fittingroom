import subprocess

from fittingroom.utils import get_default_constant


def main():
    tasks = get_default_constant("TASKS")
    log_level = get_default_constant("LOG_LEVEL")

    for task in tasks:
        subprocess.run(
            [
                "python",
                "run.py",
                "--task",
                task,
                "--log-level",
                log_level,
            ]
        )


if __name__ == "__main__":
    print()
    main()
    print()

"""Small shared entry point; both vision modules can also run independently."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run.py {apriltag|color} [options]")
        return 0 if len(sys.argv) >= 2 else 2
    module, args = sys.argv[1], sys.argv[2:]
    try:
        if module == "apriltag":
            from rov_vision.apriltag import main as run
        elif module == "color":
            from rov_vision.color_poles import main as run
        else:
            print("Unknown module. Choose apriltag or color.", file=sys.stderr)
            return 2
        return run(args)
    except ImportError as exc:
        print(
            f"Dependency error: {exc}. Install requirements.txt into the selected interpreter.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

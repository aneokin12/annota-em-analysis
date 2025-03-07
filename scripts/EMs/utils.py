#!/usr/bin/env python
import os
import argparse
import numpy as np



def clearTensorLogs(log_dir="tensorlogs"):
    """
    Deletes all files in the specified log_dir directory if it exists.
    Subdirectories (if any) are not touched.
    """
    if not os.path.isdir(log_dir):
        print(f"[INFO] Directory '{log_dir}' does not exist. Nothing to clear.")
        return

    removed_files = 0
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if os.path.isfile(full_path):
            os.remove(full_path)
            removed_files += 1
    print(f"[INFO] Removed {removed_files} files from '{log_dir}'.")

def createTensorDump(arr, filename="debug_array_dump.txt", log_dir="tensorlogs"):
    """
    Writes the full contents of 'arr' (no truncation) to a file in 'log_dir'.
    """
    # Temporarily store current print options so we can restore them
    old_opts = np.get_printoptions()

    try:
        # Make sure we don't truncate
        np.set_printoptions(threshold=np.inf, linewidth=200)

        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        # Construct full path to the output file
        file_path = os.path.join(log_dir, filename)

        with open(file_path, "w") as f:
            f.write(f"Array shape: {arr.shape}\n")
            f.write(f"Array dtype: {arr.dtype}\n")
            f.write("Array contents:\n")
            f.write(np.array2string(arr, max_line_width=200))

        print(f"[INFO] Saved full array to {file_path}")

    finally:
        # Restore old print options
        np.set_printoptions(**old_opts)

def main():
    parser = argparse.ArgumentParser(
        description="Command-line tool to manage tensorlogs and dump NumPy arrays in full detail."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all files in the 'tensorlogs' directory before doing anything else."
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        help="Path to a .npy file. If provided, the array will be loaded and dumped to a text file in 'tensorlogs'."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="debug_array_dump.txt",
        help="Name of the output text file (inside 'tensorlogs')."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="tensorlogs",
        help="Directory in which to store the text dumps or clear."
    )

    args = parser.parse_args()

    # 1) Clear the logs if requested
    if args.clear:
        clearTensorLogs(log_dir=args.logdir)

    # 2) If user provided a .npy path to dump, do that
    if args.dump is not None:
        if not os.path.isfile(args.dump):
            print(f"[ERROR] File '{args.dump}' not found.")
            return

        # Load array
        arr = np.load(args.dump, allow_pickle=True)
        print(f"[INFO] Loaded array from '{args.dump}', shape={arr.shape}, dtype={arr.dtype}")

        # Dump full array to the specified outfile
        createTensorDump(arr, filename=args.outfile, log_dir=args.logdir)

if __name__ == "__main__":
    main()

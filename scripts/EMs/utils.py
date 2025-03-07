import numpy as np
import os




#tensor dumper
def createTensorDump(arr, filename="debug_array_dump.txt"):
    """
    Writes the full contents of 'arr' (no truncation) to 'filename'.
    """
    # Temporarily store current print options so we can restore them later
    old_opts = np.get_printoptions()

    try:
        # Set printoptions so nothing is truncated
        np.set_printoptions(threshold=np.inf, linewidth=200)

        # Ensure directory for logs exists
        log_dir = "tensorlogs"
        os.makedirs(log_dir, exist_ok=True)

        # Create the full file path
        file_path = os.path.join(log_dir, filename)

        # Write the array data
        with open(file_path, "w") as f:
            f.write(f"Array shape: {arr.shape}\n")
            f.write(f"Array dtype: {arr.dtype}\n")
            f.write("Array contents:\n")
            f.write(np.array2string(arr, max_line_width=200))
            # Alternatively: f.write(repr(arr))

        print(f"[DEBUG] Saved full array to {file_path}")

    finally:
        # Restore old print options
        np.set_printoptions(**old_opts)
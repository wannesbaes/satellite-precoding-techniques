from pathlib import Path
import re

# Folder that contains the wrongly named plot files
PLOTS_DIR = Path("/Users/wannesbaes/Documents/THESIS/satellite precoding techniques/mu-mimo/report/channel_statistics/plots")

# Matches:
# ecdf_4x2.png
# pdf_16x8.png
pattern = re.compile(r"^(ecdf|pdf)_(\d+)x(\d+)\.png$", re.IGNORECASE)

def build_new_name(kind: str, nt: str, nr: str) -> str:
    kind = kind.lower()
    return (
        f"Nt_{nt}_K_1_Nr_{nr}"
        f"__IIDRayleighChannelModel__SVDPrecoder__SVDCombiner"
        f"__N_10M__UTs_1__{kind}.png"
    )

def rename_files():
    if not PLOTS_DIR.exists():
        print(f"Directory does not exist: {PLOTS_DIR}")
        return

    renamed = 0
    skipped = 0

    for old_path in PLOTS_DIR.iterdir():
        if not old_path.is_file():
            continue

        m = pattern.match(old_path.name)
        if not m:
            continue

        kind, nt, nr = m.groups()
        new_name = build_new_name(kind, nt, nr)
        new_path = old_path.with_name(new_name)

        if new_path.exists():
            print(f"SKIP (target exists): {old_path.name} -> {new_name}")
            skipped += 1
            continue

        old_path.rename(new_path)
        print(f"RENAMED: {old_path.name} -> {new_name}")
        renamed += 1

    print(f"\nDone. Renamed {renamed} file(s), skipped {skipped}.")

if __name__ == "__main__":
    rename_files()
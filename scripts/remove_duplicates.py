import os
from collections import defaultdict

import imagehash
import pandas as pd
from PIL import Image


def remove_duplicates(dam_folder: str, labels_csv: str):
    # Load labeled data
    labels_df = pd.read_csv(labels_csv)
    labeled_files = set(labels_df["test_image_name"])

    # Compute perceptual hashes for DAM folder
    def compute_hashes(folder_path):
        hash_dict = {}
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            try:
                with Image.open(path) as img:
                    img_hash = imagehash.phash(img)
                    hash_dict[filename] = str(img_hash)
            except Exception as e:
                print(f"Error with {filename}: {e}")
        return hash_dict

    dam_hashes = compute_hashes(dam_folder)

    # Group files by hash
    hash_groups = defaultdict(list)
    for filename, img_hash in dam_hashes.items():
        hash_groups[img_hash].append(filename)

    # Identify files to delete, prioritizing labeled ones
    to_delete = set()

    for img_hash, group in hash_groups.items():
        if len(group) > 1:
            # Split group into labeled vs. unlabeled
            labeled_in_group = [f for f in group if f in labeled_files]
            unlabeled_in_group = [f for f in group if f not in labeled_files]

            # Priority: Keep labeled files first
            if labeled_in_group:
                # Keep ONE labeled file (first occurrence)
                keep = labeled_in_group[0]
                # Delete all other duplicates (labeled or unlabeled)
                to_delete.update([f for f in group if f != keep])
            else:
                # Keep ONE unlabeled file, delete others
                keep = unlabeled_in_group[0]
                to_delete.update(unlabeled_in_group[1:])

    # Delete duplicates from DAM
    for filename in to_delete:
        os.remove(os.path.join(dam_folder, filename))
        print(f"Deleted: {filename}")

    # Update labels.csv to ONLY include files remaining in DAM
    remaining_files = set(os.listdir(dam_folder))
    labels_df = labels_df[labels_df["test_image_name"].isin(remaining_files)]

    # Final check
    assert len(labels_df) == labels_df["test_image_name"].nunique(), (
        "Duplicates in labels.csv!"
    )
    print("✅ Cleanup complete. Labels and DAM folder are now deduplicated.")
    labels_df.to_csv(labels_csv, index=False)


if __name__ == "__main__":
    remove_duplicates("data/DAM", "data/labels.csv")

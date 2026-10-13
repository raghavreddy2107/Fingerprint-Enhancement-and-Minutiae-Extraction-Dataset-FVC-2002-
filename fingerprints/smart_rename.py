import os
from pathlib import Path

def execution_rename():
    # Set target directory explicitly
    base_path = Path(r"E:\Biometrics\Code\fingerprints")
    
    if not base_path.exists():
        print(f"❌ Error: Path {base_path} does not exist!")
        return

    folders_processed = 0
    files_renamed = 0

    print("=" * 60)
    print("STARTING FINGERPRINT RE-INDEXING LOOP")
    print("=" * 60)

    # Loop dynamically from 1 to 40 to ensure correct order
    for i in range(1, 41):
        folder_name = f"finger_{i}"
        folder_path = base_path / folder_name
        
        if folder_path.is_dir():
            # Calculate the repeating 1-10 index value
            # Example: folder 10 -> 10, folder 11 -> 1, folder 20 -> 10
            cycle_index = i % 10
            if cycle_index == 0:
                cycle_index = 10
                
            # Determine prefixes
            old_prefix = f"{100 + cycle_index}_"
            new_prefix = f"f{i}_"
            
            count = 0
            # Look for matching images within the current folder loop
            for file_path in folder_path.glob(f"{old_prefix}*"):
                if file_path.is_file():
                    # Replace the dynamic old prefix with the absolute folder index prefix
                    new_name = file_path.name.replace(old_prefix, new_prefix)
                    new_file_path = file_path.with_name(new_name)
                    
                    file_path.rename(new_file_path)
                    count += 1
                    files_renamed += 1
            
            if count > 0:
                print(f"✅ {folder_name}: Converted {count} files ({old_prefix}* ➜ {new_prefix}*)")
                folders_processed += 1

    print("=" * 60)
    print(f"🚀 Execution Complete! Renamed {files_renamed} files across {folders_processed} folders.")
    print("=" * 60)

if __name__ == "__main__":
    execution_rename()

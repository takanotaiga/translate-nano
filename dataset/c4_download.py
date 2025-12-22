from huggingface_hub import snapshot_download

def main():
    snapshot_download(
        repo_id="allenai/c4",
        repo_type="dataset",
        local_dir="DATASET_DIR",
        allow_patterns=["en/*.json.gz"],
    )
    print("done")

if __name__ == "__main__":
    main()

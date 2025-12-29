from huggingface_hub import snapshot_download

def main():
    snapshot_download(
        repo_id="taigatakano/c4-en-64token",
        repo_type="dataset",
        local_dir="c4-en-64token",
        allow_patterns=["data/*.parquet"],
    )
    print("done")

if __name__ == "__main__":
    main()

import subprocess

def submit_to_kaggle(file_path, message, competition="aca-butterflies"):
    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "submit",
                "-c", competition,
                "-f", file_path,
                "-m", message
            ],
            capture_output=True,
            text=True
        )
        
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        if result.returncode == 0:
            print(f"Submission successful: {file_path}")
        else:
            print(f"Submission failed: {file_path}")

    except Exception as e:
        print(f"Error during submission: {e}")


# HOW TO SUBMIT 
# submit_to_kaggle( file_path=f"submission.csv", message=f"Best model test script- architecture}")
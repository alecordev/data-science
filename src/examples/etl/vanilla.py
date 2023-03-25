import subprocess

# Define ETL jobs to run
etl_jobs = [
    {"name": "job1", "cmd": ["python", "job1.py"], "output": "job1_output.txt"},
    {"name": "job2", "cmd": ["python", "job2.py"], "output": "job2_output.txt"},
    {"name": "job3", "cmd": ["python", "job3.py"], "output": "job3_output.txt"},
]

# Loop through each ETL job and run it
for job in etl_jobs:
    print(f"Running ETL job: {job['name']}")

    # Run the command using subprocess
    try:
        output = subprocess.check_output(
            job["cmd"], stderr=subprocess.STDOUT, text=True
        )
        print(f"Output: {output}")

        # Write the output to a file
        with open(job["output"], "w") as f:
            f.write(output)

    except subprocess.CalledProcessError as e:
        # Handle failure of the job
        print(f"ETL job failed with error: {e}")
        continue

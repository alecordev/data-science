"""
    pip install "prefect>=1,<2"
"""

import subprocess

from prefect import task, Flow


@task
def run_process(command):
    process = subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Error running command {command}. stderr: {stderr}")
    return stdout.decode()


@task
def collect_data(process1_output, process2_output):
    # process the output and return the collected data
    processed_data = process1_output + process2_output
    return processed_data


with Flow(name="subprocesses") as f:
    process1_output = run_process("python process1.py")
    process2_output = run_process("python process2.py")
    collected_data = collect_data(process1_output, process2_output)


if __name__ == "__main__":
    f.run()

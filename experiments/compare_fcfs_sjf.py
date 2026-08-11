from src.request import Request
from src.simulator import Simulator
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.sjf import SJFScheduler


def build_requests():
    return [
        Request(request_id=0, arrival_time=0, output_len=10),
        Request(request_id=1, arrival_time=1, output_len=100),
        Request(request_id=2, arrival_time=2, output_len=20),
        Request(request_id=3, arrival_time=3, output_len=50),
    ]


def run_experiment(scheduler):
    requests = build_requests()

    simulator = Simulator(
        requests=requests,
        scheduler=scheduler,
        decode_time_per_step=1.0,
    )

    simulator.run()

    avg_waiting = sum(r.waiting_time for r in requests) / len(requests)
    avg_response = sum(r.response_time for r in requests) / len(requests)

    return avg_waiting, avg_response


fcfs_wait, fcfs_resp = run_experiment(
    FCFSScheduler(max_batch_size=1)
)

sjf_wait, sjf_resp = run_experiment(
    SJFScheduler(max_batch_size=1)
)

print("FCFS")
print("Average waiting:", fcfs_wait)
print("Average response:", fcfs_resp)

print()

print("Oracle SJF")
print("Average waiting:", sjf_wait)
print("Average response:", sjf_resp)
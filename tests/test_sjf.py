from src.request import Request
from src.schedulers.sjf import SJFScheduler
from src.simulator import Simulator


def test_sjf_selects_shortest_job_first():
    scheduler = SJFScheduler(max_batch_size=1)

    requests = [
        Request(request_id=1, arrival_time=0, output_len=100),
        Request(request_id=2, arrival_time=0, output_len=20),
        Request(request_id=3, arrival_time=0, output_len=50),
    ]

    for request in requests:
        scheduler.add_request(request)

    admitted = scheduler.schedule(current_time=0)

    assert len(admitted) == 1
    assert admitted[0].request_id == 2

def test_sjf_with_simulator():
    requests = [
        Request(request_id=1, arrival_time=0, output_len=100),
        Request(request_id=2, arrival_time=0, output_len=20),
        Request(request_id=3, arrival_time=0, output_len=50),
    ]

    scheduler = SJFScheduler(max_batch_size=1)

    simulator = Simulator(
        requests=requests,
        scheduler=scheduler,
        decode_time_per_step=1.0,
    )

    simulator.run()

    for request in requests:
        print(
            f"Request {request.request_id}: "
            f"length={request.output_len}, "
            f"start={request.start_time}, "
            f"finish={request.finish_time}, "
            f"waiting={request.waiting_time}, "
            f"response={request.response_time}"
        )    
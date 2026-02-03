import argparse
import shlex
import subprocess
from distutils.util import strtobool

navi_robots = ['Ant', 'Car', 'Doggo', 'Point', 'Racecar']
navi_tasks = ['Button', 'Circle', 'Goal', 'Push']
diffculies = ['1', '2']
vel_robots = ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Swimmer', 'Humanoid']
vel_tasks = ['Velocity']

navi_envs = [
    f"Safety{robot}{task}{diffculty}-v0"
    for diffculty in diffculies
    for robot in navi_robots
    for task in navi_tasks
]

vel_envs = [
    f"Safety{robot}{task}-v1"
    for robot in vel_robots
    for task in vel_tasks
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=navi_envs+vel_envs,
        help="the ids of the environment to benchmark",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=[
            "pcpo",
            "ppo",
            "ppo_lag",
            "cup",
            "focops",
            "rcpo",
            "trpo_lag",
            "cpo",
            "cppo_pid"
        ],
        help="the ids of the algorithm to benchmark",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="the number of random seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="the number of the starting seed"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="the number of workers to run benchmark experimenets",
    )
    parser.add_argument(
        "--experiment", type=str, default="benchmark", help="name of the experiment"
    )
    parser.add_argument(
        "--total-steps", type=int, default=10000000, help="total number of steps"
    )
    parser.add_argument(
        "--num-envs", type=int, default=10, help="number of environments to run in parallel"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=20000, help="number of steps per epoch"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device to run the model on"
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="The device id to run the model on"
    )
    parser.add_argument(
        "--use-eval", type=lambda x: bool(strtobool(x)), default=False, help="Use evaluation environment for testing"
    )
    parser.add_argument(
        "--write-terminal", type=lambda x: bool(strtobool(x)), default=False, help="Write output to terminal"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta hyperparameter for cost improvement in SB-TRPO. If not set, the in-code default is used.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Cost limit for the baselines. If not set, the default (0.0) is used.",
    )
    args = parser.parse_args()

    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()

    commands = []
    for seed in range(0, args.num_seeds):
        for task in args.tasks:
            if "Doggo" in task:
                args.total_steps = str(100000000)
                args.steps_per_epoch = str(200000)
                args.num_envs = str(20)
            for algo in args.algo:
                commands += [
                    " ".join(
                        [
                            f"python {algo}.py",
                            "--task",
                            task,
                            "--seed",
                            str(args.start_seed + 1000*seed),
                            "--write-terminal",
                            str(args.write_terminal),
                            "--experiment",
                            args.experiment,
                            "--total-steps",
                            str(args.total_steps),
                            "--num-envs",
                            str(args.num_envs),
                            "--steps-per-epoch",
                            str(args.steps_per_epoch),
                            "--use-eval",
                            str(args.use_eval),
                            "--device",
                            str(args.device),
                            "--device-id",
                            str(args.device_id),
                            "--cost-limit" if args.cost_limit is not None else "",
                            str(args.cost_limit) if args.cost_limit is not None else "",
                            "--beta" if args.beta is not None else "",
                            str(args.beta) if args.beta is not None else "",
                        ]
                    )
                ]


    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="safepo-benchmark-worker-"
        )
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print(
            "not running the experiments because --workers is set to 0; just printing the commands to run"
        )

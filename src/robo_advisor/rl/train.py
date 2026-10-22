"""PPO 학습 명령행 진입점."""

import argparse
import logging
import sys

from robo_advisor.rl.progress import LOGGER
from robo_advisor.rl.training import load_training_config, run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description="재현 가능한 PPO 학습 실행")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="최소 학습 스텝 제한을 우회하고 기록")
    parser.add_argument(
        "--log-interval", type=float, default=30.0, help="진행률 로그 간격(초), 기본 30"
    )
    parser.add_argument("--quiet", action="store_true", help="콘솔에는 경고와 오류만 출력")
    args = parser.parse_args(argv)
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    console.setLevel(logging.WARNING if args.quiet else logging.INFO)
    LOGGER.addHandler(console)
    LOGGER.setLevel(logging.INFO)
    try:
        config = load_training_config(
            args.config,
            dataset_dir=args.dataset_dir,
            device=args.device,
            total_timesteps=args.total_timesteps,
            smoke=args.smoke,
        )
        result = run_training(
            config, run_id=args.run_id, smoke=args.smoke, log_interval=args.log_interval
        )
    finally:
        LOGGER.removeHandler(console)
    print(result.manifest_path)


if __name__ == "__main__":
    main()

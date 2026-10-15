"""PPO 학습 명령행 진입점."""

import argparse

from robo_advisor.rl.training import load_training_config, run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description="재현 가능한 PPO 학습 실행")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="최소 학습 스텝 제한을 우회하고 기록")
    args = parser.parse_args(argv)
    config = load_training_config(
        args.config,
        dataset_dir=args.dataset_dir,
        device=args.device,
        total_timesteps=args.total_timesteps,
        smoke=args.smoke,
    )
    result = run_training(config, run_id=args.run_id, smoke=args.smoke)
    print(result.manifest_path)


if __name__ == "__main__":
    main()

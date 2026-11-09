import argparse
from types import SimpleNamespace

from code.Imputation import impute
from code.config.dataset_profiles import DATASET_CATEGORIES
from code.core.data_io import build_run_stem
from code.core.path_config import add_workspace_path_arguments, apply_workspace_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Run one imputation task in a dedicated subprocess")
    add_workspace_path_arguments(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_category", choices=DATASET_CATEGORIES, required=True)
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_percent", type=float, required=True)
    parser.add_argument("--max_missing_lenth", type=int, required=True)
    parser.add_argument("--missing_mechanism", choices=["MCAR", "MAR", "MNAR"], required=True)
    parser.add_argument("--mar_logistic_beta", type=float, default=1.0)
    parser.add_argument("--mar_logistic_clip", type=float, default=3.0)
    parser.add_argument("--method", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_workspace_paths(args)
    run_args = SimpleNamespace(
        workspace_dir=args.workspace_dir,
        dataset=args.dataset,
        dataset_category=args.dataset_category,
        iter=args.iter,
        seed=args.seed,
        missing_percent=args.missing_percent,
        max_missing_lenth=args.max_missing_lenth,
        missing_mechanism=args.missing_mechanism,
        mar_logistic_beta=args.mar_logistic_beta,
        mar_logistic_clip=args.mar_logistic_clip,
        ground_truth_data_dir=args.ground_truth_data_dir,
        real_missing_data_dir=args.real_missing_data_dir,
        real_missing_truth_dir=args.real_missing_truth_dir,
        real_missing_label_col=args.real_missing_label_col,
        missing_data_dir=args.missing_data_dir,
        clean_data_dir=args.clean_data_dir,
        result_dir=args.result_dir,
    )
    dataset_name = build_run_stem(
        args.dataset,
        args.missing_percent,
        args.max_missing_lenth,
        args.missing_mechanism,
        args.iter,
        seed=args.seed,
    )
    impute(run_args, args.method, dataset_name, args.dataset)


if __name__ == "__main__":
    main()

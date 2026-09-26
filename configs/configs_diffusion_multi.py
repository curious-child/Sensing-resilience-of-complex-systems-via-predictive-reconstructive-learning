import argparse


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='modelGym')

    parser.add_argument('--cfg', default='configs/grid_search/AE_recon_pred_model.yaml', type=str,
                        help='The configuration file path.')
    parser.add_argument('--train_mode', default="grid", type=str,
                        help=" train mode: grid,hold_out,cross_val")
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')


    return parser.parse_args()
#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with wandb.init(job_type="basic_cleaning") as run:
        # Update config
        run.config.update(args)
        # Download input artifact. This will also log that this script is using this
        # particular version of the artifact
        logger.info(f'Download data: {args.input_artifact}')
        artifact_local_path = run.use_artifact(args.input_artifact).file()
        df = pd.read_csv(artifact_local_path)
        # Clean data
        logger.info('Perform data cleaning')
        # Filter price outliers out
        clean_df = (df
                    .loc[df.price.between(args.min_price, args.max_price)]
                    .copy())
        # Cast last_review col as datetime
        clean_df['last_review'] = pd.to_datetime(clean_df.last_review)
        # Save clean_sample to disk
        clean_df.to_csv('clean_df.csv', index=False)
        # Log artifact to W&B
        logger.info(f'Log artifact {args.output_artifact} to W&B')
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description)
        artifact.add_file('clean_df.csv')
        run.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Rows with price below min_price are filtered out",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Rows with price above max_price are filtered out",
        required=True
    )

    args = parser.parse_args()
    go(args)

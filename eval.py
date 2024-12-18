
import os
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset
from datasets.teach_dataset import TeachDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)


if args.use_kd:
    test_ds = TeachDataset(args.train_set_folder)
    test.test_for_teach(args, test_ds, model)

    args.val_set_folder = os.path.join(args.dataset_folder, "val")
    if not os.path.exists(args.val_set_folder):
        raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
    
    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)

    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"{val_ds}: {recalls_str}")

else:

    if args.dataset_folder.split("/")[-3] == "sf_xl":

        test_ds = TestDataset(args.test_set_folder, queries_folder="queries_night",
                            positive_dist_threshold=args.positive_dist_threshold)

        recalls, recalls_str = test.test(args, test_ds, model)
        logging.info(f"{test_ds}: {recalls_str}")

    elif args.dataset_folder.split("/")[-3] == "tokyo247":

        test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                            positive_dist_threshold=args.positive_dist_threshold)

        recalls, recalls_str, recalls_day, recalls_sunset, recalls_night = test.test_tokyo(args, test_ds, model)

        logging.info(f"Recalls on {test_ds}: {recalls_str}")
        logging.info(f"Recalls on {test_ds}: {recalls_day}")
        logging.info(f"Recalls on {test_ds}: {recalls_sunset}")
        logging.info(f"Recalls on {test_ds}: {recalls_night}")


    elif args.dataset_folder.split("/")[-3] == "svox":
        test_ds = TestDataset(args.test_set_folder, queries_folder="queries_night",
                            positive_dist_threshold=args.positive_dist_threshold)

        recalls, recalls_str = test.test(args, test_ds, model)
        logging.info(f"{test_ds}: {recalls_str}")

    elif args.dataset_folder.split("/")[-3] == "msls":
        test_ds = TestDataset(args.test_set_folder, queries_folder="queries_night",
                            positive_dist_threshold=args.positive_dist_threshold)

        recalls, recalls_str = test.test(args, test_ds, model)
        logging.info(f"{test_ds}: {recalls_str}")




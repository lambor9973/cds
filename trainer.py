import sys
import logging
import copy
import torch
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def CDS_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    _reset_peak_cuda_stats(args["device"])


    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)

    print()
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()
     
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    _log_peak_cuda_stats(args["device"])


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def _format_bytes(num_bytes):
    if num_bytes is None:
        return "N/A"
    num_bytes = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if num_bytes < 1024 or unit == units[-1]:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def _get_cuda_index(dev):
    if isinstance(dev, torch.device):
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else torch.cuda.current_device()
    if isinstance(dev, int):
        return dev if dev >= 0 else None
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return None


def _reset_peak_cuda_stats(devices):
    if not torch.cuda.is_available():
        return
    for dev in devices:
        dev_index = _get_cuda_index(dev)
        if dev_index is None:
            continue
        try:
            torch.cuda.reset_peak_memory_stats(dev_index)
        except Exception as exc:
            logging.warning("Failed to reset peak memory stats on cuda:%s (%s)", dev_index, exc)


def _log_peak_cuda_stats(devices):
    if not torch.cuda.is_available():
        print("Peak GPU memory: N/A (CUDA not available)")
        logging.info("Peak GPU memory: N/A (CUDA not available)")
        return
    for dev in devices:
        dev_index = _get_cuda_index(dev)
        if dev_index is None:
            continue
        try:
            torch.cuda.synchronize(dev_index)
            allocated = torch.cuda.max_memory_allocated(dev_index)
            reserved = torch.cuda.max_memory_reserved(dev_index)
        except Exception as exc:
            logging.warning("Failed to query peak memory on cuda:%s (%s)", dev_index, exc)
            continue
        msg = (
            f"Peak GPU memory on cuda:{dev_index}: "
            f"allocated={_format_bytes(allocated)}, "
            f"reserved={_format_bytes(reserved)}"
        )
        print(msg)
        logging.info(msg)

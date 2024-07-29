import argparse
import os
import pickle

import torch.nn as nn
from typing import Dict, List

from data.preprocessing import MyDataset
from utils.evaluation_utils import evaluate, load_network
from utils.torch_utils import fix_random, get_device


class InferencePipeline:

    def __init__(self, device) -> None:

        self.device = device

    def __call__(
        self, **args
    ):  # data: str, model_args, args=None, batch_size=8) -> Any:
        
        outputs = self.prepare(args)

        if outputs is None:
            return
        
        self.inference(*outputs)

    def prepare(self, **args):

        argskeys = list(args.keys()) 
        msg = []

        for key in ['data','model','model_args']:
            if key not in argskeys:
                msg.append(f"{key} not provided")

        if len(msg) > 0:
            raise Exception(", ".join(msg))

        mod_args_path = args['model_args']
        if not os.path.exists(mod_args_path):
            raise Exception("model args file path doesn't exist")
        
        with open(mod_args_path, 'rb') as f:
            model_args = pickle.load(f)

        model_ckpt = args['model']
        model = None
        if isinstance(model_ckpt, str): 
            # must be a ckpt path
            if not os.path.exists(model_ckpt):
                raise Exception("ckpt path doesn't exist")

            model = load_network(model_ckpt, model_args)
        elif isinstance(model, nn.Module):
            model = model_ckpt
        else:
            print("Model or ckpt path not provided")
            return
        
        data_in = args['data']

        # TODO check if file fasta or string
        #

        if isinstance(data_in, (List[str],str)):

            dataset = MyDataset(data_in)
            loader = dataset.get_data_loaders(
                model_name=model_args.esm_name, batch_size=8
            )
        else:
            loader = data_in

        return loader, model

    def inference(self, loader, model) -> Dict:
        
        criterion = nn.CrossEntropyLoss

        outputs = evaluate(model, loader, self.device, criterion)

        return outputs

if __name__ == "__main__":
    # options for using from the command line

    parser = argparse.ArgumentParser(description='My parser')#
    # input filename ifpath
    parser.add_argument(
        "-s", "--sequence", dest="data", default=None, help="File path fasta or string of sequences"
        )
    parser.add_argument(
        "-c", "--ckptpath", dest="fpath", default=None, help="File path for checkpoint"
        )
    parser.add_argument(
        "-m", "--modelargs", dest="model_args", default=None, help="File path for model init arguments"
        )
    parser.add_argument(
        "-o", "--odname", dest="odname", default="results", help="Output directory name relative or absolute"
        )
    parser.add_argument(
        "--device", dest="device", type=str, default='auto', choices=['auto','cpu','cuda','mps'], help="device"
        )

    parser.add_argument(
        "-b","--batchsize", dest="batch_size", type=int, default=8, help="batch size"
        )

    options = parser.parse_args()

    seed = options.seed
    fix_random(seed)

    device = get_device() if options.device == 'auto' else options.device

    pipeline = InferencePipeline(seed=seed, mode='train', device=device)
    pipeline(
        data=options.data, model=options.fpath, model_args=options.model_args
        )
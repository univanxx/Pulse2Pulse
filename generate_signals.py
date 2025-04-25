import numpy as np
import torch
from tqdm import tqdm
import argparse
import os


def process_batches(args, num, label, count):
    """Splits values into batches and calls generate() on each batch."""
    step = 0
    for count_i, start in enumerate(tqdm(range(0, count, args.batch_size))):
        count_i = min(args.batch_size, count - start)
        res = generate(args, label, count_i)
        np.save(os.path.join(args.save_path,  f"label_{num}_sample_{step}.npy"), res)
        step += 1

def generate(args, label, num_samples):
    cond = torch.stack([torch.tensor(label)] * num_samples).cuda().float()
    if args.model_name == "p2p":
        noise = torch.Tensor(num_samples, 8, 5000).uniform_(-1, 1).cuda()
    else:
        noise = torch.Tensor(num_samples, 100).uniform_(-1, 1).cuda()

    with torch.no_grad():
        fake = net(noise, cond.cuda()).cpu().numpy()
    return fake


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Pulse2Pulse and WaveGAN* signals generation')
    # experimental results
    parser.add_argument('--model_name', required=True, choices=['p2p', 'wg*'])  # Pulse2Pulse and WaveGAN*
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device_id', type=str, default="cpu")
    parser.add_argument('--labels_path', type=str, required=True,
                        help='path to PTBXL labels')
    parser.add_argument('--task_type', required=True, choices=['imbalance', 'addition'])
    parser.add_argument('--save_path', required=True, help='path to save generated samples by models')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, args.model_name, args.task_type)
    os.makedirs(args.save_path, exist_ok=False)

    labels = np.load(args.labels_path)[np.load(os.path.dirname(args.labels_path)+"/thirdparty/train_ids.npy")]
    
    if args.model_name == "p2p":
        from models.pulse2pulse import WaveGANGenerator as Pulse2PuseGenerator
        net = Pulse2PuseGenerator(labels.shape[1], model_size=50, ngpus=1, upsample=True)
    elif args.model_name == "wg*":
        from models.wavegan_star import WaveGANGenerator as WaveGANStarGenerator
        net = WaveGANStarGenerator(labels.shape[1], model_size=50, ngpus=1, upsample=True)
    else:
        raise NotImplementedError
    
    if args.task_type == "imbalance":
        count = np.max(labels.sum(axis=0)) - labels.sum(axis=0)[1:]  # excluding 426783006 as the biggest class which doesn't need to be upsampled
        labels = []
        for i in range(1, 9):
            tmp = np.zeros(9)
            tmp[i] = 1
            labels.append(tmp)
    elif args.task_type == "addition":
        labels, count = np.unique(labels, axis=0, return_counts=True)
        count *= 2
    else:
        raise NotImplementedError

    net_dict = torch.load(args.checkpoint_path, map_location=args.device_id)["netG_state_dict"]
    net.load_state_dict(net_dict)
    net.cuda()
    net.eval()

    for i, (label_i, count_i) in enumerate(tqdm(zip(labels, count))):
        if int(count_i) != 0:
            process_batches(args, i, label_i, count_i)
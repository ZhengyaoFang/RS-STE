import os
import torch
import importlib
import argparse
import albumentations
import math
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from data.dataset import InferenceDataset
from main import instantiate_from_config, get_parser, get_obj_from_str

def crop_and_resize(x, size, ori_size, k):
    img = x.detach().cpu()
    img = torch.clamp(img, -1., 1.)
    img = (img + 1.0) / 2.0
    img = img.permute(0, 2, 3, 1).numpy()[k]
    img = (img * 255).astype(np.uint8)
    return img


def main(args):
    decoder_config = OmegaConf.load(args.vqgan_config)
    config = OmegaConf.load(args.transformer_config)
    # The full model checkpoint (--resume, e.g. weights/model.pth) already
    # contains the decoder weights (stored under vqgan.* keys), so the
    # standalone Stage 1 autoencoder checkpoint (weights/vae.ckpt) is not
    # needed for inference. Clear it to avoid a spurious file-not-found error.
    if "params" in decoder_config.model:
        decoder_config.model.params.ckpt_path = None
    config.model.params.decoder_config = decoder_config.model
    config.model.params.ckpt_path = args.resume
    model = instantiate_from_config(config.model).to('cuda')
    model.eval()

    dataset = InferenceDataset(config.data.params.validation.params.size, args.inference_anno)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            img1 = batch["image1"].to('cuda')
            img1 = img1.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if img1.dtype == torch.double:
                img1 = img1.float()
            img1_quant_latent_ = model.conv(img1).flatten(2).permute(0, 2, 1)
            rec2_indices,_ = model.str_converter.encode(batch["rec2"])
            rec2_indices = rec2_indices.to(img1_quant_latent_.device)

            rec2_embd = model.str_embd(rec2_indices) # [bs, 8*32, n_embd]
            rec1_mask = model.masked_rec_embd.weight[0][None, None, :].expand(rec2_indices.shape[0], rec2_indices.shape[1], -1)
            img2_mask = model.masked_img_imbd.weight[0][None, None, :].expand(img1_quant_latent_.shape[0], img1_quant_latent_.shape[1], -1)

            inputs = torch.cat([rec2_embd, img1_quant_latent_, rec1_mask, img2_mask], dim=1) # [8, 576, 768]
            embeddings, logits = model.transformer(inputs)
            rec1_indices = torch.topk(logits[:,288:320,:], k=1, dim=-1)[1].view(batch["image1"].shape[0],-1)
            pred_rec = model.str_converter.decode(rec1_indices)
            img2_rec_quant_latent = model.conv_o(embeddings[:,320:576,:].permute(0, 2, 1).contiguous().view(embeddings.shape[0], -1, 8, 32))
            img2_rec = model.decoder.decode(img2_rec_quant_latent)

            for k in range(img2_rec.shape[0]):
                edited_img = crop_and_resize(img2_rec, batch["image1_size"], batch['ori_size'], k)
                target_file = os.path.join(args.target_path, batch["img_name"][k])
                Image.fromarray(edited_img).save(target_file)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inference_anno", type=str, default="data/annotation/inference_annotations.pkl")
    parser.add_argument("--target_path", type=str,  default=f"output/inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)
    print(f"Saving results to {args.target_path}")
    main(args)


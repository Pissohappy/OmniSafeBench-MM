import types
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import abc
import transformers
import json
import argparse
import os
import re
from PIL import Image
from tqdm import tqdm
import time
import base64
from pathlib import Path
from torchvision import transforms
import random
import io
import openai
from openai import OpenAI
import numpy as np
import pandas as pd
from math import ceil
import torch

try:
    # Try relative import (preferred)
    from .cider_models.diffusion_denoiser.imagenet.DRM import DiffusionRobustModel  # type: ignore
except ImportError:
    # If relative import fails, try absolute import
    from defenses.cider_models.diffusion_denoiser.imagenet.DRM import DiffusionRobustModel  # type: ignore


class QApair:
    """
    containing all informations for a query-answer process, including:
    - query
    - image path
    - refuse: bool, if is True(harmful image), VLM should directly refuse to answer.
    - answer
    - behaviour: label in Harmbench, will be used in scoring process.
    """

    def __init__(self, query, imgpth, refuse, ans, behav) -> None:
        self.query = query
        self.imgpth = imgpth
        self.refuse = refuse
        self.ans = ans
        self.behav = behav


class Defender:
    """
    given text and image, if any decreace of cosine similarity greater than threshold
    (the delta values are smaller than threshold) occurs during denoise,
    then consider it as adversarial
    contains methods:
        1. train: calculate threshold with given clean image data and specified ratio.
        2. predict: given img-text pair, return whether it is adversarial.
        3. get_confusion_matrix: given validation dataset, call predict function and return statistics (i.e. confusion matrix)
    """

    def __init__(self, threshold=None):
        self.threshold = threshold
        print(f"Defender initialized with threshold={self.threshold}")

    def train(self, data, ratio=0.9):
        """
        calculate threshold with given data and specified conservative ratio
        :data: cosine similarity between malicious query(text) & clean image
            cols: denoise times
            rows: different text
        :ratio: the ratio of clean image cosine similarity decrease value that should be lower than threshold
        """
        if type(data) == list:
            for df in data:
                names = [i for i in range(df.shape[1])]
                df.columns = names
            data = pd.concat(data, axis=0, ignore_index=True)
        # get the first col (origin image) as the base cosine similarity
        base = data.iloc[:, 0]
        # get the decrease value
        decrease = data.iloc[:, 1:].sub(base, axis=0)
        # get the ratio of decrease value that is lower than threshold
        self.threshold = np.percentile(decrease, (1 - ratio) * 100)
        print(f"Threshold updated to {self.threshold}")

    def predict(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        by calculating the interpolation between each and the first one

        :cossims: a nD array of cosine similarity (m*n)
            cols: denoise times n
            row: m text
        return: a nD array of boolean (m*1)
            True means adversarial
        """
        if len(cossims.shape) == 1:
            return True in (np.array(cossims[1:]) - cossims[0] < self.threshold)
        else:
            ret = []
            for r in range(cossims.shape[0]):
                row = cossims.iloc[r]
                ret.append(
                    True in (np.array(row.iloc[1:]) - row.iloc[0] < self.threshold)
                )
            return ret

    def get_lowest_idx(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        for adversarial data, return the index of lowest cosine similarity iff
        the decline value is more than threshold τ, otherwise return 0

        :cossims: a nD array of cosine similarity (1*n)
            cols: denoise times n
            row: 1 text
        return: a nD array of int (1*1)
            0 is clean, positive int is the index of lowest cosine similarity
        """
        # reshape the data
        cossims = np.array(cossims).reshape(1, -1)
        delta = cossims - cossims[0, 0]
        if np.min(delta) < self.threshold:
            return np.argmin(delta)
        else:
            return 0

    def get_confusion_matrix(self, datapath: str, checkpt_num=8, group=True):
        """test the defender with given cosine similarity data,
        save the statics to savepath\n
        only consider malicious text\n
        the result contains 4 rows: for each adversarial image,
        consider it as positive and clean as negative,
        output the results
        checkpt_num: the maximum number of denoise times checkpoint to consider
        group: if true, output separate matrix for different adv image"""
        df = pd.read_csv(datapath)
        try:
            df = df[df["is_malicious"] == 1]  # only consider malicious text input
        except:
            pass
        results = {
            "constraint": [],
            "accuracy": [],
            "recall(dsr)": [],
            "precision": [],
            "f1": [],
            "classification threshold": [],
            "fpr": [],
        }
        fp, tn = 0, 0
        # get the Test Set clean image data for prediction
        all_clean_header = [col for col in df.columns if "clean_" in col]
        clean_classes_names = set(
            ["_".join(h.split("_")[:2]) for h in all_clean_header]
        )
        for clean_class in clean_classes_names:
            clean_header = [col for col in df.columns if clean_class + "_" in col]
            if len(clean_header) > checkpt_num:
                clean_header = clean_header[:checkpt_num]
            clean_data = df[clean_header]
            # predict with clean image
            clean_predict = np.array(self.predict(clean_data))
            fp += sum(clean_predict[:])
            tn += sum(~clean_predict[:])

        tot_tp, tot_fn = 0, 0
        # get the adversarial image data
        all_adv_header = [col for col in df.columns if "prompt_" in col]
        adv_classes_names = set(["_".join(h.split("_")[:3]) for h in all_adv_header])
        for adv_class in adv_classes_names:
            # list the headers of adv_class constraint
            adv_header = [col for col in df.columns if adv_class + "_" in col]
            if len(adv_header) > checkpt_num:
                adv_header = adv_header[:checkpt_num]
            # get data
            adv_data = df[adv_header]
            # predict
            adv_predict = np.array(self.predict(adv_data))
            tp = sum(adv_predict[:])
            fn = sum(~adv_predict[:])
            tot_tp += tp
            tot_fn += fn

            if not group:  # calculate together later
                continue
            # if group, calculate matrix for the class now
            # check num positive = negative
            assert tp + fn == fp + tn

            acc = (tp + tn) / (tp + fn + fp + tn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            fpr = fp / (fp + tn)
            results["constraint"].append(adv_class.split("_")[-1])
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)
        if not group:
            # assert tot_tp+tot_fn==fp+tn
            tp = tot_tp
            fn = tot_fn
            acc = (tp + tn) / (tp + fn + fp + tn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            fpr = fp / (fp + tn)
            results["constraint"].append("all")
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)
        results = pd.DataFrame(results)
        return results


class Encoder:
    model_path = None

    def __init__(self, mdpth) -> types.NoneType:
        self.model_path = mdpth

    @staticmethod
    def compute_cosine(a_vec: np.ndarray, b_vec: np.ndarray):
        """calculate cosine similarity"""
        norms1 = np.linalg.norm(a_vec, axis=1)
        norms2 = np.linalg.norm(b_vec, axis=1)
        dot_products = np.sum(a_vec * b_vec, axis=1)
        cos_similarities = dot_products / (norms1 * norms2)  # ndarray with size=1
        return cos_similarities[0]

    @abc.abstractmethod
    def calc_cossim(self, pairs: list[tuple[str, str]]):
        """input list of (query, img path) pairs,
        output list of cosin similarities"""
        res = []
        for p in pairs:
            text_embed = self.embed_text(p[0])
            img_embed = self.embed_img(p[1])
            cossim = self.compute_cosine(text_embed, img_embed)
            res.append(cossim)
        return res

    @abc.abstractmethod
    def embed_img(self, imgpth) -> np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self, text) -> np.ndarray:
        pass


class LlavaEncoder(Encoder):
    def __init__(self, mdpth, device="cuda:0") -> types.NoneType:
        super().__init__(mdpth)
        self.device = device
        self.model = AutoModelForPreTraining.from_pretrained(
            mdpth, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(mdpth)
        self.imgprocessor = AutoImageProcessor.from_pretrained(mdpth)

    def embed_img(self, imgpth) -> np.ndarray:
        image = Image.open(imgpth)
        # img embedding
        pixel_value = self.imgprocessor(image, return_tensors="pt").pixel_values.to(
            self.device
        )
        image_outputs = self.model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[
            self.model.config.vision_feature_layer
        ]
        selected_image_feature = selected_image_feature[:, 1:]  # by default
        image_features = self.model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu").numpy()
        return image_features

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds


# used for defense method CIDER
def generate_denoised_img(model="diffusion", **kwargs):
    import shutil

    try:
        shutil.rmtree(kwargs["save_dir"])
    except FileNotFoundError:
        print("No existing temp dir")
    finally:
        os.makedirs(kwargs["save_dir"], exist_ok=True)

    if model == "diffusion":
        func = generate_denoised_img_diffusion
    # elif model == "dncnn":
    #     func = generate_denoised_img_DnCNN
    # elif model == "nlm":
    #     func = generate_denoised_img_NLM
    else:
        raise RuntimeError(f"Unrecognised model type: {model}")
    return func(**kwargs)


# used for defense method CIDER
def generate_denoised_img_diffusion(
    imgpth: list[str],
    save_dir: str,
    cps: int,
    step=50,
    device: int = 0,
    batch_size=50,
    **kwargs,
):

    resized_imgs = []  # type: list[tuple[Image.Image,str]]
    denoised_pth = []
    imgpth.sort()

    # preprocess
    trans = transforms.Compose([transforms.Resize([224, 224])])
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        filename = os.path.split(filepath)[1]
        # resize to 224*224
        img = trans(img)
        resized_imgs.append((img, filename))
        # save the original image
        savename = os.path.splitext(filename)[0] + "_denoised_000times.jpg"
        img.save(os.path.join(save_dir, savename))
        denoised_pth.append(os.path.join(save_dir, savename))
    if cps <= 1:
        return denoised_pth

    model = DiffusionRobustModel()
    iterations = range(step, cps * step, step)
    b_num = ceil(len(resized_imgs) / batch_size)  # how man runs do we need
    for b in tqdm(range(b_num), desc="denoising batch"):
        l = b * batch_size
        r = (b + 1) * batch_size if b < b_num - 1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        for it in iterations:
            # project value between -1,1
            ary = [np.array(_[0], dtype=np.float32) / 255 * 2 - 1 for _ in part]
            ary = np.array(ary)
            ary = torch.tensor(ary).permute(0, 3, 1, 2).to(device)
            denoised_ary = np.array(model.denoise(ary, it).to("cpu"))
            denoised_ary = denoised_ary.transpose(0, 2, 3, 1)
            denoised_ary = (denoised_ary + 1) / 2 * 255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img = Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(part[i][1])[
                    0
                ] + "_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_dir, sn))
                denoised_pth.append(os.path.join(save_dir, sn))
    del model
    torch.cuda.empty_cache()
    return denoised_pth


def get_similarity_list(
    text_file,
    pair_mode,
    encoder_pth,
    imgdir="./temp/denoised_imgs",
    device="cuda:0",
    cpnum=8,
):

    if "llava" in encoder_pth.lower():
        encoder = LlavaEncoder(encoder_pth, device)
    else:
        raise ValueError(f"Unrecognised encoder Type from:{encoder_pth}")

    # load text inputs
    # with open(text_file) as fr:
    #     reader = csv.reader(fr)
    #     queries = [line[0] for line in reader if line[1]=="standard"]

    # with open(text_file, 'r', encoding='utf-8') as f:
    #     attack_cases = json.load(f)
    #     queries = [case['attack_prompt'] for case in attack_cases]

    with open(text_file, "r", encoding="utf-8") as f:
        attack_cases = json.load(f)
        # Handle both list and dictionary formats
        if isinstance(attack_cases, dict):
            # Convert dictionary to list of values
            attack_cases = list(attack_cases.values())
        elif isinstance(attack_cases, list):
            attack_cases = attack_cases
        queries = [case.get("attack_prompt", "") for case in attack_cases]

    # print(f"Loading {len(queries)} queries from {text_file}")
    # print(queries)
    # load image paths
    dir1 = sorted(os.listdir(imgdir))
    img_pths = [os.path.join(imgdir, img) for img in dir1]
    # compute cosine similarity between text and n denoised images
    # and form a table of size ((len(queries),image_num, ckpt_num)) or (len(text_embed_list), ckpt_num)
    image_num = len(img_pths) // cpnum
    if pair_mode == "combine":
        cossims = np.zeros((len(queries), image_num, cpnum))
        for i in range(len(queries)):
            for j in range(image_num):
                inputs = [
                    (queries[i], img_pths[k]) for k in range(j * cpnum, (j + 1) * cpnum)
                ]
                temp = encoder.calc_cossim(inputs)
                cossims[i, j] = temp
    else:  # injection
        cossims = np.zeros((len(queries), cpnum))
        for i in range(image_num):
            inputs = [
                (queries[i], img_pths[k]) for k in range(i * cpnum, (i + 1) * cpnum)
            ]
            temp = encoder.calc_cossim(inputs)
            cossims[i] = temp
    return cossims


def defence(
    imgpth: list[str], args, cpnum=8
) -> tuple[list[str] | list[list], list[bool] | list[list[bool]]]:
    # count the number of images in args.img
    image_num = len(imgpth)
    denoised_img_dir = os.path.join(args.tempdir, "denoised_imgs")

    if args.denoiser == "dncnn":
        cpnum = 2  # DnCNN currently only denoise once for each img.
    denoised_imgpth = generate_denoised_img(
        args.denoiser,
        imgpth=imgpth,
        save_dir=denoised_img_dir,
        cps=cpnum,
        batch_size=50,
        device=args.cuda,
        model_path=[args.denoiser_path],
    )
    denoised_imgpth.sort()

    # compute cosine similarity
    sim_matrix = get_similarity_list(
        args.text_file,
        args.pair_mode,
        imgdir=denoised_img_dir,
        cpnum=cpnum,
        device=args.cuda,
        encoder_pth=args.encoder_path,
    )

    # for each row, check with detector
    d_denoise = Defender(threshold=args.threshold)
    adv_idx = []
    refuse = []  # type:list|list[list]
    ret_images_pths = []
    if args.pair_mode == "combine":  # 3d-array, text,img,cpnum
        text_num = sim_matrix.shape[0]

        for i in range(text_num):
            adv_row, refuse_row, ret_img_row = [], [], []
            for j in range(image_num):
                low = d_denoise.get_lowest_idx(sim_matrix[i][j])
                adv_row.append(low)
                refuse_row.append(low != 0)
                # select the lowest img and return
                idx = j * cpnum + low
                ret_img_row.append(denoised_imgpth[idx])
            adv_idx.append(adv_row)
            refuse.append(refuse_row)
            ret_images_pths.append(ret_img_row)
    else:  # injeciton
        for i in range(sim_matrix.shape[0]):  # =|queries|=|imgs|
            # the defender will find idx of image with lowest cossim(with decrease over threshold)
            adv_idx.append(d_denoise.get_lowest_idx(sim_matrix[i]))
            # and once found, VLM should refuse to respond.
            refuse.append(adv_idx[i] != 0)
            # select the lowest img and return
            idx = i % image_num * cpnum + adv_idx[i]
            ret_images_pths.append(denoised_imgpth[idx])

    return (ret_images_pths, refuse)

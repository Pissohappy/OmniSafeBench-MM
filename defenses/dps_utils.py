import json
import os

from tqdm import tqdm
import time
from torchvision import transforms
import random
import numpy as np
from math import ceil


import base64
from pathlib import Path
from PIL import Image
import torch
import re


# used for defense method DPS
def crop_image(img_path, folder_path, case_idx, idx=None):
    """
    Modification notes:
    - Added a new required parameter `case_idx` to receive the attack sample ID.
    - Created a subfolder named tmp_dps under folder_path.
    - Filename changed to `{case_idx}_center_crop.jpg` and saved in tmp_dps folder.
    - The original idx parameter is unused in this function and remains unchanged.
    """
    img = Image.open(img_path)
    new_width = img.width // 2
    new_height = img.height // 2

    center_x, center_y = img.width // 2, img.height // 2

    left = center_x - new_width // 2
    top = center_y - new_height // 2
    right = center_x + new_width // 2
    bottom = center_y + new_height // 2

    center_img = img.crop((left, top, right, bottom))

    # --- Modification points ---
    # 1. Define and create tmp_dps subfolder path (thread-safe)
    tmp_dps_path = os.path.join(folder_path, "tmp_dps")
    os.makedirs(tmp_dps_path, exist_ok=True)

    # 2. Define unique filename
    unique_filename = f"{case_idx}_center_crop.jpg"

    # 3. Point final path to tmp_dps subfolder
    output_path = os.path.join(tmp_dps_path, unique_filename)

    center_img.save(output_path)
    return output_path


# used for defense method DPS
def random_crop(img_path, folder_path, case_idx, idx=None):
    """
    Modification notes:
    - Added a new required parameter `case_idx`.
    - Created a subfolder named tmp_dps under folder_path.
    - Filename changed to `{case_idx}_random_crop_{idx}.jpg` and saved in tmp_dps folder.
    - The original idx parameter position and function are fully preserved.
    - Fixed a small bug in original code where return statement was missing "/".
    """
    img = Image.open(img_path)
    W, H = img.size
    crop_ratio = random.uniform(1 / 4, 1 / 2)
    crop_width = int(W * crop_ratio)
    crop_height = int(H * crop_ratio)
    start_x = random.randint(0, W - crop_width)
    start_y = random.randint(0, H - crop_height)
    cropped_image = img.crop(
        (start_x, start_y, start_x + crop_width, start_y + crop_height)
    )

    # --- Modification points ---
    # 1. Define and create tmp_dps subfolder path (thread-safe)
    tmp_dps_path = os.path.join(folder_path, "tmp_dps")
    os.makedirs(tmp_dps_path, exist_ok=True)

    # 2. Define unique filename
    unique_filename = f"{case_idx}_random_crop_{idx}.jpg"

    # 3. Point final path to tmp_dps subfolder
    output_path = os.path.join(tmp_dps_path, unique_filename)

    cropped_image.save(output_path)
    return output_path


# used for defense method DPS
def auto_mask(model, img_path, folder_path, case_idx, idx=None):
    """
    Modification notes:
    - Added a new required parameter `case_idx`.
    - Created a subfolder named tmp_dps under folder_path.
    - Filename changed to `{case_idx}_auto_crop_{idx}.jpg` and saved in tmp_dps folder.
    - The original idx parameter position and function are fully preserved.
    """
    img = Image.open(img_path)

    seg = "Return the coordinates of the text box in the image with the following format: {'obj1': [left, top, right, bottom], 'obj2': ...}."
    notice = f"Notice that the original image shape is {[0, 0, img.size[0], img.size[1]]}. Your coordinates must not exceed the original size.\n"
    model.ask(notice + seg + "Direct give your answer:", img_path)

    text = model.memory_lst[-1]["content"]
    print("auto_mask_response:", text)
    pattern = r"\[(\d+), (\d+), (\d+), (\d+)\]"
    matches = re.findall(pattern, text)
    coords = [list(match) for match in matches]
    image_list = []
    for i in range(len(coords)):
        coords[i] = [
            max(0, int(coords[i][0])),
            max(0, int(coords[i][1])),
            min(img.size[0], int(coords[i][2])),
            min(img.size[1], int(coords[i][3])),
        ]

    auto_img = remove_areas(img, coords)

    # --- Modification points ---
    # 1. Define and create tmp_dps subfolder path (thread-safe)
    tmp_dps_path = os.path.join(folder_path, "tmp_dps")
    os.makedirs(tmp_dps_path, exist_ok=True)

    # 2. Define unique filename
    unique_filename = f"{case_idx}_auto_crop_{idx}.jpg"

    # 3. Point final path to tmp_dps subfolder
    output_path = os.path.join(tmp_dps_path, unique_filename)

    auto_img.save(output_path)
    image_list.append(output_path)

    return image_list


# used for defense method DPS
def remove_areas(img, discard_boxes):
    img_array = np.array(img)

    for discard_box in discard_boxes:
        left, top, right, bottom = discard_box

        img_array[top:bottom, left:right] = 0
    new_img = Image.fromarray(img_array)
    return new_img


# used for defense method DPS
class GPTAgent:
    def __init__(self, model, sleep_time: float = 1) -> None:
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.name = None
        self._init_system_prompt()
        self.client = model

    # base 64 encoding
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def set_system_prompt(self, system_prompt: str):
        self._init_system_prompt()
        # Note: GPTAgent's original set_system_prompt implementation may need adjustment to match its prompt structure
        if self.memory_lst and self.memory_lst[0]["role"] == "system":
            self.memory_lst[0]["content"] = system_prompt
        else:
            # Fallback or error handling
            self.memory_lst.insert(0, {"role": "system", "content": system_prompt})

    def ask(self, instruction: str, image_file=None):
        return self._query(instruction=instruction, image_file=image_file)  # query

    def set_name(self, name):
        self.name = name

    def _init_system_prompt(self):
        self.memory_lst.clear()
        try:
            with open("./qwen_meta.json", "r") as file:
                self.memory_lst = json.load(file)
        except FileNotFoundError:
            self.memory_lst = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

    def _add_vlm_event(self, text_prompt, image_file):
        if image_file:  # first query
            base64_image = self._encode_image(image_file)
            self.image = base64_image
            event = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        else:
            event = {"role": "user", "content": [{"type": "text", "text": text_prompt}]}
        self.memory_lst.append(event)

    def _query(self, instruction, image_file=None) -> str:
        time.sleep(1)
        # call gpt api
        self._add_vlm_event(instruction, image_file)

        completion = self.client.generate(messages=self.memory_lst, max_tokens=2000)

        # Handle different return types
        if isinstance(completion, str):
            # If return is string, use directly
            response_content = completion
        elif hasattr(completion, "choices"):
            # If return is OpenAI-style response object
            try:
                if len(completion.choices) > 0:
                    choice = completion.choices[0]
                    if hasattr(choice, "message"):
                        if hasattr(choice.message, "model_dump"):
                            # Use model_dump() method (Pydantic model)
                            response_dict = choice.message.model_dump()
                            response_content = response_dict.get("content", "")
                        elif hasattr(choice.message, "content"):
                            # Directly access content attribute
                            response_content = choice.message.content
                        else:
                            response_content = str(choice.message)
                    else:
                        response_content = str(choice)
                else:
                    response_content = str(completion)
            except (AttributeError, IndexError, KeyError) as e:
                # If attribute access fails, convert to string
                response_content = str(completion)
        else:
            # Other cases, convert to string
            response_content = str(completion)

        # Add response to history
        response_dict = {"role": "assistant", "content": response_content}
        self.memory_lst.append(response_dict)

        return response_content

    def get_system_prompt(self):
        return self.memory_lst[0]["content"]

    def _add_response_history(self, content):
        response = {"role": "assistant", "content": content}
        self.memory_lst.append(response)

    def clear_history(self):
        self._init_system_prompt()


# used for defense method DPS


class JudgeAgent:
    # MODIFIED: __init__ now accepts 'api' and an optional 'base_url'.
    def __init__(self, client, name: str = "judge", sleep_time: float = 1) -> None:
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.name = name
        self._init_system_prompt()
        # MODIFIED: The OpenAI client is now initialized with the provided api and base_url.
        self.client = client

    def set_system_prompt(self, system_prompt: str):
        self._init_system_prompt()
        self.memory_lst[0]["content"] = system_prompt

    def ask(self, instruction: str):
        return self._query(instruction=instruction)  # query

    def set_name(self, name):
        self.name = name

    def _init_system_prompt(self):
        self.memory_lst.clear()
        # Added a try-except block for robustness, in case the file is missing.
        self.memory_lst.append(
            {"role": "system", "content": "You are a helpful assistant."}
        )

    def _add_llm_event(self, instruction):
        event = {"role": "user", "content": instruction}
        self.memory_lst.append(event)

    def _query(self, instruction):
        self._add_llm_event(instruction)
        if self.name == "judge":
            model = "gpt-4o"
        elif self.name == "moderator":
            model = "gpt-4o-mini"
        # print(f"Using model: {model} for JudgeAgent")

        try:
            response = self.client.generate(model=model, messages=self.memory_lst)
            self.memory_lst.append(response.choices[0].message.model_dump())
            # print(response.choices[0].message.model_dump())
        except Exception as e:
            print(f"Error in JudgeAgent query: {e}")
            self.memory_lst.append(
                {
                    "role": "assistant",
                    "content": "Sorry, I can not assist with that due to an API error.",
                }
            )

    def clear_history(self):
        self._init_system_prompt()


# used for defense method DPS
def init_crop_agent(model):
    agent = GPTAgent(model)

    agent.set_system_prompt("You are a helpful assistent")

    return agent


def init_moderate_agent(client):
    agent = JudgeAgent(client, name="moderator")
    return agent


##############################################################
# used for defense method CIDER
import types
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import abc


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
def generate_denoised_img_diffusion(
    imgpth: list[str],
    save_dir: str,
    cps: int,
    step=50,
    device: str = "cuda",
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

    model = DiffusionRobustModel(device=device)
    iterations = range(step, cps * step, step)
    b_num = ceil(len(resized_imgs) / batch_size)  # how man runs do we need
    for b in tqdm.tqdm(range(b_num), desc="denoising batch"):
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

    model = DiffusionRobustModel(device=f"cuda:{device}")
    iterations = range(step, cps * step, step)
    b_num = ceil(len(resized_imgs) / batch_size)  # how man runs do we need
    for b in tqdm.tqdm(range(b_num), desc="denoising batch"):
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

    with open(text_file, "r", encoding="utf-8") as f:
        attack_cases = json.load(f)
        queries = [case["attack_prompt"] for case in attack_cases]

    print(f"Loading {len(queries)} queries from {text_file}")
    print(queries)
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


######################################################################
# # used for defense method HiddenDectect
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
import torch.nn.functional as N

# Similarly, these internal functions are also copied over


def find_conv_mode(model_name):
    # ... (keep as is)
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"


def adjust_query_for_images(qs, model_config):
    # ... (keep as is, added model_config parameter)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


# ... Other helper functions like load_image, prepare_imgs_tensor_both_cases are also copied over ...


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


# ...


def prepare_imgs_tensor_both_cases(sample, image_processor, model_config):
    # ... (keep as is, but pass image_processor and model_config)
    img_prompt = [load_image(sample["img"])] if sample.get("img") else None
    if img_prompt is None:
        return None, None
    images_size = [img.size for img in img_prompt if img is not None]
    images_tensor = process_images(img_prompt, image_processor, model_config)
    return images_tensor, images_size


# --- This is the new core function we created ---
def score_and_generate_hiddendetect(
    sample,
    model,
    tokenizer,
    image_processor,
    refusal_token_ids,
    threshold,
    s=16,
    e=29,
    max_new_tokens=2048,
):
    """
    Calculate HiddenDetect score for a single sample, if passed then directly generate output.

    Returns: (score, output_text)
    """
    # --- 1. Prepare input (logic from hiddendetect_test) ---
    model_name = model.config._name_or_path

    # Construct prompt
    conv = conv_templates[find_conv_mode(model_name)].copy()
    qs = sample["txt"]
    if sample.get("img"):
        qs = adjust_query_for_images(qs, model.config)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # Process images
    images_tensor, image_sizes = None, None
    if sample.get("img"):
        images_tensor, image_sizes = prepare_imgs_tensor_both_cases(
            sample, image_processor, model.config
        )
        if images_tensor is not None:
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # --- 2. Calculate score (logic from hiddendetect_test) ---
    with torch.no_grad():
        outputs = model(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            output_hidden_states=True,
        )

    # Extract norm and lm_head
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    else:  # Applicable to new model structures like Llama-3
        norm = model.norm

    lm_head = model.lm_head

    F = []
    for r in outputs.hidden_states[1:]:  # Skip embedding layer
        layer_output = norm(r)
        logits = lm_head(layer_output)
        next_token_logits = logits[:, -1, :]
        reference_tokens = refusal_token_ids.to(next_token_logits.device)
        cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
        F.append(cos_sim.item())

    F_selected = F[s : e + 1]
    score = np.trapz(np.array(F_selected)) if F_selected else 0.0

    # --- 3. Judge and generate ---
    if score > threshold:
        # Score exceeds threshold, intercept and return refusal message
        output_text = "I'm sorry, I can't assist with that."
        print("Response refused due to high HiddenDetect score.")
        print(f"HiddenDetect score: {score}, output: {output_text}")
        return score, output_text
    else:
        # Score passed, use same input for generation
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,  # Ensure deterministic output
                temperature=0.0,  # Greedy decoding
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # Decode output
        # We need to remove input part from output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        outputs = outputs.strip().replace("\n", " ")
        print("Response generated successfully.")
        print(f"HiddenDetect score: {score}, output: {outputs}")

        return score, outputs


############################################################################
# used for defense method CoCA


def generate_with_coca(case, model, processor, safety_principle, alpha, max_new_tokens):
    """
    Generate output for a single sample using CoCA defense method.

    Args:
        case (dict): Sample dictionary containing 'attack_prompt' and 'attack_image_path'.
        model: Pre-loaded victim model (e.g., LlavaForConditionalGeneration).
        processor: Corresponding processor.
        safety_principle (str): Safety principle used for guidance.
        alpha (float): CoCA amplification coefficient.
        max_new_tokens (int): Maximum generation length.

    Returns:
        str: Generated text after CoCA defense.
    """
    user_question = case.get("attack_prompt", "")
    image_path = case.get("attack_image_path", None)

    # --- 1. Prepare two types of inputs ---
    # a) Input with safety principle
    prompt_text_with_safety = f"{safety_principle} {user_question}"
    conversation_with_safety = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text_with_safety},
                {"type": "image"},
            ],
        },
    ]
    prompt_with_safety = processor.apply_chat_template(
        conversation_with_safety, add_generation_prompt=True
    )

    # b) Input without safety principle
    conversation_without_safety = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question},
                {"type": "image"},
            ],
        },
    ]
    prompt_without_safety = processor.apply_chat_template(
        conversation_without_safety, add_generation_prompt=True
    )

    # --- 2. Load image and create input tensors ---
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "[CoCA Error: Failed to load image]"

    inputs_with_safety = processor(
        images=raw_image, text=prompt_with_safety, return_tensors="pt"
    ).to(model.device, dtype=torch.float16)

    inputs_without_safety = processor(
        images=raw_image, text=prompt_without_safety, return_tensors="pt"
    ).to(model.device, dtype=torch.float16)

    # --- 3. Implement CoCA custom generation loop ---
    input_ids_with_safety = inputs_with_safety["input_ids"]
    input_ids_without_safety = inputs_without_safety["input_ids"]
    pixel_values = inputs_with_safety["pixel_values"]
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        for i in range(max_new_tokens):
            # Get logits from both branches
            outputs_with_safety = model(
                input_ids=input_ids_with_safety, pixel_values=pixel_values
            )
            logits_with_safety = outputs_with_safety.logits[:, -1, :]

            outputs_without_safety = model(
                input_ids=input_ids_without_safety, pixel_values=pixel_values
            )
            logits_without_safety = outputs_without_safety.logits[:, -1, :]

            # Calculate Safety Delta and apply CoCA calibration
            safety_delta = logits_with_safety - logits_without_safety
            calibrated_logits = logits_with_safety + alpha * safety_delta

            # Select next token
            next_token_id = torch.argmax(calibrated_logits, dim=-1).unsqueeze(-1)

            # Synchronously update both sequences
            input_ids_with_safety = torch.cat(
                [input_ids_with_safety, next_token_id], dim=-1
            )
            input_ids_without_safety = torch.cat(
                [input_ids_without_safety, next_token_id], dim=-1
            )

            # Check for end token
            if next_token_id.item() == eos_token_id:
                break

    # --- 4. Decode and return result ---
    initial_prompt_len = inputs_with_safety["input_ids"].shape[1]

    if input_ids_with_safety.shape[1] > initial_prompt_len:
        generated_ids = input_ids_with_safety[:, initial_prompt_len:]
        final_output = processor.decode(generated_ids[0], skip_special_tokens=True)
        print("CoCA generated new content.")
        print(f"Output: {final_output}")
    else:
        final_output = (
            "CoCA did not generate any new content."  # No new content generated
        )
        print("CoCA did not generate any new content.")

    return final_output.strip()

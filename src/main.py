import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import callback_server
import dataloader
import performance_timer
import wandb
from double_relaxed_contrastive_loss import DoubleRelaxedContrastiveLoss
from model import ClipReverie
from stare_feature_extractor.FeatureExtractor import (
    FeatureExtractor,
    ImageResult,
    InstructionResult,
)

load_dotenv()

warnings.simplefilter("ignore")


class TextImageRetrievalMain:

    VAL_ENVIRONMENT_DICT = {
        "ltrrie": ["EU6Fwq7SyZv", "oLBMNvg9in8", "x8F5xyUWy9e", "zsNo4HB9uLZ"],
        "original_reftext": ["val"],
        "new_reftext": ["val"],
    }
    TEST_ENVIRONMENT_DICT = {
        "ltrrie": ["2azQ1b91cZZ", "QUCTc6BB5sX", "TbHJrupSAjP", "X7HyMhZNoso"],
        "original_reftext": ["home", "other", "shelf", "street", "oov", "semantic"],
        "new_reftext": ["home", "other", "shelf", "street"],
        "textcaps": ["test"],
    }

    def __init__(self, args_):
        self.device = "cuda:0"
        self.args = args_

        self.model_output_prefix = "model/model_tir"

        self.log_wandb = False
        self.wandb_url = ""
        if args_.wandb_name != "":
            self.log_wandb = True
            self.wandb_url = wandb.run.get_url()
        try:
            self.commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .strip()
                .decode("utf-8")
            )
        except subprocess.CalledProcessError:
            self.commit_id = "git not found"
        self.yyyy_mm_dd_hh_mm = time.strftime("%Y%m%d-%H%M")
        self.launch_command = " ".join(["python"] + sys.argv)

        self.model = ClipReverie(
            patch_x=int(args_.patch_x),
            patch_y=int(args_.patch_y),
            default_w=float(args_.default_w),
        ).cuda(self.device)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"model file was loaded from {path}.")

    def save_model(self, path):
        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def train_model(self, train_dataset_name, val_dataset_name, test_dataset_name):
        print(f"Currently loading train dataset {train_dataset_name}... ")
        train_dataset = dataloader.reverie_dataset(
            split="train",
            env="train",
            eval=False,
            dataset_name_list=train_dataset_name,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.args.bs))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.args.lr),
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01,
        )

        best_score = 0
        best_epoch = 0
        best_test_lb = None
        for epoch in range(int(self.args.epochs)):
            print(f"\n==== Epoch {epoch} ================")
            loss = self.train_epoch(train_dataloader, optimizer)
            print(f"Epoch: {epoch},  Loss: {loss:.4f}")
            wandb_log = {}
            if epoch >= self.args.eval_start_epoch and epoch % self.args.eval_freq == 0:
                test_lb = self.test_model(test_dataset_name)
                val_score = {}

                for val_target_dataset_name in val_dataset_name:
                    val_score[val_target_dataset_name] = self.evaluate(
                        "val_unseen",
                        self.VAL_ENVIRONMENT_DICT[val_target_dataset_name],
                        dataset_name=val_target_dataset_name,
                    )
                    print(
                        f"val_unseen_{val_target_dataset_name}: "
                        + ", ".join(
                            [
                                f"{val:.2f}"
                                for val in val_score[val_target_dataset_name][:5]
                            ]
                        )
                    )
                    wandb_log[f"val_{val_target_dataset_name}"] = {
                        "mrr": val_score[val_target_dataset_name][0],
                        "R@1": val_score[val_target_dataset_name][1],
                        "R@5": val_score[val_target_dataset_name][2],
                        "R@10": val_score[val_target_dataset_name][3],
                        "R@20": val_score[val_target_dataset_name][4],
                    }

                eval_score = 0
                eval_metrics = [
                    "mrr",
                    "r@1",
                    "r@5",
                    "r@10",
                    "r@20",
                ]
                for val_target_dataset_name in val_dataset_name:
                    for eval_metric in self.args.eval_metric:
                        eval_score += val_score[val_target_dataset_name][
                            eval_metrics.index(eval_metric)
                        ]

                if eval_score > best_score:
                    best_epoch = epoch
                    best_score = eval_score
                    model_path = f"{self.model_output_prefix}_best.pth"
                    print(f"save model file as best.: {model_path}")
                    self.save_model(model_path)
                    best_test_lb = test_lb

            if self.log_wandb:
                wandb_log["training loss per epoch"] = loss
                wandb.log(wandb_log)

            self.save_model(f"{self.model_output_prefix}_{epoch:03d}.pth")

        print(f"best epoch: {best_epoch}")
        os.system(
            f"cp {self.model_output_prefix}_{best_epoch:03d}.pth {self.model_output_prefix}_best.pth"
        )

        print(f"\n==== RESULTS for best epoch {best_epoch} =====")
        print("leaderboard_output: ", ", ".join(map(str, best_test_lb)))

        save_path = (
            f"result/{self.args.wandb_name}_leaderboard_{self.yyyy_mm_dd_hh_mm}.csv"
        )
        with open(save_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([f"launch_command: {self.launch_command}"])
            writer.writerow([f"best_epoch: {best_epoch}"])
            writer.writerow([f"commit_id: {self.commit_id}"])
            writer.writerow([f"wandb_url: {self.wandb_url}"])
            writer.writerow(best_test_lb)

    def test_model(self, dataset_name):

        wandb_log = {}
        leaderboard_output = []

        for test_target_dataset_name in dataset_name:
            test_score = self.evaluate(
                "test",
                self.TEST_ENVIRONMENT_DICT[test_target_dataset_name],
                file_output=f"result/{self.args.wandb_name}_test_{test_target_dataset_name}_{self.yyyy_mm_dd_hh_mm}",
                dataset_name=test_target_dataset_name,
            )
            print(
                f"test_{test_target_dataset_name}: "
                + ", ".join([f"{val:.2f}" for val in test_score[:5]])
            )
            wandb_log[f"test_{test_target_dataset_name}"] = {
                "mrr": test_score[0],
                "R@1": test_score[1],
                "R@5": test_score[2],
                "R@10": test_score[3],
                "R@20": test_score[4],
            }
            leaderboard_output.extend(
                [
                    f"{test_score[1]:.1f}",
                    f"{test_score[2]:.1f}",
                    f"{test_score[3]:.1f}",
                    f"{test_score[4]:.1f}",
                ]
            )

        if self.log_wandb:
            wandb.log(wandb_log)

        print("leaderboard_output: ", ", ".join(map(str, leaderboard_output)))

        save_path = (
            f"result/{self.args.wandb_name}_leaderboard_{self.yyyy_mm_dd_hh_mm}.csv"
        )
        with open(save_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([f"launch_command: {self.launch_command}"])
            writer.writerow(["best_epoch: "])
            writer.writerow([f"commit_id: {self.commit_id}"])
            writer.writerow([f"wandb_url: {self.wandb_url}"])
            writer.writerow(leaderboard_output)

        return leaderboard_output

    def cross_entropy(self, preds, targets, reduction="none"):
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        t_loss = 0
        n_ex = 0
        for (
            gt_img_ids,
            pseudo_gt_img_ids,
            instruction_clip,
            image_clip,
            stella_feats,
            ocr_info,
            instruction_stella,
            target_object_noun_stella,
            target_object_explanation_stella,
            gpt4o_stella,
            patch_clip,
            patch_dinov2,
            multilayer_dinov2,
            ne_tokens,
            ocr_tokens,
        ) in tqdm(dataloader):

            optimizer.zero_grad()

            if self.args.show_torchinfo:
                summary(
                    self.model,
                    input_data=[
                        instruction_clip.to(self.device),
                        image_clip.to(self.device),
                        instruction_stella.to(self.device),
                        target_object_explanation_stella.to(self.device),
                        gpt4o_stella.to(self.device),
                        patch_clip.to(self.device),
                        patch_dinov2.to(self.device),
                        multilayer_dinov2.to(self.device),
                        ne_tokens.to(self.device),
                        ocr_tokens.to(self.device),
                    ],
                )
                exit()

            logits = self.model(
                instruction_clip=instruction_clip.to(self.device),
                image_clip=image_clip.to(self.device),
                instruction_stella=instruction_stella.to(self.device),
                target_object_explanation_stella=target_object_explanation_stella.to(
                    self.device
                ),
                gpt4o_stella=gpt4o_stella.to(self.device),
                patch_clip=patch_clip.to(self.device),
                patch_dinov2=patch_dinov2.to(self.device),
                multilayer_dinov2=multilayer_dinov2.to(self.device),
                ne_tokens=ne_tokens.to(self.device),
                ocr_tokens=ocr_tokens.to(self.device),
            )

            criterion = DoubleRelaxedContrastiveLoss(
                alpha=self.args.alpha,
                lambda_neg=self.args.lambda_neg,
                gamma_semi=self.args.gamma,
                m=self.args.m,
            )
            loss = criterion(logits, gt_img_ids, pseudo_gt_img_ids)

            t_loss += loss
            loss.backward()
            optimizer.step()
            n_ex += 1

        return loss / n_ex

    def calc_score(self, probs, gt_id_list, imgId_list_full, eval_baseline=False):
        mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0

        ranks = []
        if not eval_baseline:
            imgId_list_full = [i[0] for i in imgId_list_full]

        imgId_list = [i for i in imgId_list_full]

        imgId_list = list(dict.fromkeys(imgId_list))

        for gt_tuple in gt_id_list:
            gt = gt_tuple[0]
            gt_idx = imgId_list.index(gt)
            rank_for_gt = sorted(probs, reverse=True).index(probs[gt_idx])
            ranks.append(rank_for_gt)

        top20 = []
        top20_rank = np.argsort(probs)[-20:][::-1]
        for i in top20_rank:
            top20.append(imgId_list_full[i])

        for i, rank in enumerate(sorted(ranks)):
            if rank < 20:
                recall20 += 1
            if rank < 10:
                recall10 += 1
            if rank < 5:
                recall5 += 1
            if rank < 1:
                recall1 += 1

            if i == 0:
                mrr = 100 / (rank + 1)

        recall20 = 100 * recall20 / len(ranks)
        recall10 = 100 * recall10 / len(ranks)
        recall5 = 100 * recall5 / len(ranks)
        recall1 = 100 * recall1 / len(ranks)

        return mrr, recall1, recall5, recall10, recall20, ranks, top20

    @torch.no_grad()
    def evaluate(self, split, environments, file_output=None, dataset_name="reftext"):
        self.model.eval()
        output = {}
        with torch.no_grad():
            mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0
            print(f"==========  {split.upper()} on {dataset_name} ===========")

            for env in environments:
                eval_dataset = dataloader.reverie_dataset(
                    split=split,
                    env=env,
                    eval=True,
                    dataset_name_list=[dataset_name],
                )
                if len(eval_dataset) == 0:
                    print(f"skip env {env} because there's no sample.")
                    continue
                eval_dataloader = DataLoader(eval_dataset, batch_size=1)

                n_ex = 0
                env_mrr, env_recall1, env_recall5, env_recall10, env_recall20 = (
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                env_output = []
                for (
                    raw_instruction,
                    gt_img_id,
                    imageId_list,
                    instId,
                    instruction_clip,
                    all_image_clip,
                    all_stella_feats,
                    all_ocr_info,
                    instruction_stella,
                    target_object_noun_stella,
                    target_object_explanation_stella,
                    all_gpt4o_stella,
                    all_patch_clip,
                    all_patch_dinov2,
                    all_multilayer_dinov2,
                    ne_tokens,
                    all_ocr_tokens,
                ) in tqdm(eval_dataloader):
                    batch_size = all_image_clip.shape[1]

                    all_image_clip = all_image_clip.to(self.device).squeeze(0)
                    all_gpt4o_stella = all_gpt4o_stella.to(self.device).squeeze(0)
                    all_stella_feats = all_stella_feats.to(self.device).squeeze(0)
                    all_ocr_info = all_ocr_info.to(self.device).squeeze(0)
                    all_patch_clip = all_patch_clip.to(self.device).squeeze(0)
                    all_patch_dinov2 = all_patch_dinov2.to(self.device).squeeze(0)
                    all_multilayer_dinov2 = all_multilayer_dinov2.to(
                        self.device
                    ).squeeze(0)
                    all_ocr_tokens = all_ocr_tokens.to(self.device).squeeze(0)

                    all_instruction_clip = instruction_clip.to(self.device).repeat(
                        batch_size, 1
                    )
                    all_instruction_stella = instruction_stella.to(self.device).repeat(
                        batch_size, 1
                    )
                    all_target_object_noun_stella = target_object_noun_stella.to(
                        self.device
                    ).repeat(batch_size, 1)
                    all_target_object_explanation_stella = (
                        target_object_explanation_stella.to(self.device).repeat(
                            batch_size, 1
                        )
                    )
                    all_ne_tokens = ne_tokens.to(self.device).repeat(batch_size, 1)

                    logits = self.model.forward(
                        instruction_clip=all_instruction_clip,
                        image_clip=all_image_clip,
                        instruction_stella=all_instruction_stella,
                        target_object_explanation_stella=all_target_object_explanation_stella,
                        gpt4o_stella=all_gpt4o_stella,
                        patch_clip=all_patch_clip,
                        patch_dinov2=all_patch_dinov2,
                        multilayer_dinov2=all_multilayer_dinov2,
                        ne_tokens=all_ne_tokens,
                        ocr_tokens=all_ocr_tokens,
                    )
                    _mrr, _recall1, _recall5, _recall10, _recall20, ranks, top20 = (
                        self.calc_score(
                            np.diag(logits.cpu().numpy()),
                            gt_img_id,
                            imageId_list,
                        )
                    )

                    if file_output:
                        dump = {}

                        dump["instruction_id"] = instId[0]
                        dump["instruction"] = raw_instruction[0]
                        dump["gt_image_id"] = gt_img_id
                        dump["mrr"] = str(_mrr)
                        dump["ranks"] = [str(x) for x in ranks]
                        dump["top20"] = top20

                        env_output.append(dump)

                    n_ex += 1
                    env_mrr += _mrr
                    env_recall1 += _recall1
                    env_recall5 += _recall5
                    env_recall10 += _recall10
                    env_recall20 += _recall20

                env_mrr = env_mrr / n_ex
                env_recall1 = env_recall1 / n_ex
                env_recall5 = env_recall5 / n_ex
                env_recall10 = env_recall10 / n_ex
                env_recall20 = env_recall20 / n_ex

                mrr += env_mrr
                recall1 += env_recall1
                recall5 += env_recall5
                recall10 += env_recall10
                recall20 += env_recall20

                print(
                    ", ".join(
                        [
                            f"num_inst : {n_ex}",
                            f"num_img : {len(imageId_list)} ... {env_mrr:.2f}",
                            f"{env_recall1:.2f}",
                            f"{env_recall5:.2f}",
                            f"{env_recall10:.2f}",
                            f"{env_recall20:.2f}",
                        ]
                    )
                )

                if file_output:
                    output[env] = env_output

            if file_output:
                path = f"{file_output}_{split}.json"
                save_dir = os.path.dirname(path)
                os.makedirs(save_dir, exist_ok=True)
                json.dump(output, open(path, "w"), indent=2, ensure_ascii=False)

        n_envs = len(environments)
        return (
            mrr / n_envs,
            recall1 / n_envs,
            recall5 / n_envs,
            recall10 / n_envs,
            recall20 / n_envs,
        )

    def predict_oneshot(self, image, instruction):
        self.model.eval()

        image_embeddings, ocr_tokens = self.embed_image(image)
        text_embeddings, ne_tokens = self.embed_instruction(instruction)

        with performance_timer.get_timer("forward with model", self.args.profiling):
            score = self.calc_similarity(
                image_embeddings, text_embeddings, ocr_tokens, ne_tokens
            )
            score = score[0][0]
        return score

    def embed_image(self, image):
        from PIL import Image

        image = Image.open(image)
        self.image_idx += 1
        image_result: ImageResult = self.feature_extractor.embed_image(image)
        image_embeddings, ocr_tokens = (
            image_result.image_embeddings,
            image_result.ocr_tokens,
        )

        return image_embeddings, ocr_tokens

    def embed_instruction(self, instruction):
        instruction_result: InstructionResult = (
            self.feature_extractor.embed_instruction(instruction)
        )
        text_embeddings, ne_tokens = (
            instruction_result.instruction_embeddings,
            instruction_result.ne_tokens,
        )
        return text_embeddings, ne_tokens

    def calc_similarity(
        self,
        image_embeddings,
        text_embeddings,
        ocr_tokens,
        ne_tokens,
    ):
        score: list[list[float]] = self.feature_extractor.calc_scores(
            image_embeddings,
            text_embeddings,
            ocr_tokens,
            ne_tokens,
        )

        return score

    def calc_cos_similarity_with_faiss(
        self,
        image_embeddings,
        text_embeddings,
        ocr_tokens,
        ne_tokens,
        num_k=10,
    ):
        print(f"image_embeddings shape: {image_embeddings.shape}")
        print(f"text_embeddings shape: {text_embeddings.shape}")
        print(f"ocr_tokens shape: {ocr_tokens.shape}")
        print(f"ne_tokens shape: {ne_tokens.shape}")

        logits: list[list[float]] = self.feature_extractor.calc_scores(
            image_embeddings,
            text_embeddings,
            ocr_tokens,
            ne_tokens,
        )

        logits = logits[0]
        return logits

        print(f"score: {logits}")
        scores = []
        for text_idx, text_score_list in enumerate(logits):
            topk_indices = np.argsort(text_score_list)[-num_k:][::-1]
            topk = [(int(idx), float(text_score_list[idx])) for idx in topk_indices]
            scores.append(topk)

        print(f"scores: {scores}")
        ret = [score[0] for score in scores]
        print(f"ret: {ret}")

        return ret


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_name", "-w", default="")
    parser.add_argument("--profiling", action="store_true", help="enable profiling")
    parser.add_argument(
        "--eval_start_epoch", type=int, default=2, help="evaluation start epoch num"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=2, help="evaluation frequency epoch num"
    )
    parser.add_argument("--infer_model_path", default="model/stare.pth")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--show_torchinfo", action="store_true")
    parser.add_argument(
        "--train_dataset_name",
        nargs="+",
        choices=["ltrrie", "new_reftext", "original_reftext"],
        default=[
            "ltrrie",
            "new_reftext",
            "original_reftext",
        ],
    )
    parser.add_argument(
        "--val_dataset_name",
        nargs="+",
        choices=["ltrrie", "original_reftext", "new_reftext"],
        default=[
            "ltrrie",
            "original_reftext",
            "new_reftext",
        ],
    )
    parser.add_argument(
        "--test_dataset_name",
        nargs="+",
        choices=["ltrrie", "original_reftext", "new_reftext", "textcaps"],
        default=[
            "ltrrie",
            "original_reftext",
            "new_reftext",
            "textcaps",
        ],
    )

    parser.add_argument(
        "--eval_metric",
        nargs="+",
        choices=["mrr", "r@1", "r@5", "r@10", "r@20"],
        default=["r@10"],
    )
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--bs", default=128)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--n_ocr", default="50")
    parser.add_argument(
        "--alpha", type=float, default="0.8", help="alpha for relaxed contrastive loss"
    )
    parser.add_argument(
        "--lambda_neg",
        type=float,
        default="0.8",
        help="lambda for relaxed contrastive loss",
    )
    parser.add_argument(
        "--gamma", type=float, default="0.5", help="gamma for relaxed contrastive loss"
    )
    parser.add_argument(
        "--m", type=float, default="2.0", help="m for relaxed contrastive loss"
    )
    parser.add_argument("--patch_x", type=int, default=3)
    parser.add_argument("--patch_y", type=int, default=3)
    parser.add_argument("--default_w", type=float, default=0.6)

    parser.add_argument(
        "--server_config",
        type=str,
        default="config/server_config.json",
        help="server config file path",
    )

    args = parser.parse_args(args=argv)

    return args


def tir_main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.wandb_name:
        wandb.init(project="clip-reverie", name=args.wandb_name)

    with performance_timer.get_timer("build model", args.profiling):
        tir = TextImageRetrievalMain(args)

    if args.mode == "train":
        tir.train_model(
            args.train_dataset_name,
            args.val_dataset_name,
            args.test_dataset_name,
        )
    elif args.mode == "test":
        if not args.infer_model_path:
            args.infer_model_path = "model/model_tir_best.pth"
        tir.load_model(args.infer_model_path)
        tir.test_model(args.test_dataset_name)
    elif args.mode == "start_server":
        tir.feature_extractor = FeatureExtractor(
            model="gpt-4o",
            max_tokens=1024,
            ckpt_path=tir.args.infer_model_path,
        )
        with open(args.server_config, "r", encoding="utf-8") as server_conf:
            conf = json.load(server_conf)
        callback_server.start(
            conf,
            tir.predict_oneshot,
            tir.embed_image,
            tir.embed_instruction,
            tir.calc_cos_similarity_with_faiss,
        )
    else:
        raise RuntimeError(f"unknown mode of [{args.mode}]")


if __name__ == "__main__":
    tir_main()

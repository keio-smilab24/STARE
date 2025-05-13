import json
import os
import numpy as np
from torch.utils.data import Dataset


class reverie_dataset(Dataset):
    def __init__(
        self,
        split="train",
        env="train",
        eval=False,
        dataset_name_list: list = None,
        train_shuffle_seed=42,
    ):
        self.eval = eval
        self.train_shuffle_seed = train_shuffle_seed
        self.scene_text_num = 50
        self.ne_num = 10

        self.dataset_name_list = dataset_name_list

        if self.eval:
            if len(dataset_name_list) != 1:
                raise ValueError(
                    "dataset_name_list must be 1. got {len(dataset_name_list)}"
                )
            dataset_name = dataset_name_list[0]
            dataset_path = f"data/{dataset_name}/{dataset_name}_dataset/{dataset_name}_{split}_{env}.json"
            eval_features_path_base = (
                f"data/{dataset_name}/eval_features/eval_features_{split}_{env}"
            )
            eval_id_path = f"data/{dataset_name}/eval_features/eval_features_{split}_{env}_bbox_list.json"
            self.get_dataset(dataset_path)
            self.imageId_list = json.load(open(eval_id_path, "r"))["bboxId_list"]
            self.all_image_clip = self.open_npy(
                f"{eval_features_path_base}_image_clip.npy"
            )
            self.all_stella_feats = self.open_npy(
                f"{eval_features_path_base}_stella_feats.npy"
            )
            self.all_ocr_info = self.open_npy(f"{eval_features_path_base}_ocr_info.npy")
            self.all_gpt4o_stella = self.open_npy(
                f"{eval_features_path_base}_kanavip_stella.npy"
            )
            self.all_patch_clip = self.open_npy(
                f"{eval_features_path_base}_patch_clip.npy"
            )
            self.all_patch_dinov2 = self.open_npy(
                f"{eval_features_path_base}_patch_dinov2.npy"
            )
            self.all_multilayer_dinov2 = self.open_npy(
                f"{eval_features_path_base}_multilayer_dinov2.npy"
            )
            self.all_ocr_tokens = self.open_npy(
                f"{eval_features_path_base}_ocr_tokens.npy"
            )

        else:
            dataset_path = []
            for dataset_name in dataset_name_list:
                dataset_path.append(
                    f"data/{dataset_name}/{dataset_name}_dataset/{dataset_name}_{split}_{env}.json"
                )
            self.get_dataset(dataset_path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        gt_img_ids = self.db[idx]["gt_bbox_id"]
        raw_instruction = self.db[idx]["instruction"]
        instruction_id = self.db[idx]["instruction_id"]

        instruction_clip_path = self._get_instructin_feats_path("clip", instruction_id)
        instruction_feats_stella_path = self._get_instructin_feats_path(
            "instruction_stella", instruction_id
        )
        target_object_noun_feats_stella_path = self._get_instructin_feats_path(
            "target_object_noun_stella", instruction_id
        )
        target_object_explanation_feats_stella_path = self._get_instructin_feats_path(
            "target_object_explanation_stella", instruction_id
        )
        ne_tokens_path = self._get_instructin_feats_path("ne_tokens", instruction_id)

        instruction_clip = self.open_npy(instruction_clip_path).squeeze(0)
        instruction_stella = self.open_npy(instruction_feats_stella_path)
        target_object_noun_stella = self.open_npy(target_object_noun_feats_stella_path)
        target_object_explanation_stella = self.open_npy(
            target_object_explanation_feats_stella_path
        )
        ne_tokens = self.open_npy(ne_tokens_path)

        if not self.eval:
            gt_img_id = gt_img_ids[0]

            image_clip_path = self._get_image_feats_path("image_clip", gt_img_id)
            stella_feats_path = self._get_image_feats_path("stella_feats", gt_img_id)
            ocr_info_path = self._get_image_feats_path("ocr_info", gt_img_id)
            gpt4o_stella_path = self._get_image_feats_path("kanavip_stella", gt_img_id)
            patch_clip_path = self._get_image_feats_path("patch_clip", gt_img_id)
            patch_dinov2_path = self._get_image_feats_path("patch_dinov2", gt_img_id)
            multilayer_dinov2_path = self._get_image_feats_path(
                "multilayer_dinov2", gt_img_id
            )
            ocr_tokens_path = self._get_image_feats_path("ocr_tokens", gt_img_id)

            image_clip = self.open_npy(image_clip_path).squeeze(0)
            stella_feats = self.open_npy(stella_feats_path)
            ocr_info = self.open_npy(ocr_info_path)
            gpt4o_stella = self.open_npy(gpt4o_stella_path)
            patch_clip = self.open_npy(patch_clip_path)
            patch_dinov2 = self.open_npy(patch_dinov2_path)
            multilayer_dinov2 = self.open_npy(multilayer_dinov2_path)
            ocr_tokens = self.open_npy(ocr_tokens_path)

            complete_pseudo_gt = {rank: "" for rank in range(20)}
            for pseudo_gt in self.db[idx]["pseudo_gt"]:
                complete_pseudo_gt[pseudo_gt["rank"]] = pseudo_gt["gt_img_id"]
            pseudo_gt_img_ids = [complete_pseudo_gt[rank] for rank in range(20)]

            ret = (
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
            )
        else:
            instId = self.db[idx]["instruction_id"]
            ret = (
                raw_instruction,
                gt_img_ids,
                self.imageId_list,
                instId,
                instruction_clip,
                self.all_image_clip,
                self.all_stella_feats,
                self.all_ocr_info,
                instruction_stella,
                target_object_noun_stella,
                target_object_explanation_stella,
                self.all_gpt4o_stella,
                self.all_patch_clip,
                self.all_patch_dinov2,
                self.all_multilayer_dinov2,
                ne_tokens,
                self.all_ocr_tokens,
            )

        return ret

    def _get_instructin_feats_path(self, feats_name, instruction_id):
        feats_dir = "data/features/instruction"
        instruction_path_base = os.path.join(
            feats_dir, feats_name, f"{instruction_id}.npy"
        )
        return instruction_path_base

    def _get_image_feats_path(self, feats_name, gt_img_id):
        feats_dir = "data/features/image"
        image_path_base = os.path.join(feats_dir, feats_name, f"{gt_img_id}.npy")
        return image_path_base

    def get_dataset(self, dataset_path: list, shuffle=True):
        self.db = []

        if type(dataset_path) is str:
            dataset_path = [dataset_path]
        for path in dataset_path:
            print(f"Loading {path}")
            data = json.load(open(path, "r"))
            self.db.extend(data)

        if shuffle:
            np.random.seed(self.train_shuffle_seed)
            np.random.shuffle(self.db)

        if not self.eval:
            for data in self.db:
                if "pseudo_gt" not in data or data["pseudo_gt"] is None:
                    self.db.remove(data)
                    print(
                        f"Removed {data['instruction_id']} because it has no pseudo_gt"
                    )

    def open_npy(self, path):
        return np.load(path, allow_pickle=True)

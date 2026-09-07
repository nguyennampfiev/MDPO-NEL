import torch
import torch.nn.functional as F
from trl import DPOTrainer


class MultiNegCollator:
    def __init__(self, tokenizer, max_prompt_length: int = 1024, max_answer_length: int = 128):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length

    def __call__(self, features):
        prompts = [f["prompt"] for f in features]
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] if isinstance(f["rejected"], list) else [f["rejected"]] for f in features]
        batch_size = len(features)
        num_neg = max(len(items) for items in rejected)

        prompt_enc = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_prompt_length, return_tensors="pt")
        chosen_enc = self.tokenizer(chosen, padding=True, truncation=True, max_length=self.max_answer_length, return_tensors="pt")
        rejected_flat = [item for items in rejected for item in items]
        rejected_enc = self.tokenizer(rejected_flat, padding=True, truncation=True, max_length=self.max_answer_length, return_tensors="pt")

        seq_len = rejected_enc["input_ids"].shape[1]
        rejected_ids = torch.zeros(batch_size, num_neg, seq_len, dtype=torch.long)
        rejected_mask = torch.zeros(batch_size, num_neg, seq_len, dtype=torch.long)
        valid_n = torch.zeros(batch_size, dtype=torch.long)

        offset = 0
        for b, items in enumerate(rejected):
            n = len(items)
            rejected_ids[b, :n] = rejected_enc["input_ids"][offset : offset + n]
            rejected_mask[b, :n] = rejected_enc["attention_mask"][offset : offset + n]
            valid_n[b] = n
            offset += n

        return {
            "prompt_input_ids": prompt_enc["input_ids"],
            "prompt_attention_mask": prompt_enc["attention_mask"],
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "valid_n": valid_n,
            "chosen": chosen,
            "rejected": [items[0] for items in rejected],
        }


class MultiNegDPOTrainer(DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name, **kwargs):
        return dataset

    def get_train_dataloader(self):
        self._precomputed_train_ref_log_probs = True
        return super(DPOTrainer, self).get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        self._precomputed_eval_ref_log_probs = True
        return super(DPOTrainer, self).get_eval_dataloader(eval_dataset)

    def _compute_logps(self, model, prompt_ids, prompt_mask, cand_ids, cand_mask):
        input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, cand_mask], dim=1)
        prompt_len = prompt_ids.shape[1]
        seq_len = cand_ids.shape[1]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        cand_logits = logits[:, prompt_len - 1 : prompt_len + seq_len - 1, :]
        token_lps = cand_logits.log_softmax(-1).gather(2, cand_ids.unsqueeze(-1)).squeeze(-1)
        token_lps = token_lps.masked_fill(~cand_mask.bool(), 0.0)
        return token_lps.sum(-1) / cand_mask.sum(-1).clamp(min=1)

    def _compute_ref_logps(self, prompt_ids, prompt_mask, cand_ids, cand_mask):
        with torch.no_grad():
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    return self._compute_logps(self.model, prompt_ids, prompt_mask, cand_ids, cand_mask)
            if self.ref_model is not None:
                return self._compute_logps(self.ref_model, prompt_ids, prompt_mask, cand_ids, cand_mask)
        raise ValueError("No ref_model and model has no disable_adapter")

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]
        valid_n = inputs["valid_n"]

        batch_size, num_neg, rejected_len = rejected_ids.shape
        chosen_len = chosen_ids.shape[1]
        seq_len = max(chosen_len, rejected_len)
        pad_id = self.tokenizer.pad_token_id or 0
        if chosen_len < seq_len:
            chosen_ids = F.pad(chosen_ids, (0, seq_len - chosen_len), value=pad_id)
            chosen_mask = F.pad(chosen_mask, (0, seq_len - chosen_len), value=0)
        if rejected_len < seq_len:
            rejected_ids = F.pad(rejected_ids, (0, seq_len - rejected_len), value=pad_id)
            rejected_mask = F.pad(rejected_mask, (0, seq_len - rejected_len), value=0)

        all_ids = torch.cat([chosen_ids.unsqueeze(1), rejected_ids], dim=1).reshape(batch_size * (1 + num_neg), seq_len)
        all_mask = torch.cat([chosen_mask.unsqueeze(1), rejected_mask], dim=1).reshape(batch_size * (1 + num_neg), seq_len)
        prompt_exp = prompt_ids.unsqueeze(1).expand(batch_size, 1 + num_neg, -1).reshape(batch_size * (1 + num_neg), -1)
        mask_exp = prompt_mask.unsqueeze(1).expand(batch_size, 1 + num_neg, -1).reshape(batch_size * (1 + num_neg), -1)

        policy = self._compute_logps(model, prompt_exp, mask_exp, all_ids, all_mask)
        ref = self._compute_ref_logps(prompt_exp, mask_exp, all_ids, all_mask)
        rewards = (policy - ref).reshape(batch_size, 1 + num_neg)
        diff = rewards[:, :1] - rewards[:, 1:]
        valid_mask = torch.arange(num_neg, device=diff.device).unsqueeze(0) < valid_n.unsqueeze(1)
        loss = -F.logsigmoid(self.beta * diff)
        loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return (loss, None) if return_outputs else loss


class CandidateCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        prompts = [f["prompt"] for f in features]
        candidate_texts = [f["candidate_texts"] for f in features]
        labels = [f["labels"] for f in features]
        batch_size = len(features)
        num_candidates = max(len(c) for c in candidate_texts)

        prompt_enc = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        flat = [candidate for candidates in candidate_texts for candidate in candidates]
        cand_enc = self.tokenizer(flat, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        seq_len = cand_enc["input_ids"].shape[1]

        cand_ids = torch.zeros(batch_size, num_candidates, seq_len, dtype=torch.long)
        cand_mask = torch.zeros(batch_size, num_candidates, seq_len, dtype=torch.long)
        label_tensor = torch.zeros(batch_size, num_candidates, dtype=torch.long)
        offset = 0
        for b, candidates in enumerate(candidate_texts):
            n = len(candidates)
            cand_ids[b, :n] = cand_enc["input_ids"][offset : offset + n]
            cand_mask[b, :n] = cand_enc["attention_mask"][offset : offset + n]
            label_tensor[b, :n] = torch.tensor(labels[b], dtype=torch.long)
            offset += n

        return {
            "prompt_input_ids": prompt_enc["input_ids"],
            "prompt_attention_mask": prompt_enc["attention_mask"],
            "candidates": cand_ids,
            "candidate_masks": cand_mask,
            "labels": label_tensor,
            "chosen": [f.get("chosen", "") for f in features],
            "rejected": [f.get("rejected", "") for f in features],
        }


class MDPOTrainer(MultiNegDPOTrainer):
    """Candidate-array objective with rank loss plus NIL abstention loss."""

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        candidates = inputs["candidates"]
        masks = inputs["candidate_masks"]
        labels = inputs["labels"].bool()

        batch_size, num_candidates, seq_len = candidates.shape
        prompt_exp = prompt_ids.unsqueeze(1).expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        mask_exp = prompt_mask.unsqueeze(1).expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        cand_flat = candidates.reshape(batch_size * num_candidates, seq_len)
        mask_flat = masks.reshape(batch_size * num_candidates, seq_len)

        policy = self._compute_logps(model, prompt_exp, mask_exp, cand_flat, mask_flat).reshape(batch_size, num_candidates)
        ref = self._compute_ref_logps(prompt_exp, mask_exp, cand_flat, mask_flat).reshape(batch_size, num_candidates)
        rewards = policy - ref

        pos = labels
        neg = ~labels
        pair_mask = pos.unsqueeze(2) & neg.unsqueeze(1)
        rank_loss = -F.logsigmoid(self.beta * (rewards.unsqueeze(2) - rewards.unsqueeze(1)))
        rank_loss = (rank_loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        no_positive = ~pos.any(dim=1)
        nil_margin = -rewards.mean(dim=1)
        abstain_loss = -F.logsigmoid(self.beta * nil_margin)
        abstain_loss = abstain_loss[no_positive].mean() if no_positive.any() else torch.tensor(0.0, device=rewards.device)

        loss = rank_loss + abstain_loss
        return (loss, None) if return_outputs else loss

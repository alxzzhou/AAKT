import torch
import torch.nn.functional as F
from transformers import GPTJModel, GPTJConfig

from losses.custom_loss import CustomLoss


class AAKT(torch.nn.Module):
    def __init__(
            self,
            num_questions,
            num_tags,
            max_seq_len=8096,
            with_tags=True,
            with_times=True,
            **kwargs
    ):
        super().__init__()
        self.with_tags = with_tags
        self.with_times = with_times
        self.tag_emb = False
        self.correct_token_id = num_questions
        self.incorrect_token_id = num_questions + 1
        self.bos_token_id = num_questions + 2
        self.eos_token_id = num_questions + 3

        self.config = GPTJConfig(
            vocab_size=num_questions + 4,
            n_positions=max_seq_len,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            **kwargs,
        )
        self.model = GPTJModel(self.config)

        if self.with_times:
            self.times_encoder = torch.nn.Sequential(
                torch.nn.Linear(1, self.config.n_embd),
                torch.nn.Tanh(),
                torch.nn.Linear(self.config.n_embd, self.config.n_embd)
            )

        self.kts_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.config.n_embd, self.config.n_embd),
            torch.nn.LogSigmoid(),
            torch.nn.Linear(self.config.n_embd, 2)
        )

        if self.with_tags:
            self.tags_classifier = torch.nn.Sequential(
                # torch.nn.Linear(self.config.n_embd, self.config.n_embd),
                # torch.nn.LogSigmoid(),
                torch.nn.Linear(self.config.n_embd, num_tags)
            )

        if self.tag_emb:
            self.tags_embedding = torch.nn.Sequential(
                torch.nn.Linear(num_tags, self.config.n_embd),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.n_embd, self.config.n_embd),
                torch.nn.Tanh()
            )

    def forward(self,
                input_ids,  # [bs, seq_len], long
                input_times,  # [bs, seq_len], float
                labels_kts,  # [bs, seq_len]
                labels_tags):  # [bs, seq_len, num_skill]
        inputs_embeds = self.model.wte(input_ids)

        if self.with_times:
            times_embeds = self.times_encoder(input_times.unsqueeze(-1))
        else:
            times_embeds = torch.zeros_like(inputs_embeds)

        if self.tag_emb:
            tags_embeds = self.tags_embedding(labels_tags) / torch.sum(labels_tags, dim=-1, keepdim=True)
        else:
            tags_embeds = torch.zeros_like(inputs_embeds)

        hidden_states = self.model(inputs_embeds=inputs_embeds + times_embeds + tags_embeds)[0]  # [bs, seq_len, n_embd]
        preds_kts = self.kts_classifier(hidden_states)  # [bs, seq_len, 2]

        mask = torch.ne(labels_kts, -100)  # [bs, seq_len]

        focal = CustomLoss()
        loss_kts = focal(
            preds_kts.view(-1, 2),  # [bs * seq_len, 2]
            labels_kts.view(-1),  # [bs * seq_len]
            mask.view(-1)  # [bs * seq_len]
        )

        if self.with_tags and not self.tag_emb:
            preds_tags = self.tags_classifier(inputs_embeds)  # [bs, seq_len, num_tags]
            preds_tags = F.softmax(preds_tags, dim=-1)
            loss_tags_mask = (labels_tags[:, :, 0] != -100).float()
            assert torch.all((labels_tags[:, :, 0] == -100) + torch.all(labels_tags != -100, dim=-1))
            labels_tags = labels_tags / labels_tags.sum(dim=-1, keepdims=True)
            loss_tags = torch.kl_div(preds_tags.log(), labels_tags)
            loss_tags = (loss_tags.sum(dim=-1) * loss_tags_mask).sum() / loss_tags_mask.sum()
        else:
            loss_tags = torch.zeros_like(loss_kts)

        return loss_kts + loss_tags, loss_kts, loss_tags, preds_kts

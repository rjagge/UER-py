"""
This script provides an example to wrap TencentPretrain for text-to-text fine-tuning.
"""
import sys
import os
import random
import argparse
import torch
from rouge import Rouge

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from uer.model_saver import save_model
from uer.decoders import *
from uer.targets import *
from finetune.run_classifier import *

rouge = Rouge()

class Text2text(torch.nn.Module):
    def __init__(self, args):
        super(Text2text, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.tgt_embedding = str2embedding[args.tgt_embedding](
            args, len(args.tokenizer.vocab)
        )
        self.decoder = str2decoder[args.decoder](args)
        self.target = LmTarget(args, len(args.tokenizer.vocab))
        if args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight
        if args.share_embedding:
            self.tgt_embedding.word_embedding.weight = (
                self.embedding.word_embedding.weight
            )

    def encode(self, src, seg):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        return memory_bank

    def decode(self, src, memory_bank, tgt):
        tgt_in, tgt_out, _ = tgt
        decoder_emb = self.tgt_embedding(tgt_in, None)
        hidden = self.decoder(memory_bank, decoder_emb, (src,))
        output = self.target.output_layer(hidden)
        return output

    def forward(self, src, tgt, seg, memory_bank=None, only_use_encoder=False):
        if only_use_encoder:
            return self.encode(src, seg)
        if memory_bank is not None:
            return self.decode(src, memory_bank, tgt)
        tgt_in, tgt_out, _ = tgt
        memory_bank = self.encode(src, seg)
        output = self.decode(src, memory_bank, tgt)
        if tgt_out is None:
            return None, output
        else:
            decoder_emb = self.tgt_embedding(tgt_in, None)
            hidden = self.decoder(memory_bank, decoder_emb, (seg,))
            loss = self.target(hidden, tgt_out, seg)[0]
            return loss, output

def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):

            line = line[:-1].split('\t')

            if len(line) == 3:
                text = line[0] + SEP_TOKEN + line[1]
                label = line[2]
            else:
                text, label = line[0], line[1]

            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN]
            )
            tgt_in = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(label) + [SEP_TOKEN]
            )
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            if len(tgt_in) > args.tgt_seq_length:
                tgt_in = tgt_in[: args.tgt_seq_length]
            tgt_out = tgt_in[1:] + [PAD_ID]

            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            while len(tgt_in) < args.tgt_seq_length:
                tgt_in.append(PAD_ID)
                tgt_out.append(PAD_ID)

            dataset.append((src, tgt_in, tgt_out, seg))

    return dataset


def batch_loader(batch_size, src, tgt_in, tgt_out, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_in_batch = tgt_in[i * batch_size : (i + 1) * batch_size, :]
        tgt_out_batch = tgt_out[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, None

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_in_batch = tgt_in[instances_num // batch_size * batch_size :, :]
        tgt_out_batch = tgt_out[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, None


def train_model(
    args, model, optimizer, scheduler, src_batch, tgt_in_batch, tgt_out_batch, seg_batch
):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_in_batch = tgt_in_batch.to(args.device)
    tgt_out_batch = tgt_out_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _ = model(src_batch, (tgt_in_batch, tgt_out_batch, src_batch), seg_batch)

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):

    src = torch.LongTensor([example[0] for example in dataset])
    tgt_in = torch.LongTensor([example[1] for example in dataset])
    tgt_out = torch.LongTensor([example[2] for example in dataset])
    seg = torch.LongTensor([example[3] for example in dataset])

    generated_sentences = []
    args.model.eval()

    for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, _) in enumerate(
        batch_loader(args.batch_size, src, tgt_in, tgt_out, seg)
    ):

        src_batch = src_batch.to(args.device)
        tgt_in_batch = torch.zeros(
            tgt_in_batch.size()[0], 1, dtype=torch.long, device=args.device
        )
        for j in range(tgt_in_batch.size()[0]):
            tgt_in_batch[j][-1] = args.tokenizer.vocab.get(CLS_TOKEN)

        seg_batch = seg_batch.to(args.device)

        with torch.no_grad():
            memory_bank = args.model(**dict(src = src_batch, tgt = None, seg = seg_batch, only_use_encoder = True))
        for _ in range(args.tgt_seq_length):
            tgt_out_batch = tgt_in_batch
            with torch.no_grad():
                outputs = args.model(
                    src_batch,
                    (tgt_in_batch, tgt_out_batch, src_batch),
                    None,
                    memory_bank=memory_bank
                )

            next_token_logits = outputs[:, -1]
            next_tokens = torch.argmax(next_token_logits, dim=1).unsqueeze(1)
            tgt_in_batch = torch.cat([tgt_in_batch, next_tokens], dim=1)

        for j in range(len(outputs)):
            sentence = " ".join(
                [
                    args.tokenizer.inv_vocab[token_id.item()]
                    for token_id in tgt_in_batch[j][1:]
                ]
            )
            generated_sentences.append(sentence)

    if args.metrics == 0:
        labels = {}
        labels_num = 0
        for example in dataset:
            label = "".join(
                [args.tokenizer.inv_vocab[token_id] for token_id in example[2][:-2]]
            ).split(SEP_TOKEN)[0]
            if not labels.get(label, None):
                labels[label] = labels_num
                labels_num += 1
        confusion_matrix = torch.zeros(labels_num, labels_num, dtype=torch.long)
        correct = 0
        # print(labels)
        for i, example in enumerate(dataset):

            tgt = example[2]
            tgt_token = " ".join(
                [args.tokenizer.inv_vocab[token_id] for token_id in tgt[:-2]]
            )
            generated_sentences[i] = generated_sentences[i].split(SEP_TOKEN)[0]

            pred = "".join(generated_sentences[i].split(" "))
            gold = "".join(tgt_token.split(SEP_TOKEN)[0].split(" "))
            # 这里为什么要加一个normlize
            pred = normlize_output(pred, list(labels.keys()))

            # print(pred, gold)

            if pred in labels.keys():
                confusion_matrix[labels[pred], labels[gold]] += 1

            if pred == gold:
                correct += 1

        args.logger.info(
            "Acc. (Correct/Total): {:.4f} ({}/{}) ".format(
                correct / len(dataset), correct, len(dataset)
            )
        )
        return correct / len(dataset)

    elif args.metrics == 1:
        r_l = 0
        for i, example in enumerate(dataset):
            label = " ".join(
                [args.tokenizer.inv_vocab[token_id] for token_id in example[2][:-2]]
            ).split(SEP_TOKEN)[0]
            score = rouge.get_scores(hyps=generated_sentences[i], refs=label)
            r_l += score[0]['rouge-l']['f']
        args.logger.info("Rouge-L. {:.4f} ".format(r_l / len(dataset)))

        return r_l / len(dataset)

    elif args.metrics == 2:
        bp = 0
        for i, example in enumerate(dataset):
            label = (
                "".join(
                    [args.tokenizer.inv_vocab[token_id] for token_id in example[2][:-2]]
                )
                .split(SEP_TOKEN)[0]
                .split('_')
            )
            pred = "".join(generated_sentences[i].split(' ')).split('_')
            # print(pred, label)
            bp += bpref(pred, label)
        args.logger.info("Bpref. {:.4f} ".format(bp / len(dataset)))

        return bp / len(dataset)


def bpref(pred, gold):
    score, n, p = 0, 0, 0
    for word in pred:
        if word in gold:
            score += 1 - n / len(pred)
            p += 1
        else:
            n += 1

    if p != 0:
        return score / p
    else:
        return score

def normlize_output(pred, labels):
    scores = torch.ones(len(labels))
    try:
        for i, label in enumerate(labels):
            scores[i] = rouge.get_scores(hyps=pred, refs=label)[0]['rouge-l']['f']
        return labels[torch.argmax(scores).item()]
    except:
        return random.choice(labels)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument(
        "--tgt_seq_length", type=int, default=32, help="Output sequence length."
    )
    parser.add_argument(
        "--metrics",
        type=int,
        default=0,
        help="0: Accuracy for classification. 1: Rouge-L for summarization, 2: Bpref. for keyword generation",
    )

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Text2text(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
        args.amp = amp

    if torch.cuda.device_count() > 1:
        args.logger.info(
            "{} GPUs are available. Let's use them.".format(torch.cuda.device_count())
        )
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt_in = torch.LongTensor([example[1] for example in trainset])
        tgt_out = torch.LongTensor([example[2] for example in trainset])
        seg = torch.LongTensor([example[3] for example in trainset])

        model.train()
        for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, _) in enumerate(
            batch_loader(batch_size, src, tgt_in, tgt_out, seg)
        ):
            loss = train_model(
                args,
                model,
                optimizer,
                scheduler,
                src_batch,
                tgt_in_batch,
                tgt_out_batch,
                seg_batch,
            )
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info(
                    "Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(
                        epoch, i + 1, total_loss / args.report_steps
                    )
                )
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()

import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed
from uer.utils.mask import mask_seq
import pkuseg


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2**20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.dict_path = args.dict_path
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if pos >= end:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                    break

                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        # Clear buffer.
                        docs_buffer = []
                    continue
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                if len(sentence) > 0:
                    document.append(sentence)

        dataset_writer.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_random_next = 0

                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    pad_num = 0
                    if len(src) != self.seq_length:
                        pad_num = self.seq_length - len(src)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        src = (src, pad_num)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        src = (src, pad_num)
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences
        self.seg = pkuseg.pkuseg(model_name = "default", user_dict = self.dict_path, postag = False)

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f: 
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) + [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            # Build instances from documents.
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            # Clear buffer.
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                # Save instances.
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for i in range(len(docs_buffer)):
            all_documents += docs_buffer[i]
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = self.mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = ((src, 0), tgt, seg_pos)
            else:
                instance = ((src, 0), seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]

        if len(src) == 0:
            return instances

        seg_pos = [len(src)]

        pad_num = self.seq_length - len(src)
        
        if not self.dynamic_masking:
            src, tgt = self.mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = ((src, pad_num), tgt, seg_pos)
        else:
            instance = ((src, pad_num), seg_pos)

        instances.append(instance)
        return instances
    
    def mask_seq(self, src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
        vocab = tokenizer.vocab
        PAD_ID = vocab.get(PAD_TOKEN)
        for i in range(len(src) - 1, -1, -1):
            if src[i] != PAD_ID:
                break
        src_no_pad = src[:i + 1]

        # 这里会通过jieba的分词得到该文本的token_index，[[0,3],[3,2]] [[start_idx, length], [start_idx, length]]
        tokens_index, src_no_pad = self.create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
        if len(src_no_pad) < len(src):
            src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
        else:
            src = src_no_pad

        random.shuffle(tokens_index)
        num_to_predict = max(1, int(round(len(src_no_pad) * 0.15)))
        tgt_mlm = []
        for index_set in tokens_index:
            if len(tgt_mlm) >= num_to_predict:
                break
            if whole_word_masking:
                i = index_set[0]
                mask_len = index_set[1]
                if len(tgt_mlm) + mask_len > num_to_predict:
                    continue

                for j in range(mask_len):
                    token = src[i + j]
                    tgt_mlm.append((i + j, token))
                    prob = random.random()
                    # 这里mask的机制和原生的bert有区别：
                    # 原生的mask是对该词组下的每一个idx进行单独的mask
                    # 这里的mask是对该词组做为一个整体，进行整体的替换
                    if prob < 0.8:
                        src[i + j] = vocab.get(MASK_TOKEN)
                    elif prob < 0.9:
                        while True:
                            rdi = random.randint(1, len(vocab) - 1)
                            if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                                break
                        src[i + j] = rdi
            elif span_masking:
                i = index_set[0]
                span_len = index_set[1]
                if len(tgt_mlm) + span_len > num_to_predict:
                    continue

                for j in range(span_len):
                    token = src[i + j]
                    tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    for j in range(span_len):
                        src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    for j in range(span_len):
                        while True:
                            rdi = random.randint(1, len(vocab) - 1)
                            if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                                break
                        src[i + j] = rdi
            else:
                i = index_set[0]
                token = src[i]
                tgt_mlm.append((i, token))
                prob = random.random()
                if prob < 0.8:
                    src[i] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i] = rdi
        tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
        return src, tgt_mlm


    def create_index(self, src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
        tokens_index = []
        span_end_position = -1
        vocab = tokenizer.vocab
        PAD_ID = vocab.get(PAD_TOKEN)
        if whole_word_masking:
            # 用来保存该句的index，和src的差别目前知道的是多一个index为101的[CLS]
            src_wwm = []
            src_length = len(src)
            has_cls, has_sep = False, False
            if src[0] == vocab.get(CLS_TOKEN):
                src = src[1:]
                has_cls = True
            if src[-1] == vocab.get(SEP_TOKEN):
                src = src[:-1]
                has_sep = True
            sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
            # 需要改动的代码就只有这一块，感觉有点简单哈哈哈
            # import jieba as seg
            
            wordlist = self.seg.cut(sentence)
            if has_cls:
                src_wwm += [vocab.get(CLS_TOKEN)]
            # 使用jieba的分词结果，然后将原文本进行分词，用idx的方式保存下来
            for word in wordlist:
                position = len(src_wwm)
                src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                if len(src_wwm) < src_length:
                    # 每次保存一个词的起始位置和他的长度
                    tokens_index.append([position, len(src_wwm)-position])
            if has_sep:
                src_wwm += [vocab.get(SEP_TOKEN)]
            if len(src_wwm) > src_length:
                src = src_wwm[:src_length]
            else:
                src = src_wwm
        else:
            for (i, token) in enumerate(src):
                if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                    continue
                if not span_masking:
                    tokens_index.append([i])
                else:
                    if i < span_end_position:
                        continue
                    span_len = self.get_span_len(span_max_length, span_geo_prob)
                    span_end_position = i + span_len
                    if span_end_position > len(src):
                        span_len = len(src) - i
                    tokens_index.append([i, span_len])
        return tokens_index, src


    def get_span_len(self, max_span_len, p):
        geo_prob_cum = [0.0]
        geo_prob = 1.0
        for i in range(max_span_len + 1):
            if i == 0:
                continue
            if i == 1:
                geo_prob *= p
                geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
            else:
                geo_prob *= (1 - p)
                geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

        prob = geo_prob_cum[-1] * random.random()
        for i in range(len(geo_prob_cum) - 1):
            if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
                current_span_len = i + 1
        return current_span_len

class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    pad_num = 0
                    if len(src) != self.seq_length:
                        pad_num = self.seq_length - len(src)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.dict_path, self.span_masking, self.span_geo_prob, self.span_max_length)
                        src = (src, pad_num)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        src = (src, pad_num)
                        instance = (src, is_wrong_order, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                document = [self.vocab.get(CLS_TOKEN)] + document + [self.vocab.get(SEP_TOKEN)]

                instances_num = len(document) // (self.seq_length + 1)
                for i in range(instances_num):
                    src = document[i * (self.seq_length + 1): (i + 1) * (self.seq_length + 1)]
                    seg_pos = [self.seq_length]
                    src = (src, 0)
                    pickle.dump((src, seg_pos), dataset_writer)

                src = document[instances_num * (self.seq_length + 1):]
                if len(src) > 0:
                    seg_pos = [len(src)]
                    pad_num = self.seq_length + 1 - len(src)
                    src = (src, pad_num)
                    pickle.dump((src, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))

                instances_num = len(document) // self.seq_length
                for i in range(instances_num):
                    src = document[i * self.seq_length: (i + 1) * self.seq_length]
                    tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                    tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                    seg_pos = [self.seq_length]
                    src = (src, 0)
                    pickle.dump((src, tgt_forward, tgt_backward, seg_pos), dataset_writer)

                src = document[instances_num * self.seq_length:]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                seg_pos = [len(src)]
                pad_num = self.seq_length - len(src)
                src = (src, pad_num)
                pickle.dump((src, tgt_forward, tgt_backward, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


class MtDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MtDataset, self).__init__(args, vocab, tokenizer)
        self.tgt_seq_length = args.tgt_seq_length
        self.src_vocab, self.src_tokenizer = vocab, tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.tgt_vocab = self.tgt_tokenizer.vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(document_src))
                tgt = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(document_tgt))

                src = [self.src_vocab.get(CLS_TOKEN)] + src + [self.src_vocab.get(SEP_TOKEN)]
                tgt = [self.tgt_vocab.get(CLS_TOKEN)] + tgt + [self.tgt_vocab.get(SEP_TOKEN)]

                src, tgt = src[:self.seq_length], tgt[:self.tgt_seq_length + 1]
                seg_pos = [len(src)]

                pad_num = self.seq_length - len(src)
                src = (src, pad_num)
                pad_num = self.tgt_seq_length + 1 - len(tgt)
                tgt = (tgt, pad_num)

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class T5Dataset(MlmDataset):
    '''
    T5 can reuse the code of MlmDataset.
    '''
    pass


class GsgDataset(BertDataset):
    def __init__(self, args, vocab, tokenizer):
        super(GsgDataset, self).__init__(args, vocab, tokenizer)
        self.sentence_selection_strategy = args.sentence_selection_strategy
        self.tgt_seq_length = args.tgt_seq_length

    def create_single_instance(self, src, tgt):
        src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
        tgt = [self.vocab.get(CLS_TOKEN)] + tgt + [self.vocab.get(SEP_TOKEN)]
        seg_pos = [len(src)]
        pad_num = self.seq_length - len(src)
        src = (src, pad_num)
        pad_num = self.tgt_seq_length - len(tgt)
        tgt = (tgt, pad_num)
        instance = (src, tgt, seg_pos)
        return instance

    def create_ins_from_doc(self, all_documents, document_index):
        sentence_selection_strategy = self.sentence_selection_strategy
        instances = []
        mask_seq_list = []
        tmp_document = []
        src = []
        tgt = []
        i = 0
        document = all_documents[document_index]
        target_seq_length, target_tgt_seq_length = self.seq_length - 2, self.tgt_seq_length - 2
        for segment in document:
            if len(segment) < target_seq_length and len(segment) < target_tgt_seq_length:
                tmp_document.append(segment)
        document = tmp_document
        mask_seq_num = int(round(len(document) * 0.3, 0))
        if sentence_selection_strategy == "random":
            mask_seq_list = random.sample(range(0, len(document) - 1), mask_seq_num)
        else:
            mask_seq_list = list(range(0, mask_seq_num))

        while i < len(document):
            segment = document[i]
            if i in mask_seq_list and len(tgt) + len(segment) < target_tgt_seq_length and len(src) + 1 < target_seq_length:
                tgt = tgt + segment
                src = src + [self.vocab.get(MASK_TOKEN)]
            elif i not in mask_seq_list and len(src) + len(segment) < target_seq_length:
                src = src + segment
            else:
                if len(tgt) > 0 and len(src) > 0:
                    instance = self.create_single_instance(src, tgt)
                    instances.append(instance)
                if i in mask_seq_list:
                    tgt = segment
                    src = [self.vocab.get(MASK_TOKEN)]
                else:
                    src = segment
                    tgt = []
            i += 1

        if len(tgt) > 0 and len(src) > 0:
            instance = self.create_single_instance(src, tgt)
            instances.append(instance)
        return instances


class BartDataset(BertDataset):

    def create_single_instance(self, src, tgt):
        src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
        tgt = [self.vocab.get(CLS_TOKEN)] + tgt + [self.vocab.get(SEP_TOKEN)]
        seg_pos = [len(src)]

        pad_num = self.seq_length - len(src)

        src = (src, pad_num)
        tgt = (tgt, pad_num)

        instance = (src, tgt, seg_pos)

        return instance

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        target_seq_length = self.seq_length - 2
        src = []
        tgt = []
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            if len(segment) > target_seq_length:
                i += 1
                continue
            if current_length + len(segment) < target_seq_length:
                current_chunk.append(segment)
                current_length += len(segment)
            else:
                shuf_chunk = current_chunk.copy()
                random.shuffle(shuf_chunk)
                for k in range(len(current_chunk)):
                    src = src + shuf_chunk[k]
                    tgt = tgt + current_chunk[k]
                instance = self.create_single_instance(src, tgt)
                instances.append(instance)
                current_length = len(segment)
                current_chunk = [segment]
                src = []
                tgt = []
            i += 1
        if len(current_chunk) > 0:
            shuf_chunk = current_chunk.copy()
            random.shuffle(shuf_chunk)
            for k in range(len(current_chunk)):
                src = src + shuf_chunk[k]
                tgt = tgt + current_chunk[k]
            instance = self.create_single_instance(src, tgt)
            instances.append(instance)

        return instances


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = line[1]
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [self.vocab.get(CLS_TOKEN)] + src
                    tgt = label
                    seg_pos = [len(src)]
                    if len(src) >= self.seq_length:
                        pad_num = 0
                        src = (src[:self.seq_length], pad_num)
                        seg_pos = [self.seq_length]
                    else:
                        pad_num = self.seq_length - len(src)
                        src = (src, pad_num)
                    pickle.dump((src, tgt, seg_pos), dataset_writer)
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_b)]
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    tgt = label
                    seg_pos = [len(src_a)] + [len(src_b)]

                    if len(src) >= self.seq_length:
                        pad_num = 0
                        src = (src[:self.seq_length], pad_num)
                        if len(src_a) >= self.seq_length:
                            seg_pos = [self.seq_length]
                        else:
                            seg_pos = [len(src_a)] + [self.seq_length - len(src_a)]
                    else:
                        pad_num = self.seq_length - len(src)
                        src = (src, pad_num)
                    pickle.dump((src, tgt, seg_pos), dataset_writer)

                else:
                    pass

                if pos >= end:
                    break

        dataset_writer.close()


class PrefixlmDataset(Dataset):

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_src))
                tgt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_tgt))
                src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
                tgt = tgt + [self.vocab.get(SEP_TOKEN)]
                seg_pos = [len(src)]

                if seg_pos[0] >= self.seq_length:
                    continue

                src = src + tgt
                tgt = [0] * (seg_pos[0] - 1) + tgt + [self.vocab.get(PAD_TOKEN)]
                seg_pos.append(len(src))
                src, tgt = src[:self.seq_length], tgt[:self.seq_length]
                pad_num = self.seq_length - len(src)
                src = (src, pad_num)
                if seg_pos[1] > self.seq_length:
                    seg_pos[1] = self.seq_length

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class ClsMlmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = line[1]
                    src = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)) + [self.vocab.get(SEP_TOKEN)]
                    tgt_cls = label
                    seg_pos = [len(src)]
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_a))
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_b))
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    tgt_cls = label
                    seg_pos = [len(src_a)] + [len(src_b)]
                else:
                    if pos >= end:
                        break
                    continue

                if len(src) >= self.seq_length:
                    pad_num = 0
                    src = (src[:self.seq_length], pad_num)
                    if len(seg_pos) == 1:
                        seg_pos = [self.seq_length]
                    else:
                        if len(src_a) >= self.seq_length:
                            seg_pos = [self.seq_length]
                        else:
                            seg_pos = [len(src_a)] + [self.seq_length - len(src_a)]
                else:
                    pad_num = self.seq_length - len(src)
                    src = (src, pad_num)

                if not self.dynamic_masking:
                    src_single, pad_num = src
                    src_single, tgt_mlm = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    src = (src_single, pad_num)
                    instance = (src, tgt_mlm, tgt_cls, seg_pos)
                else:
                    instance = (src, tgt_cls, seg_pos)

                pickle.dump(instance, dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()

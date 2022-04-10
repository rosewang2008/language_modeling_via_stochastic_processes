import json
import torch
import torch.utils.data as data
from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer


class BaseDataset(data.Dataset):

    def __init__(
            self,
            train,
            seed,
            config,
            all_dataset=None,
            tokenizer_name="GPT2",
            ):
        super().__init__()
        self.one_hot_labels = False
        self.train = train
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.config=config
        self.all_dataset = all_dataset

        self._load_data()

        self._set_section_names()

        self.split_pattern = '. '
        self._set_tokenizer()
        self._process_data()

        print("Examples: {}".format(self.processed_data[0]))
        print("Examples: {}".format(self.processed_data[10]))

    def _load_data(self):
        pass

    def _process_data(self):
        self.processed_data = []
        for doc_id in range(len(self.data)):
            doc_info = []
            sentence_counter = 0
            for section_id, section_name in enumerate(self.section_names):
                if section_id == 0: # abstract
                    try:
                        text = self.data[doc_id][section_name]
                    except:
                        break
                else:
                    doc = self.data[doc_id]
                    info = list(filter(
                        lambda x: x['sectionHeading']==section_name, doc['annotations']))[0]
                    text = doc['text'][info['begin']:info['begin']+info['length']]
                truncated = text.replace(".\n", ". ").split(self.split_pattern)[:-1]
                if len(truncated) == 0:
                    # NOTE: bug noted in log 04/25/2021; see pic for context
                    truncated = [text]

                for sentence_i, sentence in enumerate(truncated):
                    if not sentence:
                        continue

                    if (sentence_i == 0
                            and self.config.data_params.include_section_ids_in_tokenizer):
                        # adding " . " EOS
                        sentence = "{} {} . ".format(self.section_ids[section_id], sentence)
                    else:
                        sentence += " . "

                    sentence_info = {
                        "sentence": sentence,
                        "sentence_id": sentence_counter,
                        "section": section_name,
                        "section_id": section_id,
                        "doc_id": doc_id,
                    }
                    doc_info.append(sentence_info)
                    sentence_counter += 1

            # Track total number of sentences in a document
            for info in doc_info:
                info['total_doc_sentences'] = sentence_counter

            self.processed_data += doc_info

        # print examples
        print("Examples: {}".format(self.processed_data[0]))
        print("Examples: {}".format(self.processed_data[10]))

    def get_tokenized(self, sentence):
        tokenized = self.tokenizer(sentence, truncation=True, max_length=self.max_length)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        input_ids += [self.tokenizer.eos_token_id] * (self.max_length - len(input_ids))
        attention_mask += [0] * (self.max_length - len(attention_mask))

        return input_ids, attention_mask

    def _set_tokenizer(self):
        if self.tokenizer_name == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token_id
            self.max_length = 1024
        elif self.tokenizer_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.max_length = 512
        elif self.tokenizer_name == "ALBERT":
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.max_length = 512
        else:
            raise ValueError("Dont recognize name {}".format(self.tokenizer_name))
        # Add section ids
        self.cl_eos_str = " . "
        self.tokenizer.add_tokens(self.section_ids + [self.cl_eos_str])
        try:
            self.special_tokens = [_[0] for _ in self.tokenizer(self.section_ids)['input_ids']]
        except:
            print("no section ids: {}".format(self.section_ids))
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]

    def tokenize_caption(self, caption, device):
        if self.tokenizer_name == "GPT2":
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
            eos_input_ids = torch.tensor([[self.end_token]*input_ids.shape[0]])
            eos_attention = torch.tensor([[0]*input_ids.shape[0]])
            input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        elif self.tokenizer_name == "BERT" or self.tokenizer_name == "ALBERT":
            # Prepend [CLS] so I can use the first embedding
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
        return input_ids.to(device), attention_mask.to(device)


    def __len__(self):
        return len(self.processed_data) - 1

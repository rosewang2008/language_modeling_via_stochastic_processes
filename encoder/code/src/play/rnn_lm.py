import torch
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch import nn

class RNNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = None
    base_model_prefix = "rnnlm"
    is_parallelizable = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# class RNNLM(PreTrainedModel):
class RNNLM(RNNPreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config):
        super().__init__(config)
        # print(self.base_model_prefix)
        # print(self.base_model_prefix)

        # if hasattr(config, 'rnn_dropout_i'):
        #     self.dropout_i = config.rnn_dropout_i
        # else:
        #     self.dropout_i = 0.5
        #
        # if hasattr(config, 'rnn_dropout_h'):
        #     self.dropout_h = config.rnn_dropout_h
        # else:
        #     self.dropout_h = 0.5

        if hasattr(config, 'rnn_dropout'):
            self.dropout = config.rnn_dropout
        else:
            self.dropout = 0.5

        if hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            assert False, "need to have vocab_size"

        if hasattr(config, 'embed_dim'):
            self.embed_dim = config.embed_dim
        else:
            assert False, "need to have embed_dim"

        if hasattr(config, 'rnn_type'):
            self.rnn_type = config.rnn_type
        else:
            self.rnn_type = 'LSTM'

        if hasattr(config, 'num_layers'):
            self.num_layers = config.num_layers
        else:
            self.num_layers = 2


        if hasattr(config, 'hidden_dim'):
            self.hidden_dim = config.hidden_dim
        else:
            self.hidden_dim = 256


        self.drop = nn.Dropout(self.dropout)

        self.encoder = nn.Embedding(self.vocab_size, self.embed_dim)
        assert self.rnn_type in ['LSTM', 'LSTM2'], 'RNN type is not supported'
        if self.rnn_type == 'LSTM2':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif self.rnn_type == 'LSTM':
            self.rnns = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout)

        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size)
        self.pad_idx = self.vocab_size-1

        # print('before loading param: ')

        # print([x for x in list(self.named_parameters())[:2] ])


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "cell_state": past,
        }


    def forward(self, input_ids=None, attention_mask=None, labels=None, cell_state=None, **kwargs,):
        lengths =  (input_ids != self.pad_idx).sum(dim=-1) + 1
        # print(input_ids.shape)
        # print(input_ids)
        # print(lengths)
        input_emb = self.encoder(input_ids)
        input_emb = self.drop(input_emb)

        input_rnn=input_emb
        # input_rnn = torch.nn.utils.rnn.pack_padded_sequence(input_emb, lengths.cpu(), batch_first=True, enforce_sorted=False )
        output_rnn, (h, c) = self.rnns(input_rnn, cell_state)
        # output_rnn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(output_rnn, batch_first=True)

        output_rnn = self.drop(output_rnn)
        logits = self.decoder(output_rnn) #bsz, seqlen, vocab

        if labels is not None:
            shift_logits = logits[...,:-1,: ]
            shift_labels = labels[..., 1:]

            loss_fct = torch.nn.CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        else:
            loss = None
        # print(loss)

        if False:

            output = (logits,) +  (output_rnn,)
            return ((loss,) + output) if loss is not None else output


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=(h, c),
            hidden_states=output_rnn,
            attentions=None,
            cross_attentions=None,
        )




    # def forward(self, input, hidden, return_h=False):
    #     emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
    #     # emb = self.idrop(emb)
    #
    #     emb = self.lockdrop(emb, self.dropouti)
    #
    #     raw_output = emb
    #     new_hidden = []
    #     # raw_output, hidden = self.rnn(emb, hidden)
    #     raw_outputs = []
    #     outputs = []
    #     for l, rnn in enumerate(self.rnns):
    #         current_input = raw_output
    #         raw_output, new_h = rnn(raw_output, hidden[l])
    #         new_hidden.append(new_h)
    #         raw_outputs.append(raw_output)
    #         if l != self.nlayers - 1:
    #             # self.hdrop(raw_output)
    #             raw_output = self.lockdrop(raw_output, self.dropouth)
    #             outputs.append(raw_output)
    #     hidden = new_hidden
    #
    #     output = self.lockdrop(raw_output, self.dropout)
    #     outputs.append(output)
    #
    #     result = output.view(output.size(0) * output.size(1), output.size(2))
    #     if return_h:
    #         return result, hidden, raw_outputs, outputs
    #     return result, hidden


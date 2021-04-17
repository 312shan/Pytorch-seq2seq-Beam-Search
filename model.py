import math
import operator
import random
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from log import timeit

SOS_token = 2
EOS_token = 3
MAX_LENGTH = 50


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size  # 8014
        self.hidden_size = hidden_size  # 512
        self.embed_size = embed_size  # 256
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)  # [max_len, batch_size]
        outputs, hidden = self.gru(embedded, hidden)  # ([27, 32, 256],None)=>([27, 32, 1024],[4, 32, 512])
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])  # =>[27, 32, 512] + [27, 32, 512]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [32, 512]=>[32, 27, 512]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H] # [27, 32, 512]=>[32,27,512]
        attn_energies = self.score(h, encoder_outputs)  # =>[B*T]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [B*T]=>[B*1*T]

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*H] bmm [B*H*T]=>[B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size  # 256
        self.hidden_size = hidden_size  # 512
        self.output_size = output_size  # 10004
        self.n_layers = n_layers  # 1

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):  # 上一步的 output,上一步的 hidden_state
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N) # [32]=>[32, 256]=>[1, 32, 256]
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # [32, 512][27, 32, 512]=>[32, 1, 27]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N) # [32, 1, 27]bmm[32, 27, 512]=>[32,1,512]
        context = context.transpose(0, 1)  # (1,B,N) # [32, 1, 512]=>[1, 32, 512]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)  # [1, 32, 256] cat [1, 32, 512]=> [1, 32, 768]
        output, hidden = self.gru(rnn_input, last_hidden)  # in:[1, 32, 768],[1, 32, 512]=>[1, 32, 512],[1, 32, 512]
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))  # [32, 512] cat [32, 512] => [32, 512*2]
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights  # [32, 10004] [1, 32, 512] [32, 1, 27]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)  # output:[32, 10004] [1, 32, 512] [32, 1, 27]
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]  # 按照 dim=1 求解最大值和最大值索引,x[1] 得到的是最大值的索引=>top1.shape=32
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

    def decode(self, src, trg, method='beam-search'):
        encoder_output, hidden = self.encoder(src)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
        if method == 'beam-search':
            return self.beam_decode(trg, hidden, encoder_output)
        else:
            return self.greedy_decode(trg, hidden, encoder_output)

    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
        decoder_input = Variable(trg.data[0, :]).cuda()  # sos
        print(decoder_input.shape)
        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch

    @timeit
    def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        target_tensor = target_tensor.permute(1, 0)
        beam_width = 10
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_token]).cuda()

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng

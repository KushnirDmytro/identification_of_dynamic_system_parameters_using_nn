# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
import torch.nn.functional as F

import random
import utils
import smiles_processing
import time
import os
import logging
import datetime
import yaml
from models.model import EncoderRNN, DecoderRNN, AttnDecoderRNN


#ToDo: add beam search

random.seed(42)

CONFIG_FILENAME = 'experiments/lstm_2l_256_train_on_random_errors/error_corection_train.yaml'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_logger(config):
    if not os.path.exists(config['exp_path']):
        os.mkdir(config['exp_path'])
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.exists(config['exp_path'] + 'logs/'):
        os.mkdir(config['exp_path'] + 'logs/')
    logfile = f"{config['exp_path']}logs/exp_{st}.log"
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG)


def train(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, lang, config):
    batch_size = input_batch.size()[0]
    encoder_hidden = encoder.init_hidden(device, batch_size=batch_size)

    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_batch.size()[0]
    target_length = target_batch.size()[0]

    encoder_outputs = torch.zeros(config['max_len'], batch_size, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_batch[ei], encoder_hidden, batch_size=batch_size)
        # print(encoder_output.size())
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = torch.tensor([lang.sos_token] * batch_size, device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < config['teacher_forcing'] else False

    for di in range(target_length):
        if config['model']['attn']:
            decoder_output, decoder_hidden, attention_weights = decoder(
                decoder_input, decoder_hidden, encoder_outputs, batch_size=batch_size)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, batch_size=batch_size)
        loss += criterion(decoder_output, torch.squeeze(target_batch[di], dim=-1))
        if use_teacher_forcing:
            decoder_input = target_batch[di]  # Teacher forcing
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.squeeze(topi).detach()

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_epoch(train_loader, encoder, decoder, enc_optimizer, dec_optimizer, lang, config):

    start = time.time()
    batch_losses = []
    print_loss_total = 0
    criterion = nn.NLLLoss(ignore_index=lang.pad_token)
    iter = 1
    for batch_input, batch_target in train_loader:

        loss = train(batch_input, batch_target, encoder,
                     decoder, enc_optimizer, dec_optimizer, criterion, lang, config)
        print_loss_total += loss

        batch_losses.append(loss)

        if iter % config['print_every'] == 0:
            print('%s (%d %d%%) %.4f' % (utils.time_since(start, iter / len(train_loader)),
                                         iter, iter / len(train_loader) * 100, print_loss_total / config['print_every']))
            logging.debug('%s (%d %d%%) %.4f' % (utils.time_since(start, iter / len(train_loader)),
                                                 iter, iter / len(train_loader) * 100, print_loss_total / config['print_every']))
            print_loss_total = 0

        iter += 1

    return sum(batch_losses) / len(train_loader)


def evaluate(encoder, decoder, input_batch, target_batch, lang, config):
    criterion = nn.NLLLoss(ignore_index=lang.pad_token)
    loss = 0
    with torch.no_grad():
        batch_size = input_batch.size()[0]

        encoder_hidden = encoder.init_hidden(device, batch_size=batch_size)

        input_batch = input_batch.transpose(0, 1)
        target_batch = target_batch.transpose(0, 1)

        input_length = input_batch.size()[0]
        target_length = target_batch.size()[0]

        encoder_outputs = torch.zeros(config['max_len'], batch_size, encoder.hidden_size, device=device)

        output = torch.zeros(target_length, batch_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_batch[ei], encoder_hidden, batch_size=batch_size)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor([lang.sos_token] * batch_size, device=device)

        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(config['max_len'], batch_size, config['max_len'], device=device)

        for di in range(target_length):
            if config['model']['attn']:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, batch_size=batch_size)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, batch_size=batch_size)
            loss += criterion(decoder_output, torch.squeeze(target_batch[di], dim=-1))
            topv, topi = decoder_output.data.topk(1)
            output[di] = topi.squeeze().detach()
            decoder_input = topi.squeeze().detach()

        output = output.transpose(0, 1)
        decoded_smiles = []
        for di in range(output.size()[0]):
            decoded_smile = []
            for indx in output[di]:
                if indx.item() == lang.eos_token:
                    decoded_smile.append('EOS')
                    break
                decoded_smile.append(smiles_lang.indx2char[indx.item()])
            decoded_smiles.append(decoded_smile)
    print(decoder_attentions.shape)

    return decoded_smiles, decoder_attentions, loss.item() / target_length


def evaluate_on_many(data_loader, encoder, decoder, lang, show_results, config):
    all_loss = 0
    
    with open(f"{config['exp_path']}{config['results_f_name']}", 'w') as file:
        j = 0
        for input_batch, target_batch in data_loader:
            decoded_smiles, decoder_attentions, loss = evaluate(encoder, decoder, input_batch, target_batch, lang, config)

            all_loss += loss
            if show_results:
                for i in range(input_batch.size()[0]):
                    input_sm = ''.join([smiles_lang.indx2char[indx.item()] for indx in input_batch[i]]).split('EOS')[0]
                    target_sm = ''.join([smiles_lang.indx2char[indx.item()] for indx in target_batch[i]]).split('EOS')[0]
                    pred = ''.join(decoded_smiles[i]).split('EOS')[0]
                    print(f'With errors: {input_sm}')
                    print(f'Correct:     {target_sm}')
                    print(f'Predicted:   {pred}')
                    print('')

                    file.write(f'With errors: {input_sm}\n')
                    file.write(f'Correct:     {target_sm}\n')
                    file.write(f'Predicted:   {pred}\n')
                    file.write('\n')

                    utils.plot_attention([smiles_lang.indx2char[indx.item()] for indx in input_batch[i]], decoded_smiles[i], decoder_attentions[:, i, :], f"{config['exp_path']}attentions/attn_{j}.png", log=False)
                    # utils.plot_attention(input_sm, pred, decoder_attentions[:, i, :],
                    #                      f"{config['exp_path']}attentions/attn_log_{j}.png")
                j += 1

    return all_loss / len(data_loader)


if __name__ == '__main__':

    with open(CONFIG_FILENAME, 'r') as f:
        config = yaml.load(f)
    init_logger(config)
    smiles_lang = smiles_processing.Lang()

    pairs = smiles_processing.read_smiles(f"{config['data_path']}", pairs_to_read=config['pairs_to_read'])
    # with open('data/smote_new_from_500k.txt', 'r') as f:
    #     pairs = [[l.strip(), l.strip()] for l in f.readlines()]
    print(f'Read {len(pairs)} smile pairs')

    random.shuffle(pairs)
    train_pairs = pairs[int(len(pairs) * config['test_size']):]

    val_pairs = [random.choice(train_pairs) for i in range(round(len(train_pairs) * config['test_size']))]
    test_pairs = pairs[:int(len(pairs) * config['test_size'])]

    train_pairs = [smiles_processing.tensors_from_pair(pair, smiles_lang, config, device) for pair in train_pairs]
    val_pairs = [smiles_processing.tensors_from_pair(pair, smiles_lang, config, device) for pair in val_pairs]
    test_pairs = [smiles_processing.tensors_from_pair(pair, smiles_lang, config, device) for pair in test_pairs]

    train_loader = torch.utils.data.DataLoader(train_pairs, batch_size=config['batch_size'])   #[2, batch_size, max_len, 1]
    val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=config['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=config['batch_size'], shuffle=False)

    encoder1 = EncoderRNN(config['model']['r_cell'], smiles_lang.n_chars, config['model']['hidden_size'],
                          n_layers=config['model']['n_layers']).to(device)
    if config['model']['attn']:
        decoder1 = AttnDecoderRNN(config['model']['r_cell'], config['model']['hidden_size'], smiles_lang.n_chars,
                                  n_layers=config['model']['n_layers'], max_length=config['max_len'],
                                  dropout_emb=config['model']['dropout_emb']).to(device)
    else:
        decoder1 = DecoderRNN(config['model']['r_cell'], config['model']['hidden_size'], smiles_lang.n_chars,
                              n_layers=config['model']['n_layers']).to(device)

    enc_optimizer = optim.SGD(encoder1.parameters(), lr=config['init_lr'])
    dec_optimizer = optim.SGD(decoder1.parameters(), lr=config['init_lr'])

    print(f'Trainable params in encoder: {encoder1.get_num_params()}')
    print(f'Trainable params in decoder: {decoder1.get_num_params()}')
    best_score = 1000
    if config['resume_training']:
        checkpoint = torch.load(config['exp_path'] + 'encoder_final.pth', map_location=lambda storage, loc: storage)
        encoder1.load_state_dict(checkpoint['model_state_dict'])
        enc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint = torch.load(config['exp_path'] + 'decoder_final.pth', map_location=lambda storage, loc: storage)
        decoder1.load_state_dict(checkpoint['model_state_dict'])
        dec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #best_score = checkpoint['tr_loss']

    if config['scheduler']['name'] == 'reduceonplateau':
        enc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(enc_optimizer, patience=config['scheduler']['patience'], verbose=True)
        dec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dec_optimizer,  patience=config['scheduler']['patience'], verbose=True)

    else:
        enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(enc_optimizer, T_max=18)
        dec_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dec_optimizer, T_max=18)

    print(f'Training on {len(train_pairs)} smiles pairs, validating on {len(test_pairs)} smiles pairs')
    logging.debug(f'Training on {len(train_pairs)} smiles pairs, validating on {len(test_pairs)} smiles pairs')

    if config['test_only']:
        test_loss = evaluate_on_many(test_loader, encoder1.eval(), decoder1.eval(), smiles_lang, True, config)
        print(f'Test loss: {test_loss}')

    else:
        tr_loss, val_loss = 5.0, 5.0
        tr_losses, val_losses = [], []

        for epoch in range(config['n_epochs']):
            print(f'EPOCH {epoch}')
            logging.debug(f'EPOCH {epoch}')

            tr_loss = train_epoch(train_loader, encoder1, decoder1, enc_optimizer, dec_optimizer, smiles_lang, config)

            evaluate_on_many(val_loader, encoder1, decoder1, smiles_lang, epoch % 10 == 0, config)
            val_loss = evaluate_on_many(test_loader, encoder1, decoder1, smiles_lang, epoch % 10 == 0, config)
            print(f'Train loss {tr_loss}, Val loss {val_loss}')
            logging.debug(f'Train loss {tr_loss}, Val loss {val_loss}')
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)

            enc_scheduler.step(tr_loss)
            dec_scheduler.step(tr_loss)

            if tr_loss < best_score:
                torch.save({
                    'model_state_dict': encoder1.state_dict(),
                    'optimizer_state_dict': enc_optimizer.state_dict(),
                    'tr_loss': tr_loss,
                    'val_loss': val_loss

                }, config['exp_path'] + 'encoder_final.pth')

                torch.save({
                    'model_state_dict': decoder1.state_dict(),
                    'optimizer_state_dict': dec_optimizer.state_dict(),
                    'tr_loss': tr_loss,
                    'val_loss': val_loss

                }, config['exp_path'] + 'decoder_final.pth')
                best_score = tr_loss
            utils.plot_losses(tr_losses, val_losses, epoch+1, config)




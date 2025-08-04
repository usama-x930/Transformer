import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import casual_mask, MyDataset
from model import BuildTransformer

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weight_file_path
from tqdm import tqdm
import warnings



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute encoder output and reuse it for every token we get from the decoder

    encoder_output = model.encode(source, source_mask)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_samples=2):
    model.eval()
    count = 0
    source_sentences = []
    expected = []
    predicted = []

    console_widht = 80


    with torch.no_grad():
        for batch in validation_ds:
            if count>=num_samples:
                break
            count += 1

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1




def build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        # tokenizer = Tokenizer(WordLevel(unk_token=['UNK']))
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_dataset(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tok_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tok_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size

    # train_size, val_size = random_split(ds_raw, [train_size, val_size])


    # train_ds = MyDataset(train_size, tok_src, tok_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    # val_ds = MyDataset(val_size, tok_src, tok_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])
    train_ds = MyDataset(train_ds_raw, tok_src, tok_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = MyDataset(val_ds_raw, tok_src, tok_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tok_src.encode(item['translation'][config['lang_src']])
        tgt_ids = tok_tgt.encode(item['translation'][config['lang_tgt']])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print("Max Length of SOurce Sentence: ", max_len_src)
    print("Max Length of Target Sentence: ", max_len_tgt)


    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tok_src, tok_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    mod = BuildTransformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return mod

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print(f'Using Device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, tok_src, tok_tgt = get_dataset(config)
    model = get_model(config, tok_src.get_vocab_size(), tok_tgt.get_vocab_size()).to(device)

    #Tensorboarddd
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    init_epochs = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        init_epochs = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tok_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)



    for epoch in range(init_epochs, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)  

            # encoder_out = model.encode(encoder_input, encoder_mask)
            # decoder_out = model.decode(decoder_input, encoder_mask, decoder_input, decoder_mask)

            encoder_out = model.encode(encoder_input, encoder_mask)
            decoder_out = model.decode(encoder_out, encoder_mask, decoder_input, decoder_mask)

            proj_output = model.project(decoder_out)
            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tok_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():0.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weight_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()

    train_model(config)







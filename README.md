# 🖼️ Flickr8k Image Caption Generator

An image captioning system that generates natural-language descriptions of photos, using a CNN encoder + LSTM decoder architecture — the same foundational design behind modern vision-language models.

**🔗 Live demo:** https://flickr8-caption-generator-jgn5dhpgxqzekczgbjvhfx.streamlit.app/

## What it does

Given a photo, the model generates a descriptive sentence about what's in it. This bridges computer vision (understanding the image) and NLP (producing coherent language) — the model has to both "see" and "describe," not just classify.

## Architecture — "Show and Tell"

```
Image → ResNet50 (frozen) → 2048-dim features → Linear + BatchNorm → 256-dim embedding
                                                                            ↓
                                                              LSTM decoder → word-by-word caption
```

**Encoder (CNN):**
- Pre-trained **ResNet50**, with the final classification layer removed
- Outputs a 2048-dimensional feature vector representing the image's visual content
- Projected into a 256-dimensional embedding space via a Linear layer + BatchNorm

**Decoder (LSTM):**
- Word embedding layer converts tokens into 256-dimensional vectors
- LSTM takes the image features as its initial state, then processes the caption token-by-token
- A final Linear layer maps LSTM hidden states to vocabulary-sized predictions for the next word

## Dataset

**Flickr8k** — a standard academic benchmark for image captioning.
- 8,000 real-world photographs
- 5 human-written captions per image (40,000 total image-caption pairs)
- Vocabulary built with a minimum word-frequency threshold of 2 (~5,240 words)

## Training details

- Images resized to 224×224, normalized; captions tokenized with NLTK
- `<SOS>`, `<EOS>`, `<PAD>`, `<UNK>` special tokens handle sequence generation and variable-length captions
- **Transfer learning**: ResNet50 encoder weights frozen; only the decoder LSTM and the encoder's projection layers were trained
- 15 epochs, Adam optimizer, cross-entropy loss

## Results

*(Fill this in with your actual numbers/examples before publishing — see the note below.)*

| Metric | Score |
|---|---|
| BLEU-1 | *[add your score]* |
| BLEU-4 | *[add your score]* |
| Training loss (final epoch) | *[add your value]* |

**Example captions generated:**

| Image | Generated caption |
|---|---|
| *[add a real example]* | *"a dog is running through the grass"* |
| *[add a real example]* | *"a group of people sitting at a table"* |

**Honest note on scope:** BLEU scores for Flickr8k models in the literature typically range from ~55-65 (BLEU-1) down to ~15-25 (BLEU-4) depending on architecture — this isn't a state-of-the-art result, it's a correct, working implementation of the classic encoder-decoder captioning approach. If your BLEU scores land in a similar range, that's expected and fine; report the real numbers rather than omitting them.

## Tech stack

- **Deep learning:** PyTorch, Torchvision
- **NLP:** NLTK (tokenization)
- **Web app:** Streamlit
- **Model hosting:** Hugging Face Hub (keeps the GitHub repo lightweight — weights download automatically when the app runs)

## Running it locally

```bash
git clone https://github.com/kaleemgujjar07/flickr8-caption-generator.git
cd flickr8-caption-generator
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## Project structure

```
flickr8-caption-generator/
├── app.py              # Streamlit app + caption generation logic
├── vocab.json          # Vocabulary mappings (stoi/itos)
├── requirements.txt
└── README.me

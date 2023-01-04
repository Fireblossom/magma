# Changxu's babbling idea about scientific figures captioning

## Where do the idea comes from?

After I tried the captioning model for photos, I have been trying to find a model that can do the same thing for scientific paper figures.

Unfortunately, there seems to be none, and the existing similar models are all photo-specific.
Also, since they are generated very unreliably (like ChatGPT), especially in the scientific domian it is not acceptable.

In the paper recommended to me by Wolfgang, we can have the model generate many relevant and irrelevant sentences, and then we label it and boost the model.

One day Sherry reminded me that there was a workshop on document understanding. 
At that time I was thinking of writing a GPT-style model that would give hints about sources in sentence keywords (in the form of keyword#1234.5678 with arxiv id).
This idea was later scrapped because add the id suffix meant we had to add millions of new tokens in the tokenizer (it originally had only ~50,000)

Then one day Elena showed me a new model from Facebook, which was trained from many papers. 
It had the ability to infer the reference source from the article.
That's the main support of my idea so far.

Also I read some papers about scientific paper figures understanding, such as ChartOCR.
They focus on extracting the text in the image, such as the range of the data axis.
So I did OCR on all the figures, and then replaced the model's CNN image encoder with one that encodes both text and image inputs,
which is LayoutLMv3 so far (but did not work..)

## Title: TBD

### Hypothsis
Figure captioning task benefited from transformers based image encoder and reference based feedback training process. 

### What I did so far
- Base on MAGMA framework
- Replace the LM model
- Replace the image encoder
- OCR the SciCap dataset
- Check dataset
- Modify the Dataloader that encoding the text information in figures
- Add paper title information to captions
- Test forward pass
- Test generation
- Check pretrained model loading
- Fix training loop
- Reduce the seq_len to less model GRAM usage
- Test CUDA and checkpointing (59GB GRAM and 98GB checkpointing)

### issues, bugs and TODO
- If a transformer model like ViT and LayoutLMv3 is used as a image encoder, the training loss will drop to 0.0 after about 40 steps. (seems fixed by using full layoutlmv3 output)
- Only one vector for image encoding may not be enough. (now output vector length is 300+)
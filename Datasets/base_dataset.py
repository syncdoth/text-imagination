import re
import os

from tensorflow.keras.preprocessing.text import Tokenizer
import nltk

from utility import preprocess_sentence, is_noun, get_wordnet_repr


class Dataset:
    def __init__(self,
                 mode="train",
                 data_dir="data",
                 tag_func=nltk.pos_tag_sents,
                 input_tokenizer=None,
                 target_tokenizer=None):
        assert tag_func is not None

        self.tag_func = tag_func
        self.data_dir = data_dir
        self.mode = mode

        if input_tokenizer is not None:
            self.input_tokenizer = input_tokenizer
        else:
            assert mode == "train"
            self.input_tokenizer = Tokenizer(oov_token="[UNK]")

        if target_tokenizer is not None:
            self.target_tokenizer = target_tokenizer
        else:
            assert mode == "train"
            self.target_tokenizer = Tokenizer()

    def get_corpus(self):
        """List of Dialogues. Each dialogue is list of sentences (question and answer
        are separate sentences). Each sentece is tokenized.
        """
        corpus = []
        with open(os.path.join(self.data_dir, f"dialog_{self.mode}.txt")) as f:
            for line in f:
                line = preprocess_sentence(line)

                # simple heuristic to avoid mistake in pos tagging
                line = line.replace("yes", "Yes")

                line = line.replace("</q>", "?")
                line = line.replace("</a>", ".")

                # split by sentence
                line = re.split("<[qa]>", line)
                # drop empty strings created from split, and tokenize
                line = [nltk.word_tokenize(s) for s in line if s]
                corpus.append(line)

        print("finished loading corpus")
        return corpus

    def get_description(self):
        """List of descriptions. The descriptions are tokenized."""
        with open(os.path.join(self.data_dir, f"desc_{self.mode}.txt")) as f:
            descriptions = []
            for line in f:
                line = preprocess_sentence(line)
                line = nltk.word_tokenize(line)
                descriptions.append(line)

        print("finished loading descriptions")
        return descriptions

    def extract_nouns(self, tagged_sentence):
        nouns = []
        for w in tagged_sentence:
            if not is_noun(*w):
                continue
            synset = get_wordnet_repr(tagged_sentence, w[0])
            if synset is None:
                continue
            nouns.append(synset.name())
        return nouns

    def extract_context(self, corpus):
        """Context from dialogues.
        shape: [number of dialogue, number of context]"""
        contexts = []
        for i, dialog in enumerate(corpus):
            tagged_dialog = self.tag_func(dialog)
            context = set()
            for sent in tagged_dialog:
                nouns = self.extract_nouns(sent)
                context.update(nouns)
            contexts.append(list(context))

        print("finished extracting contexts")
        return contexts

    def extract_target(self, description, contexts):
        """Extract ground truth OOC.

        NOTE: There are quite a lot of instances there is no OOC.
        """
        targets = []
        tagged_description = self.tag_func(description)
        for i, desc in enumerate(tagged_description):
            nouns = self.extract_nouns(desc)
            ooc = set(nouns) - set(contexts[i])
            targets.append(list(ooc))

        print("finished extracting targets (OOCs)")
        return targets

    def fit_tokenizer(self, input_sents, target):
        self.input_tokenizer.fit_on_texts(input_sents)
        self.target_tokenizer.fit_on_texts(target)

    def get_data_target(self, whole_dialog=False, filter_head_words=False, maxlen=300):
        raise NotImplementedError

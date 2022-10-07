import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class S2VT(torch.nn.Module):
    def __init__(self, vocab_size, frame_dim = 4096, hidden_dim = 1000, steps = 80, video_embed_size = 500, word_embed_size = 500):
        super().__init__()
        self.step = steps
        self.vocab_size = vocab_size
        self.video_embed_size = video_embed_size
        self.word_embed_size = word_embed_size
        self.hidden_dim = hidden_dim

        self.word_embedding = torch.nn.Embedding(vocab_size, word_embed_size)
        self.video_embed = torch.nn.Linear(frame_dim, video_embed_size)
        self.lstm1 = torch.nn.LSTM(input_size=self.video_embed_size, hidden_size=self.hidden_dim)
        self.word_out = torch.nn.Linear(hidden_dim, vocab_size)
        self.lstm2 = torch.nn.LSTM(input_size=self.word_embed_size + self.hidden_dim, hidden_size=self.hidden_dim)

    def forward(self, video, caption=None):
        batch, _, _ = video.shape
        video_embeds = self.video_embed(video)

        if self.training:
            word_vec = self.word_embedding(caption)
            padding_lstm1 = torch.zeros(batch, self.step, self.video_embed_size).to(device)
            padding_lstm2 = torch.zeros(batch, self.step, self.word_embed_size).to(device)

            lstm1_input = torch.cat((video_embeds, padding_lstm1), dim=1).to(device)
            lstm1_output, _ = self.lstm1(lstm1_input)
 
            lstm2_input = torch.cat((padding_lstm2, word_vec), dim=1).to(device)
            lstm2_input = torch.cat((lstm1_output, lstm2_input), dim=2).to(device)
            lstm2_output, _ = self.lstm2(lstm2_input)
            lstm2_output_cap = lstm2_output[:, self.step:, :]

            word_tensor = self.word_out(lstm2_output_cap)
            word_one_hot = torch.argmax(word_tensor, dim = 2)

            return word_one_hot

        else:
            padding_lstm1 = torch.zeros(batch, self.video_embed_size).to(device)
            padding_lstm2 = torch.zeros(batch, self.word_embed_size).to(device)

            for step in range (self.step):
                if step == 0:
                    output1, (hidden1, cell1) = self.lstm1(video_embeds[:, step, :])
                    output2, (hidden2, cell2) = self.lstm2(torch.cat((output1, padding_lstm2), 1))
                else:
                    output1, (hidden1, cell1) = self.lstm1(video_embeds[:, step, :], (hidden1, cell1))
                    output2, (hidden2, cell2) = self.lstm2(torch.cat((output1, padding_lstm2), 1), (hidden2, cell2))

            previous_word = self.word_embedding(torch.ones((5)).type(torch.LongTensor).to(device))

            for step in range (self.step):
                if step == 0:
                    output1, (hidden1, cell1) = self.lstm1(padding_lstm1)
                    output2, (hidden2, cell2) = self.lstm2(torch.cat((output1, previous_word), 1))
                else:
                    output1, (hidden1, cell1) = self.lstm1(padding_lstm1, (hidden1, cell1))
                    output2, (hidden2, cell2) = self.lstm2(torch.cat((output1, previous_word), 1), (hidden2, cell2))

                word_tensor = self.word_out(output2)
                word_one_hot = torch.argmax(word_tensor, dim = 1)
                word_one_hot_out = torch.argmax(word_tensor, dim = 1).reshape(5, -1)
                previous_word = self.word_embedding(word_one_hot)

                if step == 0:
                    cap = word_one_hot_out
                else:
                    cap = torch.cat((cap, word_one_hot_out),1)

            return cap
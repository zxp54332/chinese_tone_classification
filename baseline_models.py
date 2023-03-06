import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, num_classes=5)

    def forward(self, x):
        return self.model(x)

    def freeze_params(self):
        # except last layer
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False


class CNNet(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2688, 50)
        self.fc2 = nn.Linear(50, num_labels)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return x


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, sparse=False, padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class CustomTextCNN(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, num_class=5):
        super().__init__()
        self.cnn = CNNet()
        self.text_model = TextClassificationModel(
            vocab_size=vocab_size, embed_dim=text_embed_dim, num_class=num_class
        )
        self.text_norm = nn.LayerNorm(num_class)
        self.cnn_norm = nn.LayerNorm(num_class)
        self.fc = nn.Linear(5, 5)

    def forward(self, audio_arrays, ids, offsets):
        cnn_logits = self.cnn(audio_arrays)
        text_logits = self.text_model(ids, offsets=offsets)
        cnn_logits = F.dropout(self.cnn_norm(cnn_logits), training=self.training)
        text_logits = F.dropout(self.text_norm(text_logits), training=self.training)
        return F.relu(self.fc(cnn_logits + text_logits))


class CustomTextCNNCat(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, num_class=5):
        super().__init__()
        self.cnn = CNNet()
        self.text_model = TextClassificationModel(
            vocab_size=vocab_size, embed_dim=text_embed_dim, num_class=num_class
        )
        self.text_norm = nn.LayerNorm(num_class)
        self.cnn_norm = nn.LayerNorm(num_class)
        self.fc = nn.Linear(10, 5)

    def forward(self, audio_arrays, ids, offsets):
        cnn_logits = self.cnn(audio_arrays)
        text_logits = self.text_model(ids, offsets=offsets)
        cnn_logits = F.dropout(self.cnn_norm(cnn_logits), training=self.training)
        text_logits = F.dropout(self.text_norm(text_logits), training=self.training)
        cat_logits = torch.cat((text_logits, cnn_logits), dim=-1)
        return F.relu(self.fc(cat_logits))


class CustomResNetText(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, num_class=5):
        super().__init__()
        self.cnn = timm.create_model("resnet18", pretrained=True, num_classes=5)
        self.text_model = TextClassificationModel(
            vocab_size=vocab_size, embed_dim=text_embed_dim, num_class=num_class
        )
        self.text_norm = nn.LayerNorm(num_class)
        self.cnn_norm = nn.LayerNorm(num_class)
        self.fc = nn.Linear(5, 5)

    def forward(self, audio_arrays, ids, offsets):
        cnn_logits = self.cnn(audio_arrays)
        text_logits = self.text_model(ids, offsets=offsets)
        cnn_logits = F.dropout(self.cnn_norm(cnn_logits), training=self.training)
        text_logits = F.dropout(self.text_norm(text_logits), training=self.training)
        return F.relu(self.fc(cnn_logits + text_logits))

    def freeze_params(self):
        # except last layer
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False


class CustomResNetTextCat(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, num_class=5):
        super().__init__()
        self.cnn = timm.create_model("resnet18", pretrained=True, num_classes=5)
        self.text_model = TextClassificationModel(
            vocab_size=vocab_size, embed_dim=text_embed_dim, num_class=num_class
        )
        self.text_norm = nn.LayerNorm(num_class)
        self.cnn_norm = nn.LayerNorm(num_class)
        self.fc = nn.Linear(10, 5)

    def forward(self, audio_arrays, ids, offsets):
        cnn_logits = self.cnn(audio_arrays)
        text_logits = self.text_model(ids, offsets=offsets)
        cnn_logits = F.dropout(self.cnn_norm(cnn_logits), training=self.training)
        text_logits = F.dropout(self.text_norm(text_logits), training=self.training)
        cat_logits = torch.cat((text_logits, cnn_logits), dim=-1)
        return F.relu(self.fc(cat_logits))

    def freeze_params(self):
        # except last layer
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image
from typing import List
from transformers import BertTokenizer, BertModel
from datetime import datetime


class COCO_Dataset(Dataset):
    def __init__(self, root_dir='/data0/mscoco/', set_name='train2014', split='TRAIN'):
        super().__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'captions_'+self.set_name + '.json' ))

        whole_image_ids = self.coco.getImgIds()
        self.image_ids = []

        # remove not captioned image idx
        self.no_anno_list = []
        self.anno_ids = []
        for idx in whole_image_ids:
            annotation_ids = self.coco.getAnnIds(imgIds=idx)
            self.anno_ids += annotation_ids
            if len(annotation_ids) == 0:
                self.no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)
        # print(len(self.image_ids))
        # print(min(self.image_ids))
        # print(len(self.anno_ids))
        # print(min(self.anno_ids))
        self.split = split

    def __getitem__(self, idx):
        caption, img_idx = self.load_annotations(idx)
        image, (w, h) = self.load_image(img_idx)

        image = image.resize((256,256))
        tf = transforms.ToTensor()
        image = tf(image)
        return image, caption

    def __len__(self):
        return len(self.anno_ids)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(image_index)[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        image = Image.open(path).convert('RGB')
        # image.save(f'test{image_index}.png')
        return image, (image_info['width'], image_info['height'])

    def load_annotations(self, anno_index):
        annotation_info = self.coco.loadAnns(self.anno_ids[anno_index])[0]
        caption_text = annotation_info['caption']
        img_idx = annotation_info['image_id']
        return caption_text, img_idx

    # def load_annotations(self, image_index):
    #     # to be revised -> annotation id를 받아서 caption 추출
    #     annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index] )
    #     coco_annotations = self.coco.loadAnns(annotations_ids)
    #     # to be revised.
    #     for idx, a in enumerate(coco_annotations):
    #         caption_text = a['caption']
    #     return caption_text


class BertEmbeddings():
    def __init__(self):
        self._cls_token = '[CLS]'
        self._sep_token = '[SEP]'
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bert_layer = BertModel.from_pretrained('bert-base-uncased')
        # _bert_layer = torch.nn.DataParallel(_bert_layer)
        # _bert_layer = _bert_layer.cuda()
        self._bert_layer.eval()

    def get_bert_for_caption(self, captions: List[str], max_text_length: int = 17):
        """Returns BERT pooled and sequence outputs for a given list of captions."""
        all_tokens = []
        all_input_mask = []
        for text in captions:
            truncated_tokens = self._tokenizer.tokenize(text)[:max_text_length - 2]
            tokens = [self._cls_token] + truncated_tokens + [self._sep_token]
            tokens = self._tokenizer.convert_tokens_to_ids(tokens)
            num_padding = max_text_length - len(tokens)
            input_mask = [1] * len(tokens)
            tokens = tokens + [0] * num_padding
            input_mask = input_mask + [0] * num_padding
            all_tokens.append(np.asarray(tokens, np.int32))
            all_input_mask.append(np.asarray(input_mask, np.int32))

        ids = torch.tensor(all_tokens)
        input_mask = torch.tensor(all_input_mask, )
        segment_ids = torch.zeros_like(ids)

        # print(ids)
        # print(input_mask)
        # print(segment_ids)

        with torch.no_grad():
            token_embedding = self._bert_layer(ids, attention_mask=input_mask, token_type_ids=segment_ids)
            token_embedding = token_embedding[0]
            max_len = torch.sum(input_mask, dim=1)
            sent_embedding = torch.sum(token_embedding, dim=1) / max_len[:, None]
        return token_embedding, sent_embedding, max_len


set_list = ['train2014', 'val2014']
start_time = datetime.now()
bert = BertEmbeddings()

for set_name in set_list:
    print('\n------------------------------------------------------')
    test_class = COCO_Dataset(set_name=set_name)
    test_loader = DataLoader(test_class,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    print(f'Testing with {test_class.__len__()} inputs.')
    for idx, (images, captions) in enumerate(test_loader):
        # print(images)
        # print(captions)
        t1, t2, t3 = bert.get_bert_for_caption(captions)
        print(f'\r{(idx + 1) * 64}/{test_class.__len__()}\t{round(((idx + 1) * 64 / test_class.__len__()) * 100, 1)}%',
              end='')
        # if idx == 10:
        #      break

end_time = datetime.now()

print('\nDone. It takes', end_time - start_time)

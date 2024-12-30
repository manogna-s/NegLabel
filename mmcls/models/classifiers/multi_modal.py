import os
from matplotlib import pyplot as plt
import torch
from torch import nn
import xml.etree.ElementTree as ET
import random
import numpy as np

import clip
import time
from torchvision.datasets import CIFAR100, CIFAR10
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
from .class_names import CLASS_NAME, prompt_templates, adj_prompts_templetes
import utils

@CLASSIFIERS.register_module()
class CLIPScalableClassifier(BaseClassifier):

    def __init__(self,
                 arch='ViT-B/16',
                 train_dataset=None,
                 wordnet_database=None,
                 txt_exclude=None,
                 neg_subsample=-1,
                 neg_topk=10000,
                 emb_batchsize=1000,
                 init_cfg=None,
                 prompt_idx_pos=None,
                 prompt_idx_neg=None,
                 exclude_super_class=None,
                 dump_neg=False,
                 cls_mode=False,
                 load_dump_neg=False,
                 pencentile=1,
                 pos_topk=None,):
        super(CLIPScalableClassifier, self).__init__(init_cfg)
        self.local_rank = os.environ['LOCAL_RANK']
        self.device = "cuda:{}".format(self.local_rank)

        self.clip_model, _ = clip.load(arch, self.device, jit=False)
        self.clip_model.eval()
        self.model = self.clip_model
        self.cls_mode = cls_mode
        
        #------------Interpret clip----------------
        self.activations = []
        self.feat_type = 'last_layer'
        if self.feat_type == 'last_layer':
            handlers = [self.model.visual.transformer.register_forward_hook(self.save_activation)]
        else:      # keys
            handlers = [self.model.visual.transformer.resblocks[-1].attn.register_forward_hook(self.get_hook(self.model.visual.transformer.resblocks[-1].attn.in_proj_weight))]
        
        self.batch_idx = 0 # for visualization of results
        #------------Interpret clip----------------


        if prompt_idx_pos is None:
            prompt_idx_pos = -1
        if exclude_super_class is not None:
            class_name=CLASS_NAME[train_dataset][exclude_super_class]
        else:
            class_name=CLASS_NAME[train_dataset]
        prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
        text_inputs_pos = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(self.device)

        with torch.no_grad():
            self.text_features_pos = self.clip_model.encode_text(text_inputs_pos).to(torch.float32)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

        if not load_dump_neg or not os.path.exists('/data/neg_label/neg_embedding/neg_dump.pth'):
            txtfiles = os.listdir(wordnet_database)
            if txt_exclude:
                file_names = txt_exclude.split(',')
                for file in file_names:
                    txtfiles.remove(file)
            words_noun = []
            words_adj = []
            if prompt_idx_neg is None:
                prompt_idx_neg = -1
            prompt_templete = dict(
                    adj='This is a {} photo',
                    noun=prompt_templates[prompt_idx_neg],
                )
            dedup = dict()
            for file in txtfiles:
                filetype = file.split('.')[0]
                if filetype not in prompt_templete:
                    continue
                with open(os.path.join(wordnet_database, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() in dedup:
                            continue
                        dedup[line.strip()] = None
                        if filetype == 'noun':
                            if pos_topk is not None:
                                if line.strip() in class_name:
                                    continue
                            words_noun.append(prompt_templete[filetype].format(line.strip()))
                        elif filetype == 'adj':
                            words_adj.append(prompt_templete[filetype].format(line.strip()))
                        else:
                            raise TypeError

            if neg_subsample > 0:
                random.seed(42)
                words_noun = random.sample(words_noun, neg_subsample)

            text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in words_noun]).to(self.device)
            text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in words_adj]).to(self.device)
            text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
            noun_length=len(text_inputs_neg_noun)
            adj_length = len(text_inputs_neg_adj)


            with torch.no_grad():
                self.text_features_neg = []
                for i in range(0, len(text_inputs_neg), emb_batchsize):
                    x = self.clip_model.encode_text(text_inputs_neg[i : i + emb_batchsize])
                    self.text_features_neg.append(x)
                self.text_features_neg = torch.cat(self.text_features_neg, dim=0)
                self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)
                if dump_neg:
                    tmp = self.text_features_neg.cpu()
                    dump_dict=dict(neg_emb=tmp, noun_length=noun_length, adj_length=adj_length)
                    os.makedirs('/data/neg_label/neg_embedding', exist_ok=True)
                    torch.save(dump_dict, '/data/neg_label/neg_embedding/neg_dump.pth')
                    assert False
        else:
            tic = time.time()
            dump_dict = torch.load('/data/neg_label/neg_embedding/neg_dump.pth')
            self.text_features_neg = dump_dict['neg_emb'].to(self.device)
            toc = time.time()
            print('Successfully load the negative embedding and cost {}s.'.format(toc-tic))
            noun_length = dump_dict['noun_length']
            adj_length = dump_dict['adj_length']

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

            if pos_topk is not None:
                pos_mask = torch.zeros(len(self.text_features_neg), dtype=torch.bool, device=self.device)
                for i in range(self.text_features_pos.shape[0]):
                    sim = self.text_features_pos[i].unsqueeze(0) @ self.text_features_neg.T
                    _, ind = torch.topk(sim.squeeze(0), k=pos_topk)
                    pos_mask[ind] = 1
                self.text_features_pos = torch.cat([self.text_features_pos, self.text_features_neg[pos_mask]])

            neg_sim = []
            for i in range(0, noun_length+adj_length, emb_batchsize):
                tmp = self.text_features_neg[i: i + emb_batchsize] @ self.text_features_pos.T
                tmp = tmp.to(torch.float32)
                sim = torch.quantile(tmp, q=pencentile, dim=-1)
                neg_sim.append(sim)
            neg_sim = torch.cat(neg_sim, dim=0)
            neg_sim_noun = neg_sim[:noun_length]
            neg_sim_adj = neg_sim[noun_length:]
            text_features_neg_noun = self.text_features_neg[:noun_length]
            text_features_neg_adj = self.text_features_neg[noun_length:]

            ind_noun = torch.argsort(neg_sim_noun)
            ind_adj = torch.argsort(neg_sim_adj)


            self.text_features_neg = torch.cat([text_features_neg_noun[ind_noun[0:int(len(ind_noun)*neg_topk)]],
                                                text_features_neg_adj[ind_adj[0:int(len(ind_adj)*neg_topk)]]], dim=0)


            self.adj_start_idx = int(len(ind_noun) * neg_topk)

            ## If you want to dump the selected negative labels (with prompt), please uncomment these lines.
            # with open("selected_neg_labels.txt", "w") as f:
            #     for i in ind_noun[0:int(len(ind_noun)*neg_topk)]:
            #         f.write("{}\n".format(words_noun[i]))
            #     for j in ind_adj[0:int(len(ind_adj)*neg_topk)]:
            #         f.write("{}\n".format(words_adj[j]))
    
    
    def get_hook(self, weights):

        def inner_hook(module, input, output):
            inp = input[1].transpose(1,0)  # all inputs to attn are same, does not matter. Can also do input[0] or input[2] 
            output_qkv = torch.matmul(inp, weights.transpose(0,1)).detach()  # (B, 197, dim * 3)
            B, T, _ = output_qkv.shape
            num_heads = self.model.transformer.resblocks[-1].attn.num_heads
            output_qkv = output_qkv.reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            k_feats = output_qkv[1].transpose(1, 2).reshape(B, T, -1)
            self.activations.append(k_feats)
        
        return inner_hook
    
    def save_activation(self, module, input, output):
        self.activations.append(output.detach())
        
    def clear(self):
        self.activations = []
    
    def extract_feat(self, img, stage='neck'):
        raise NotImplementedError

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True, **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            #------------Interpret clip----------------
            patch_feats = self.activations[0]
            if self.feat_type == 'last_layer':
                patch_feats = patch_feats.permute(1, 0, 2)
                # add self.model.ln_post() for post-norm features
            patch_feats = patch_feats[:, 1:]
            all_flat_features = patch_feats.reshape(-1, patch_feats.shape[-1]).cpu().numpy()  
            
            all_flat_features_pt = torch.from_numpy(all_flat_features)

            K = 3
            threshold_by_negative = True
            foreground_method = 'svd'
            num_patches = 196
            patch_h, patch_w = 14, 14
            img_size = 224
            feat_type = 'last_layer'
            pca_features, foreground_component = utils.get_pca_features(K, foreground_method, all_flat_features, all_flat_features_pt,
                                                                        threshold_by_negative)
            
            number_of_images = img.shape[0]
            all_fg_indices, _ = utils.get_foreground_background_indices(pca_features, num_patches, number_of_images,
                                                                        initial_threshold = 0 if threshold_by_negative else 0.4,
                                                                        foreground_component = foreground_component,
                                                                        patch_h = patch_h, patch_w = patch_w)

            # for single image
            # selected_patches = patch_feats[0][all_fg_indices[0].flatten()]
            # selected_patches = selected_patches @ self.model.visual.proj
            # patch_logits = selected_patches @ self.text_features_pos.to(torch.float32).T
            # patch_logits = patch_logits.softmax(dim=-1)
            # predictions = patch_logits.argmax(dim=-1)


            # visualization
            upscaled_masks = utils.upscale_foreground_masks(all_fg_indices, img_size)

            # takes a while for all cls images
            overlaid_results, crf_masks = [], []

            for i in range(len(all_fg_indices)):
                overlaid_result, crf_mask = utils.get_foreground_crf_map(upscaled_mask = upscaled_masks[i], image = img[i],
                                                                        img_size = img_size, DEFAULT_CRF_PARAMS = (10, 40, 13, 3, 3, 5.0))

                overlaid_results.append(overlaid_result)
                crf_masks.append(crf_mask)

            crf_masks = np.stack(crf_masks).astype(np.uint8)

            fig, axs = plt.subplots(1, 8, figsize=(20,10))
            np.vectorize(lambda ax:ax.axis('off'))(axs)
            for i in range(8):
                axs[i].imshow(utils.show_image(img[i]))

            fig, axs = plt.subplots(1, 8, figsize=(20,10))
            np.vectorize(lambda ax:ax.axis('off'))(axs)
            for i in range(8):
                axs[i].imshow(overlaid_results[i])
            plt.savefig(f'results/interpret_clip/batch_{self.batch_idx}.png')
            self.batch_idx += 1
            #------------Interpret clip----------------


        if self.cls_mode:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            pos_sim = list(pos_sim.softmax(dim=-1).detach().cpu().numpy())            
            return pos_sim
        else:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            neg_sim = (100.0 * image_features @ self.text_features_neg.T)
            return pos_sim, neg_sim

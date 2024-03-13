import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
# from torch.autograd import Variable
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

print(torch.cuda.is_available()) # should be True

t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA


a = "ông_hoàng của chúng_ta thương_lượng với nhau ."

# vocab = tokenizer.get_vocab()
# b = len(vocab)
#
# for i in a:
#     if i not in vocab:
#         tokenizer.add_tokens([i])
#         vocab = tokenizer.get_vocab()
#
# ids = torch.tensor([tokenizer.convert_tokens_to_ids(a)])
# print(ids)
# # print(vocab[7054])
# print(tokenizer.convert_ids_to_tokens(ids.tolist()[0]))
# print(tokenizer.decode(ids.tolist()[0]))
b = tokenizer.encode(a)
print(b)
c = tokenizer.decode(b)
print(c)
a = a.split(" ")
c = c.split(" ")
print(len(b), len(a), len(c))

#
#
#
# from chainer.variable import Variable
# import chainer.functions as F
# from scipy.spatial import distance
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
#
# def transform_and_normalize(vecs, kernel, bias):
#     """
#         Applying transformation then standardize
#     """
#     if not (kernel is None or bias is None):
#         vecs = (vecs + bias).dot(kernel)
#     return normalize(vecs)
#
# def normalize(vecs):
#     """
#         Standardization
#     """
#     return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
#
# def compute_kernel_bias(vecs):
#     """
#     Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
#     """
#
#     mu = vecs.mean(axis=0, keepdims=True)
#     cov = np.cov(vecs.T)
#     u, s, vh = np.linalg.svd(cov)
#     W = np.dot(u, np.diag(s**0.5))
#     W = np.linalg.inv(W.T)
#     return W, -mu
#
#
# a = "em that xinh dep"
# b = "em that la ngu ngoc"
#
# words = [np.array(tokenizer.encode(a),dtype=np.int32), np.array(tokenizer.encode(b),dtype=np.int32)]
# words_tmp = []
# vs = []
#
# for i in range(len(words)):
#     w = words[i].tolist()
#     input_ids = torch.tensor([w])
#     with torch.no_grad():
#         v = phobert(input_ids).last_hidden_state[0].numpy().astype(float)
#         vs.append(v)
# vecs = np.vstack(vs)
# def reduce_dimenstion(v, dim):
#     kernel, bias = compute_kernel_bias(v)
#     kernel = kernel[:, :dim]
#         #Sentence embeddings can be converted into an identity matrix
#         #by utilizing the transformation matrix
#     embeddings = transform_and_normalize(v,
#                     kernel=kernel,
#                     bias=bias
#                 )
#     return embeddings
# embeddings = reduce_dimenstion(vecs, 256)
# print(embeddings.shape)
# print(type(embeddings))
#
# embeddings = Variable(embeddings)
# print(embeddings.shape)
# print(type(embeddings))
#
#
# embeddings = embeddings.data
# embeddings = reduce_dimenstion(embeddings, 128)
# print(embeddings.shape)
# print(type(embeddings))
#
# embeddings = Variable(embeddings)
# print(embeddings.shape)
# print(type(embeddings))

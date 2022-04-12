import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from layers import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

def evaluate(embeds, idx_train, idx_val, idx_test, label, device, ratio, lr, wd
            , nb_classes=3, dataset='imdb_new', isTest=True):

    hid_units = embeds.shape[2]
    nb_classes = label.shape[2] 
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train] 
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(label[0, idx_train], dim=1)
    val_lbls = torch.argmax(label[0, idx_val], dim=1)
    test_lbls = torch.argmax(label[0, idx_test], dim=1)
    
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # print(loss)

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result/result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()

def validate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd):

    hid_units = embeds.shape[2]
    nb_classes = label.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(label[0, idx_train], dim=1)
    val_lbls = torch.argmax(label[0, idx_val], dim=1)
    test_lbls = torch.argmax(label[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(5):
        log = LogReg(hid_units, nb_classes) # in fact just a fc classifier
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()
            # if iter_ == 190: print(loss)

            log.eval()
            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))
        metric = np.zeros((3))
        metric[0] = np.mean(macro_f1s)
        metric[1] = np.mean(micro_f1s)
        metric[2] = np.mean(auc_score_list)

    return metric


# def evaluate(embeds, idx_train, idx_val, idx_test, labels, device, isTest=True):
#     hid_units = embeds.shape[2]
#     nb_classes = labels.shape[2]
#     xent = nn.CrossEntropyLoss()
#     train_embs = embeds[0, idx_train]
#     val_embs = embeds[0, idx_val]
#     test_embs = embeds[0, idx_test]

#     train_lbls = torch.argmax(labels[0, idx_train], dim=1)
#     val_lbls = torch.argmax(labels[0, idx_val], dim=1)
#     test_lbls = torch.argmax(labels[0, idx_test], dim=1)

#     accs = []
#     micro_f1s = []
#     macro_f1s = []
#     macro_f1s_val = [] ##

#     for _ in range(50):
#         log = LogReg(hid_units, nb_classes)
#         opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
#         log.to(device)

#         val_accs = []; test_accs = []
#         val_micro_f1s = []; test_micro_f1s = []
#         val_macro_f1s = []; test_macro_f1s = []
#         for iter_ in range(50):
#             # train
#             log.train()
#             opt.zero_grad()

#             logits = log(train_embs)
#             loss = xent(logits, train_lbls)

#             loss.backward()
#             opt.step()

#             # val
#             logits = log(val_embs)
#             preds = torch.argmax(logits, dim=1)

#             val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
#             val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
#             val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

#             val_accs.append(val_acc.item())
#             val_macro_f1s.append(val_f1_macro)
#             val_micro_f1s.append(val_f1_micro)

#             # test
#             logits = log(test_embs)
#             preds = torch.argmax(logits, dim=1)

#             test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
#             test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
#             test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

#             test_accs.append(test_acc.item())
#             test_macro_f1s.append(test_f1_macro)
#             test_micro_f1s.append(test_f1_micro)


#         max_iter = val_accs.index(max(val_accs))
#         accs.append(test_accs[max_iter])

#         max_iter = val_macro_f1s.index(max(val_macro_f1s))
#         macro_f1s.append(test_macro_f1s[max_iter])
#         macro_f1s_val.append(val_macro_f1s[max_iter]) ###

#         max_iter = val_micro_f1s.index(max(val_micro_f1s))
#         micro_f1s.append(test_micro_f1s[max_iter])

#     if isTest:
#         print("\t[Classification] acc:{}| Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(accs),
#                                                                                                 np.mean(macro_f1s),
#                                                                                                 np.std(macro_f1s),
#                                                                                                 np.mean(micro_f1s),
#                                                                                                 np.std(micro_f1s)))
#     else:
#         return np.mean(macro_f1s_val), np.mean(macro_f1s)

#     test_embs = np.array(test_embs.cpu())
#     test_lbls = np.array(test_lbls.cpu())

#     run_kmeans(test_embs, test_lbls, nb_classes)
#     run_similarity_search(test_embs, test_lbls)

# def run_similarity_search(test_embs, test_lbls):
#     numRows = test_embs.shape[0]

#     cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
#     st = []
#     for N in [5, 10, 20, 50, 100]:
#         indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
#         tmp = np.tile(test_lbls, (numRows, 1))
#         selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
#         original_label = np.repeat(test_lbls, N).reshape(numRows,N)
#         st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4)))

#     st = ','.join(st)
#     print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))


# def run_kmeans(x, y, k):
#     estimator = KMeans(n_clusters=k)

#     NMI_list = []
#     for i in range(10):
#         estimator.fit(x)
#         y_pred = estimator.predict(x)

#         s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
#         NMI_list.append(s1)

#     s1 = sum(NMI_list) / len(NMI_list)

#     print('\t[Clustering] NMI: {:.4f}'.format(s1))
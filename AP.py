# # coding:utf-8
#
# from sklearn.datasets.samples_generator import make_blobs
# import matplotlib.pyplot as plt
# import random
# import numpy as np
#
# ##############################################################################
# # 生成测试数据
# X = []
# line_num = 0
# with open('./middledata_nearest.csv') as f:
#     for line in f:
#         line_num += 1
#         if line_num != 1:
#             line = line.strip('\n')
#             line = line.split(',')
#             for j in xrange(len(line)):
#                 line[j] = float(line[j])
#             X.append(line)
#
# ##############################################################################
#
#
# def euclideanDistance(X, Y):
#     """计算每个点与其他所有点之间的欧几里德距离"""
#     X = np.array(X)
#     Y = np.array(Y)
#     # print X,Y
#     diff = np.zeros(len(X))
#     for i in xrange(len(X)):
#         diff[i] = X[i] - Y[i]
#     return len(X)-list(X).count(0)
#     # return np.sqrt(np.sum((X - Y) ** 2))
#
#
#
# def computeSimilarity(datalist):
#
#     num = len(datalist)
#
#     Similarity = []
#     for pointX in datalist:
#         dists = []
#         for pointY in datalist:
#             dist = euclideanDistance(pointX, pointY)
#             if dist == 0:
#                 dist = 1.5
#             # dists.append(dist)
#             dists.append(dist * -1)
#         Similarity.append(dists)
#
#     return Similarity
#
#
# def affinityPropagation(Similarity, lamda):
#
#     #初始化 吸引矩阵 和 归属 矩阵
#     Responsibility = np.zeros_like(Similarity, dtype=np.int)
#     Availability = np.zeros_like(Similarity, dtype=np.int)
#
#     num = len(Responsibility)
#
#     count = 0
#     while count < 10:
#         count += 1
#         # update 吸引矩阵
#
#         for Index in range(num):
#             # print len(Similarity[Index])
#             kSum = [s + a  for s, a in zip(Similarity[Index], Availability[Index])]
#             # print kSum
#             for Kendex in range(num):
#                 kfit = delete(kSum, Kendex)
#                 # print fit
#                 ResponsibilityNew = Similarity[Index][Kendex] - max(kfit)
#                 Responsibility[Index][Kendex] = lamda * Responsibility[Index][Kendex] + (1 - lamda) * ResponsibilityNew
#
#         # print "Responsibility", Responsibility
#
#
#         # update 归属矩阵
#
#         ResponsibilityT = Responsibility.T
#
#         # print ResponsibilityT, Responsibility
#
#         for Index in range(num):
#
#             iSum = [r for r in ResponsibilityT[Index]]
#
#             for Kendex in range(num):
#
#                 # print Kendex
#                 # print "ddddddddddddddddddddddddddd", ResponsibilityT[Kendex]
#                 #
#                 ifit = delete(iSum, Kendex)
#                 ifit = filter(isNonNegative, ifit)   #上面 iSum  已经全部大于0  会导致  delete 下标错误
#
#                 #   k == K 对角线的情况
#                 if Kendex == Index:
#                     AvailabilityNew  = sum(ifit)
#                 else:
#                     result = Responsibility[Kendex][Kendex] + sum(ifit)
#                     AvailabilityNew = result if result > 0 else 0
#                 Availability[Kendex][Index] = lamda * Availability[Kendex][Index] + (1 - lamda) * AvailabilityNew
#         print "###############################################"
#         print Responsibility
#         print Availability
#         print "###############################################"
#         return Responsibility + Availability
#
# def computeCluster(fitable, data):
#     clusters = {}
#     num = len(fitable)
#     for node in range(num):
#         fit = list(fitable[node])
#         key = fit.index(max(fit))
#         if not clusters.has_key(key):
#             clusters[key] = []
#         point = tuple(data[node])
#         clusters[key].append(point)
#
#     return clusters
# ##############################################################################
#
# """切片删除 返回新数组"""
# def delete(lt, index):
#     lt = lt[:index] + lt[index+1:]
#     return lt
#
# def isNonNegative(x):
#     return x >= 0
#
#
# ##############################################################################
#
# Similarity = computeSimilarity(X)
#
# Similarity = np.array(Similarity)
#
# print "Similarity", Similarity
#
# fitable = affinityPropagation(Similarity, 0.34)
#
# print fitable
#
# clusters = computeCluster(fitable, X)
#
# # print clusters
#
# ##############################################################################
# clusters = clusters.values()
#
# print len(clusters)
#
# ##############################################################################
# # def plotClusters(clusters, title):
# #     """ 画图 """
# #     plt.figure(figsize=(8, 5), dpi=80)
# #     axes = plt.subplot(111)
# #     col=[]
# #     r = lambda: random.randint(0,255)
# #     for index in range(len(clusters)):
# #         col.append(('#%02X%02X%02X' % (r(),r(),r())))
# #     color = 0
# #     for cluster in clusters:
# #         cluster = np.array(cluster).T
# #         axes.scatter(cluster[0],cluster[1], s=20, c = col[color])
# #         color += 1
# #     plt.title(title)
# #     # plt.show()
# # ##############################################################################
# # plotClusters(clusters, "clusters by affinity propagation")
# # plt.show()
#
# ##############################################################################



#coding:utf-8

from sklearn.cluster import AffinityPropagation
import numpy as np
import csv

# 生成测试数据
X = []
line_num = 0
with open('./middledata_nearest.csv') as f:
    for line in f:
        line_num += 1
        if line_num != 1:
            line = line.strip('\n')
            line = line.split(',')
            for j in xrange(len(line)):
                line[j] = int(line[j])
            X.append(line)
f.close()

# AP模型拟合
X = np.array(X)
af = AffinityPropagation(affinity='different').fit(X)  ##自己在库函数中添加了一个计算不同值的个数
# af = AffinityPropagation(affinity='euclidean').fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
new_X = np.column_stack((X, labels))
# print new_X
n_clusters_ = len(cluster_centers_indices)
print n_clusters_

csvfile = file('./cluster.csv','wb')
writer = csv.writer(csvfile)
writer.writerow(['Age','Location','Gender','President','Ideology','Education','Party','Ethnicity','Relationship','Income','Interested','Occupation','Looking','Religion','class'])
for i in new_X:
    writer.writerow(i)
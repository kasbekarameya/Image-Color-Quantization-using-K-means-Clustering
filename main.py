import numpy as np
import math
import cv2
import random

UBIT = 'ameyakir'
PersonNo = 50292574
random.seed(sum([ord(c) for c in UBIT]))

def EuclideanDistance(Mu, Pt):
    S = 0; #  The sum of the squared differences of the elements
    for i in range(len(Mu)):
        S += math.pow(Mu[i]-Pt[i], 2);
        #print(S)
    return math.sqrt(S)

def FindXYnZ(d):
	p1 = []
	p2 = []
	p3 = []
	for i in range(len(d)):
		for j in range(len(d[i])):
			for k in range(len(d[i][j])):
				if(k==0):
					p1.append(d[i][j][0])
				elif(k==1):
					p2.append(d[i][j][1])
				else:
					p3.append(d[i][j][2])
	return p1,p2,p3



def FindCentroid(c):
	m1 = 0
	m2 = 0
	m3 = 0
	#print(type(c[1][1]))
	for i in range(len(c)):
		for j in range(len(c[i])):
			if(j==0):
				m1 += c[i][j]
			elif(j==1):
				m2 += c[i][j]
			else:
				m3 += c[i][j]
	avg1 = m1/len(c)
	avg2 = m2/len(c)
	avg3 = m3/len(c)
	return np.round(avg1,0),np.round(avg2,0),np.round(avg3,0)

# To select first random centroids




#Create clusters
def CreateClusters(m,orig,k):
	clust1 = []
	clust2 = []
	clust3 = []
	clust4 = []
	clust5 = []
	clust6 = []
	clust7 = []
	clust8 = []
	clust9 = []
	clust10 = []
	clust11 = []
	clust12 = []
	clust13 = []
	clust14 = []
	clust15 = []
	clust16 = []
	clust17 = []
	clust18 = []
	clust19 = []
	clust20 = []
	ClassVect = []

	if(k==3):
		for i in range(len(orig)):
			for j in range(len(orig[i])):
				len1 = EuclideanDistance(m[0], orig[i][j])
				len2 = EuclideanDistance(m[1], orig[i][j])
				len3 = EuclideanDistance(m[2], orig[i][j])
				#print(orig[i][j])
				if(len1 == min(len1,len2,len3)):
					clust1.append(list(orig[i][j]))
					ClassVect.append(0)
				elif(len2 == min(len1,len2,len3)):
					clust2.append(list(orig[i][j]))
					ClassVect.append(1)
				else:
					clust3.append(list(orig[i][j]))
					ClassVect.append(2)

		return clust1, clust2, clust3, ClassVect

	if(k==5):
		for i in range(len(orig)):
			for j in range(len(orig[i])):
				len1 = EuclideanDistance(m[0], orig[i][j])
				len2 = EuclideanDistance(m[1], orig[i][j])
				len3 = EuclideanDistance(m[2], orig[i][j])
				len4 = EuclideanDistance(m[3], orig[i][j])
				len5 = EuclideanDistance(m[4], orig[i][j])

				#print(orig[i][j])
				if(len1 == min(len1,len2,len3,len4,len5)):
					clust1.append(list(orig[i][j]))
					ClassVect.append(0)
				elif(len2 == min(len1,len2,len3,len4,len5)):
					clust2.append(list(orig[i][j]))
					ClassVect.append(1)
				elif(len3 == min(len1,len2,len3,len4,len5)):
					clust3.append(list(orig[i][j]))
					ClassVect.append(2)
				elif(len4 == min(len1,len2,len3,len4,len5)):
					clust4.append(list(orig[i][j]))
					ClassVect.append(3)
				else:
					clust5.append(list(orig[i][j]))
					ClassVect.append(4)

		return clust1, clust2, clust3, clust4, clust5, ClassVect

	if(k==10):
		for i in range(len(orig)):
			for j in range(len(orig[i])):
				len1 = EuclideanDistance(m[0], orig[i][j])
				len2 = EuclideanDistance(m[1], orig[i][j])
				len3 = EuclideanDistance(m[2], orig[i][j])
				len4 = EuclideanDistance(m[3], orig[i][j])
				len5 = EuclideanDistance(m[4], orig[i][j])
				len6 = EuclideanDistance(m[5], orig[i][j])
				len7 = EuclideanDistance(m[6], orig[i][j])
				len8 = EuclideanDistance(m[7], orig[i][j])
				len9 = EuclideanDistance(m[8], orig[i][j])
				len10 = EuclideanDistance(m[9], orig[i][j])

				#print(orig[i][j])
				if(len1 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust1.append(list(orig[i][j]))
					ClassVect.append(0)
				elif(len2 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust2.append(list(orig[i][j]))
					ClassVect.append(1)
				elif(len3 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust3.append(list(orig[i][j]))
					ClassVect.append(2)
				elif(len4 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust4.append(list(orig[i][j]))
					ClassVect.append(3)
				elif(len5 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust5.append(list(orig[i][j]))
					ClassVect.append(4)
				elif(len6 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust6.append(list(orig[i][j]))
					ClassVect.append(5)
				elif(len7 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust7.append(list(orig[i][j]))
					ClassVect.append(6)
				elif(len8 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust8.append(list(orig[i][j]))
					ClassVect.append(7)
				elif(len9 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10)):
					clust9.append(list(orig[i][j]))
					ClassVect.append(8)
				else:
					clust10.append(list(orig[i][j]))
					ClassVect.append(9)

		return clust1, clust2, clust3, clust4, clust5, clust6, clust7, clust8, clust9, clust10, ClassVect

	if(k==20):
		for i in range(len(orig)):
			for j in range(len(orig[i])):
				len1 = EuclideanDistance(m[0], orig[i][j])
				len2 = EuclideanDistance(m[1], orig[i][j])
				len3 = EuclideanDistance(m[2], orig[i][j])
				len4 = EuclideanDistance(m[3], orig[i][j])
				len5 = EuclideanDistance(m[4], orig[i][j])
				len6 = EuclideanDistance(m[5], orig[i][j])
				len7 = EuclideanDistance(m[6], orig[i][j])
				len8 = EuclideanDistance(m[7], orig[i][j])
				len9 = EuclideanDistance(m[8], orig[i][j])
				len10 = EuclideanDistance(m[9], orig[i][j])
				len11 = EuclideanDistance(m[10], orig[i][j])
				len12 = EuclideanDistance(m[11], orig[i][j])
				len13 = EuclideanDistance(m[12], orig[i][j])
				len14 = EuclideanDistance(m[13], orig[i][j])
				len15 = EuclideanDistance(m[14], orig[i][j])
				len16 = EuclideanDistance(m[15], orig[i][j])
				len17 = EuclideanDistance(m[16], orig[i][j])
				len18 = EuclideanDistance(m[17], orig[i][j])
				len19 = EuclideanDistance(m[18], orig[i][j])
				len20 = EuclideanDistance(m[19], orig[i][j])

				#print(orig[i][j])
				if(len1 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust1.append(list(orig[i][j]))
					ClassVect.append(0)
				elif(len2 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust2.append(list(orig[i][j]))
					ClassVect.append(1)
				elif(len3 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust3.append(list(orig[i][j]))
					ClassVect.append(2)
				elif(len4 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust4.append(list(orig[i][j]))
					ClassVect.append(3)
				elif(len5 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust5.append(list(orig[i][j]))
					ClassVect.append(4)
				elif(len6 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust6.append(list(orig[i][j]))
					ClassVect.append(5)
				elif(len7 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust7.append(list(orig[i][j]))
					ClassVect.append(6)
				elif(len8 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust8.append(list(orig[i][j]))
					ClassVect.append(7)
				elif(len9 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust9.append(list(orig[i][j]))
					ClassVect.append(8)
				elif(len10 ==  min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust10.append(list(orig[i][j]))
					ClassVect.append(9)
				elif(len11 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust11.append(list(orig[i][j]))
					ClassVect.append(10)
				elif(len12 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust12.append(list(orig[i][j]))
					ClassVect.append(11)
				elif(len13 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust13.append(list(orig[i][j]))
					ClassVect.append(12)
				elif(len14 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust14.append(list(orig[i][j]))
					ClassVect.append(13)
				elif(len15 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust15.append(list(orig[i][j]))
					ClassVect.append(14)
				elif(len16 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust16.append(list(orig[i][j]))
					ClassVect.append(15)
				elif(len17 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust17.append(list(orig[i][j]))
					ClassVect.append(16)
				elif(len18 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust18.append(list(orig[i][j]))
					ClassVect.append(17)
				elif(len19 == min(len1,len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12,len13,len14,len15,len16,len17,len18,len19,len20)):
					clust19.append(list(orig[i][j]))
					ClassVect.append(18)
				else:
					clust20.append(list(orig[i][j]))
					ClassVect.append(19)


		return clust1, clust2, clust3, clust4, clust5, clust6, clust7, clust8, clust9, clust10, clust11, clust12, clust13, clust14, clust15, clust16, clust17, clust18, clust19, clust20, ClassVect

def Quantization(n,k):
	orig = cv2.imread("baboon.jpg")

	data = []
	for i in range(len(orig)):
		for j in range(len(orig[i])):
			data.append(list(orig[i][j]))

	x, y, z= FindXYnZ(orig)

	Mu = []
	for i in range(k):
		temp = data[random.randrange(len(data))]
		Mu.append(temp)


	for i in range(n):
		if(k==3):
			cluster1, cluster2, cluster3, CV = CreateClusters(Mu,orig,k)

			Mu[0] = FindCentroid(cluster1)
			Mu[1] = FindCentroid(cluster2)
			Mu[2] = FindCentroid(cluster3)

		elif(k==5):
			cluster1, cluster2, cluster3, cluster4, cluster5, CV = CreateClusters(Mu,orig,k)

			Mu[0] = FindCentroid(cluster1)
			Mu[1] = FindCentroid(cluster2)
			Mu[2] = FindCentroid(cluster3)
			Mu[3] = FindCentroid(cluster4)
			Mu[4] = FindCentroid(cluster5)

		elif(k==10):
			cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10, CV = CreateClusters(Mu,orig,k)

			Mu[0] = FindCentroid(cluster1)
			Mu[1] = FindCentroid(cluster2)
			Mu[2] = FindCentroid(cluster3)
			Mu[3] = FindCentroid(cluster4)
			Mu[4] = FindCentroid(cluster5)
			Mu[5] = FindCentroid(cluster6)
			Mu[6] = FindCentroid(cluster7)
			Mu[7] = FindCentroid(cluster8)
			Mu[8] = FindCentroid(cluster9)
			Mu[9] = FindCentroid(cluster10)

		elif(k==20):
			cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10, cluster11, cluster12, cluster13, cluster14, cluster15, cluster16, cluster17, cluster18, cluster19, cluster20, CV = CreateClusters(Mu,orig,k)

			Mu[0] = FindCentroid(cluster1)
			Mu[1] = FindCentroid(cluster2)
			Mu[2] = FindCentroid(cluster3)
			Mu[3] = FindCentroid(cluster4)
			Mu[4] = FindCentroid(cluster5)
			Mu[5] = FindCentroid(cluster6)
			Mu[6] = FindCentroid(cluster7)
			Mu[7] = FindCentroid(cluster8)
			Mu[8] = FindCentroid(cluster9)
			Mu[9] = FindCentroid(cluster10)
			Mu[10] = FindCentroid(cluster1)
			Mu[11] = FindCentroid(cluster2)
			Mu[12] = FindCentroid(cluster3)
			Mu[13] = FindCentroid(cluster4)
			Mu[14] = FindCentroid(cluster5)
			Mu[15] = FindCentroid(cluster6)
			Mu[16] = FindCentroid(cluster7)
			Mu[17] = FindCentroid(cluster8)
			Mu[18] = FindCentroid(cluster9)
			Mu[19] = FindCentroid(cluster10)

	npdata = np.asarray(data)
	npvector = np.asarray(CV)
	for i in range(npvector.shape[0]):
		npdata[i]=Mu[npvector[i]]

	quantimg=np.reshape(npdata,(orig.shape))
	cv2.imwrite('images/task3_baboon_'+str(k)+'.jpg',quantimg)

# Executing the code
Quantization(20,3)
Quantization(20,5)
Quantization(20,10)
Quantization(20,20)

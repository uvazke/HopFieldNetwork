#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

class HopfieldNetWork:
    def __init__(self):
        #Definition
        self.node_row_num = 5
        self.node_col_num = 5
        self.num = self.node_col_num * self.node_row_num
        self.w = np.zeros((self.num,self.num))
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.now_iter_num = 0
        self.correct = 0
        self.simirality = 0
        #parameters
        self.deltaE = 0.001
        self.iter_num = 500
        self.dirname = "../datas/Hopfield/"


    def normailize_data(self,data):
        tmin = np.min(data, 1)
        tmax = np.max(data, 1)
        data_nor = data #initialize
        # normalize from -1 to 1
        row_size = data[:,0].size
        for i in range (row_size):
            data_nor[i] = 2*(data[i] - tmin[i]).astype(float)/(tmax[i]- tmin[i]) - 1

        return (data_nor)


    def fit(self, train_data, threshold = 0):
        self.threshold = threshold
        start = time.time()

        self.train_data_size = train_data[:,0].size
        for i in range (self.train_data_size):
            self.w += train_data[i] * train_data[i].reshape(1,self.num).transpose()

        for i in range (self.num):
            self.w[i][i] = 0
        elapsed_time = time.time() - start
        print("elapsed_time: " + str(elapsed_time))


    def predict_and_calculate(self,train_data):

        simirality_mean = 0
        simirality_ste = 0
        correct_percentage = 0
        correct_ste = 0
        noise_percentages = 0
        #initialize
        predict_data = np.array(train_data)

        for noise in range (10):
            self.TP = 0
            self.TN = 0
            self.FP = 0
            self.FN = 0
            self.now_iter_num = 0
            self.correct = 0
            self.simirality = 0

            noise_percentage = (noise + 1) * 0.05
            noise_percentages = np.append(noise_percentages,noise_percentage)
            for i in range (self.iter_num):
                print("now:" + str(self.now_iter_num))
                #each test_data is based on each train_data
                test_data = self.make_test_data(train_data,noise_percentage)

                for j in range (self.train_data_size):
                    predict_data[j] = self.predict_based_on_E(test_data[j])
                    self.assess(train_data[j],predict_data[j])
                    self.calculate_accuracy()

            [SM,CP,S_STE,C_STE] = self.accuracy()
            simirality_mean = np.append(simirality_mean, SM)
            correct_percentage = np.append(correct_percentage,CP)
            simirality_ste = np.append(simirality_ste,S_STE)
            correct_ste = np.append(correct_ste, C_STE)
        simirality_mean = simirality_mean[1:] * 100
        simirality_ste = simirality_ste[1:] * 100
        correct_percentage = correct_percentage[1:] * 100
        correct_ste = correct_ste[1:] * 100
        noise_percentages = noise_percentages[1:] * 100

        accdataname = "accuracydata%d.csv" % self.train_data_size
        accdataname = self.dirname + accdataname
        np.savetxt(accdataname,np.array([simirality_mean,correct_percentage]))
        print([simirality_mean,correct_percentage])

        size = 0.5
        fontsize = 10
        loc = "lower left"
        plt.figure(figsize=(8 * size, 6 * size))
        plt.errorbar(noise_percentages, simirality_mean, fmt = '.', yerr = simirality_ste * 2, label = "simirality_mean",c = 'red')
        plt.errorbar(noise_percentages, correct_percentage, fmt = '.', yerr = correct_ste * 2, label = "correct_percentage", c = 'blue')
        plt.xticks([0,10,20,30,40,50],  fontsize = fontsize)
        plt.xlim(0,55)
        plt.xlabel("noise(%)",fontsize = fontsize)
        plt.yticks([0,20,40,60,80,100],fontsize = fontsize)
        plt.ylim(0,105)
        plt.ylabel("accuracy(%)", fontsize = fontsize)
        plt.legend(loc= loc,frameon = True,fontsize = fontsize)
        plt.tight_layout()
        accimagename = "accuracy_%d.png" % self.train_data_size
        accimagename = self.dirname + accimagename
        plt.savefig(accimagename)
        plt.show()
        plt.close()

        self.picture_show(train_data,test_data,predict_data)


    def predict_based_on_E(self, test_data):
        predict_data = np.array(test_data)

        energy = 1 + self.energy(predict_data)
        while energy - self.energy(predict_data) >= self.deltaE:
            energy = self.energy(predict_data)
            thfunc = np.dot(self.w, predict_data) - self.threshold
            j = np.where(thfunc >= 0)
            predict_data[j] = 1
            k = np.where(thfunc < 0)
            predict_data[k] = -1
        return (predict_data)


    def energy(self, predict_data):
        E = -np.dot(np.dot(predict_data,self.w),predict_data)/2 + np.sum(self.threshold * predict_data)
        return E


    def assess(self,train_data,predict_data):
        self.now_iter_num += 1
        accuracy_of_data = np.array(predict_data) + np.array(train_data)
        if predict_data.ndim == 1:
            AOD = accuracy_of_data
        else:
            # detect the fitted data
            AOD_S = np.sum(np.absolute(accuracy_of_data),1)
            AOD_ind = np.argmax(AOD_S)
            AOD = accuracy_of_data[AOD_ind]

        TP = np.where(AOD == 2)
        TN = np.where(AOD == -2)
        FP = np.in1d(np.where(AOD == 0),np.where(predict_data == 1))
        FN = np.in1d(np.where(AOD == 0),np.where(predict_data == -1))
        self.TP += np.size(TP)
        self.TN += np.size(TN)
        self.FP += np.size(FP)
        self.FN += np.size(FN)
        #degree of simirality
        self.sim = np.size(TP) + np.size(TN)


    def calculate_accuracy(self):
        #self.precision = self.TP/(self.TP + self.FP)
        #self.recall = self.TP/(self.TP + self.FN)
        #self.F_value = 2 * self.recall * self.precision / (self.recall + self.precision)
        self.simirality =np.append(self.simirality, self.sim/self.num)

        if self.sim == self.num:
            self.correct = np.append(self.correct,1)
        else:
            self.correct = np.append(self.correct,0)




    def accuracy(self):
        self.simirality = np.array(self.simirality[1:])
        self.correct = np.array(self.correct[1:])

        self.simirality_mean = self.simirality.mean()
        self.simirality_ste = self.simirality.std()/math.sqrt(self.now_iter_num)

        self.correct_percentage = self.correct.mean()
        self.correct_ste = self.correct.std()/math.sqrt(self.now_iter_num)

        return [self.simirality_mean, self.correct_percentage,self.simirality_ste,self.correct_ste]


    def make_test_data(self,train_data,noise_percentage):
        test_data = train_data.astype(float)
        noise_num = int(self.num * noise_percentage)
        noise_index = np.random.randint(self.num, size = noise_num)

        for row in range (self.train_data_size):
            for i in range(noise_num):
                test_data[row,noise_index[i]] = -test_data[row,noise_index[i]]

        return test_data


    def put_predict_img(self,number,predict_data):
        out_data = np.array(predict_data) * 255
        out_data.resize((self.node_row_num,self.node_col_num))
        out_data.flags.writeable = True
        image = Image.fromarray(np.uint8(out_data))
        image_name = 'predict_img%d.png' % number
        image.save(image_name)

    def put_train_image(self,train_data):
        for i in range (self.train_data_size):
            out_data = np.array(train_data[i]) * 255
            out_data.resize((self.node_row_num,self.node_col_num))
            out_data.flags.writeable = True
            image = Image.fromarray(np.uint8(out_data))
            image_name = 'train_img%d.png' % i
            image.save(image_name)

    def put_test_image(self,test_data):
        for i in range (self.train_data_size):
            out_data = np.array(test_data[i]) * 255
            out_data.resize((self.node_row_num,self.node_col_num))
            out_data.flags.writeable = True
            image = Image.fromarray(np.uint8(out_data))
            image_name = 'test_img%d.png' % i
            image.save(image_name)


    def picture_show(self,train_data,test_data,predict_data):
        for i in range (self.train_data_size):
            if self.train_data_size ==1:
                trainD = (train_data + 1) *255 / 2
                testD = (test_data + 1) * 255 / 2
                predictD = (predict_data + 1) * 255 / 2

            else:
                trainD = (train_data[i] + 1) * 255 / 2
                testD = (test_data[i] + 1) * 255 / 2
                predictD = (predict_data[i] + 1) * 255 / 2
            trainD.resize(self.node_row_num, self.node_col_num)
            testD.resize(self.node_row_num, self.node_col_num)
            predictD.resize(self.node_row_num, self.node_col_num)
            plt.subplot(3, self.train_data_size, i + 1)
            title ="train%d" % i
            plt.title(title)
            plt.imshow(trainD, interpolation="nearest")
            plt.subplot(3, self.train_data_size, i + 1 + self.train_data_size)
            title = "test%d" % i
            plt.title(title)
            plt.imshow(testD, interpolation="nearest")
            plt.subplot(3, self.train_data_size, i + 1 + 2 * self.train_data_size)
            title = "predict%d" % i
            plt.title(title)
            plt.imshow(predictD, interpolation="nearest")
        plt.tight_layout()

        imagename = "image%d.png" % self.train_data_size
        imagename = self.dirname + imagename
        plt.savefig(imagename)
        plt.show()


if __name__ == "__main__":
    train_data1 = np.array([0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,1,0,1,0])#A
    train_data2 = np.array([1,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,1,0])#B
    train_data3 = np.array([0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0])#C
    train_data4 = np.array([1,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0])#D
    train_data5 = np.array([1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1])#E
    train_data6 = np.array([1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0])#F
    train_data7 = np.array([1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1])#Z


    train_datas = np.array([train_data1,train_data2])
    hop = HopfieldNetWork()
    # normailizing data
    train_data = hop.normailize_data(train_datas)

    hop.fit(train_data,0)
    hop.predict_and_calculate(train_data)

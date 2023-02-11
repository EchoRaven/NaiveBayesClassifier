import numpy as np

class BNC:
    def __init__(self):
        self.data = []
        self.label = []
        self.classNum = 0
        #matrix表示条件概率矩阵
        self.matrix = {}
        self.labelProbability = {}
        self.classifyTot = []

    def Train(self, data, label):
        # data是向量组
        self.data = data
        for l in label:
            if l not in self.labelProbability.keys():
                self.labelProbability[l] = 0
            self.labelProbability[l] += 1
        self.classNum = len(np.unique(label))
        for key in self.labelProbability.keys():
            self.labelProbability[key] = (self.labelProbability[key] + 1)/(len(label)+self.classNum)
        size = len(data[0])
        self.classifyTot = []
        for index in range(size):
            #表示第i个属性的条件概率矩阵
            self.matrix[index] = {}
            #按照index的属性重新分配数据
            classifyData = {}
            for i in range(len(data)):
                if label[i] not in classifyData.keys():
                    classifyData[label[i]] = {}
                if data[i][index] not in classifyData[label[i]].keys():
                    classifyData[label[i]][data[i][index]] = 0
                classifyData[label[i]][data[i][index]] += 1
            for key in classifyData.keys():
                tot = 0
                #计算总数
                for ky in classifyData[key].keys():
                    tot += classifyData[key][ky]
                if key not in self.matrix[index].keys():
                    self.matrix[index][key] = {}
                for ky in classifyData[key].keys():
                    if ky not in self.matrix[index][key].keys():
                        #获取条件概率矩阵
                        self.matrix[index][key][ky] = (classifyData[key][ky]+1)/(tot+len(classifyData.keys()))
                self.classifyTot.append(tot)

    def Predict(self, data=[]):
        probability = 0
        res = None
        for key in self.labelProbability.keys():
            res = key
            break
        for key in self.labelProbability.keys():
            prob = self.labelProbability[key]
            for index in range(len(data)):
                if data[index] not in self.matrix[index][key].keys():
                    multi = 1/(len(self.matrix[index].keys())+self.classifyTot[index])
                else:
                    multi = self.matrix[index][key][data[index]]
                prob *= multi
            if prob > probability:
                probability = prob
                res = key
        return res

if __name__ == "__main__":
    bnc = BNC()
    datas = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
             ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑"],
             ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"],
             ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
             ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘"],
             ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑"],

             ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑"],
             ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘"],
             ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑"],
             ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘"],
             ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑"],
             ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑"],
             ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘"],
             ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑"],
             ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑"]]

    labels = ["好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜", "好瓜",
              "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜", "坏瓜"]
    bnc.Train(data=datas, label=labels)
    print(bnc.Predict(["乌黑", "稍蜷", "沉闷", "稍糊", "平坦", "硬滑"]))
# 该函数用于迭代各个参数
#库的导入
import random

import numpy as np
import matplotlib.pyplot as plt
import main_1

#待求解问题，求解问题为求最小值
def function(vocab_wai, w2v_model_wai, maxLen_wai, trainSeq_wai, trainCate_wai, testSeq_wai, testCate_wai, conv1_op, conv2_op, maxpool_op, stride, qlj_op, fznh_1, fznh_2, epoch, batch):
    score = main_1.CNN_score(vocab_wai, w2v_model_wai, maxLen_wai, trainSeq_wai, trainCate_wai, testSeq_wai, testCate_wai, conv1_op, conv2_op, maxpool_op, stride, qlj_op, fznh_1, fznh_2, epoch, batch)
    return score[1]


def KHA_1(vocab_wai, w2v_model_wai, maxLen_wai, trainSeq_wai, trainCate_wai, testSeq_wai, testCate_wai):
    # vocab = vocab_wai
    # maxLen = maxLen_wai
    # w2v_model = w2v_model_wai
    # trainSeq = trainSeq_wai
    # trainCate = trainCate_wai
    # testSeq = testSeq_wai
    # testCate = testCate_wai
    m = 3  # 种群数量
    imax = 3  # 迭代次数
    dimen = 9  # 解的维度
    rangelow = [200, 20, 3, 2, 200, 0.1, 0.1, 5, 200]  # 解的最小取值
    rangehigh = [400, 50, 10, 8, 300, 0.5, 0.5, 10, 300]  # 解的最大取值
    Nmax = 0.01  # 最大诱导速度
    Vf = 0.02  # 觅食速度

    # pop用于存储种群个体的位置信息，pop_fitness用于存储个体对应的适应度值
    pop = np.zeros((m, dimen))
    pop_fitness = np.zeros(m)
    # 对种群个体进行初始化并计算对应适应度值
    for j in range(m):
        for i in range(dimen):
            low_1 = rangelow[i]
            high_1 = rangehigh[i]
            if low_1 >= 1:
                pop[j, i] = random.randint(low_1, high_1)##########################
                print(int(pop[j, i]))
            else:
                pop[j, i] = random.uniform(low_1, high_1)
                print(pop[j, i])
        pop_fitness[j] = function(vocab_wai=vocab_wai, w2v_model_wai=w2v_model_wai, maxLen_wai=maxLen_wai, trainSeq_wai=trainSeq_wai, trainCate_wai=trainCate_wai, testSeq_wai=testSeq_wai, testCate_wai=testCate_wai, conv1_op=int(pop[j, 0]), conv2_op=int(pop[j, 1]), maxpool_op=int(pop[j, 2]), stride=int(pop[j, 3]), qlj_op=int(pop[j, 4]), fznh_1=pop[j, 5], fznh_2=pop[j, 6], epoch=int(pop[j, 7]), batch=int(pop[j, 8]))

    # allbestpop,allbestfit分别存储种群在历史迭代过程中最优个体解及对应适应度
    allbestpop, allbestfit = pop[pop_fitness.argmin()].copy(), pop_fitness.min()

    # bestpop,bestfit分别存储每个个体历史最优解及对应适应度
    bestpop, bestfit = pop.copy(), pop_fitness.copy()

    # 分别为当前种群适应度最优个体适应度值、位置信息与适应度最差个体适应度值
    bestfitness = pop_fitness.min()
    bestplace = pop[pop_fitness.argmin()].copy()
    worstfitness = pop_fitness.max()

    # his_bestfit存储每次迭代时种群历史适应度值最优的个体适应度
    his_bestfit = np.zeros(imax)

    # Nold为历史诱导值，Fold为历史捕食移动量
    Nold = np.zeros((m, dimen))
    Fold = np.zeros((m, dimen))

    # 计算变量的上下界差距之和
    sum1 = 448.8

    # 开始迭代训练
    for i in range(imax):
        print("The iteration is:", i + 1)
        # 当前迭代次数与总迭代次数的比例
        iratio = (i + 1) / imax
        # 将Ct、wn以及wf这三个参数设置为在取值范围内随迭代次数减小的数，也按照取值范围可设置为常数
        Ct = 0.01 + 2 * (1 - iratio)
        wn = 0.01 + 1 * (1 - iratio)
        wf = 0.01 + 1 * (1 - iratio)
        # 对每个个体依次进行位移
        for j in range(m):
            # 1、其他个体引起的移动
            # 计算个体受到影响的范围距离
            sum2 = 0
            for l in range(m):
                e = np.sum(np.square(pop[l] - pop[j]))
                sum2 = sum2 + e
            d = sum2 / (5 * m)
            # 计算受到一定范围内其他个体的影响alphalocal
            alphalocal = 0
            for l in range(m):
                sum3 = np.sum(np.square(pop[l] - pop[j]))
                if sum3 <= d:
                    k = (pop_fitness[l] - pop_fitness[j]) / (worstfitness - bestfitness)
                    x = (pop[l] - pop[j]) / (sum3 + 0.0001)
                    alphalocal = alphalocal + k * x
            # 计算种群最优目标的影响alphatarget
            Cbest = 2 * (np.random.rand() + iratio)
            Kbest1 = (pop_fitness[j] - bestfitness) / (worstfitness - bestfitness)
            xbest1 = (bestplace - pop[j]) / (np.sum(np.square(bestplace - pop[j])) + 0.0001)
            alphatarget = Cbest * Kbest1 * xbest1

            # 计算第i个个体受到的综合影响
            alphai = alphalocal + alphatarget
            # 计算其他个体引起的移动Nnew
            Nnew = Nmax * alphai + wn * Nold[j]

            # 2、觅食移动
            # 食物影响系数
            Cfood = 2 * (np.random.rand() - iratio)
            # 计算食物的估计位置Xfood及对应的适应度值
            multi1 = 0
            multi2 = 0
            for j in range(m):
                multi1 = multi1 + (1 / pop_fitness[j]) * pop[j]
                multi2 = multi2 + (1 / pop_fitness[j])
            Xfood = multi1 / multi2
            fitnessfood = function(vocab_wai=vocab_wai, w2v_model_wai=w2v_model_wai, maxLen_wai=maxLen_wai, trainSeq_wai=trainSeq_wai, trainCate_wai=trainCate_wai, testSeq_wai=testSeq_wai, testCate_wai=testCate_wai, conv1_op=int(Xfood[0]), conv2_op=int(Xfood[1]), maxpool_op=int(Xfood[2]), stride=int(Xfood[3]), qlj_op=int(Xfood[4]), fznh_1=Xfood[5], fznh_2=Xfood[6], epoch=int(Xfood[7]), batch=int(Xfood[8]))

            # 计算受到食物的影响移动量belta
            Kbest2 = (pop_fitness[j] - bestfit[j]) / (worstfitness - bestfitness)
            xbest2 = (bestpop[j] - pop[j]) / (np.sum(np.square(bestpop[j] - pop[j])) + 0.0001)
            beltabest = xbest2 * Kbest2
            Kfood = (pop_fitness[j] - fitnessfood) / (worstfitness - bestfitness)
            xbestfood = (Xfood - pop[j]) / (np.sum(np.square(Xfood - pop[j])) + 0.0001)
            beltafood = Kfood * xbestfood
            belta = beltafood + beltabest

            # 计算觅食移动量Fnew
            Fnew = Vf * belta + wf * Fold[j]

            # 3、计算物理扩散位移量
            Dmax = np.random.uniform(low=0.002, high=0.01)
            D = Dmax * iratio * np.random.uniform(low=-1, high=1, size=(1, dimen))

            # 4、计算前面三种行为的综合影响
            change = Nnew + Fnew + D
            # 计算速度的比例因子向量deltat
            deltat = Ct * sum1

            # 5、计算当前个体在移动后的新位置
            pop[j] = pop[j] + deltat * change
            if pop[j, 5] or pop[j, 6] >= 0.5:
                pop[j, 5] = 0.5
                pop[j, 6] = 0.5
            if pop[j, 0] or pop[j, 4] <= 200:
                pop[j, 0] = 200
                pop[j, 4] = 200
            # 对历史诱导值Nold，历史捕食移动量Fold，个体适应度值进行更新
            Nold[j] = Nnew
            Fold[j] = Fnew
            pop_fitness[j] = function(vocab_wai=vocab_wai, w2v_model_wai=w2v_model_wai, maxLen_wai=maxLen_wai, trainSeq_wai=trainSeq_wai, trainCate_wai=trainCate_wai, testSeq_wai=testSeq_wai, testCate_wai=testCate_wai, conv1_op=int(pop[j, 0]), conv2_op=int(pop[j, 1]), maxpool_op=int(pop[j, 2]), stride=int(pop[j, 3]), qlj_op=int(pop[j, 4]), fznh_1=pop[j, 5], fznh_2=pop[j, 6], epoch=int(pop[j, 7]), batch=int(pop[j, 8]))

        # 对当前种群适应度最优个体适应度值、位置信息与适应度最差个体适应度值进行更新
        bestfitness = pop_fitness.min()
        worstfitness = pop_fitness.max()
        bestplace = pop[pop_fitness.argmin()].copy()

        # 6、对每个个体进行交叉操作：
        for j in range(m):
            # 计算交叉概率
            Cr = 0.2 * (1 - (worstfitness - pop_fitness[j]) / (worstfitness - bestfitness))
            for a in range(dimen):
                # 进行交叉操作
                r1 = np.random.rand()
                if r1 < Cr:
                    rangem = list(range(0, m))
                    rangem.pop(j)
                    b = np.random.choice(rangem)
                    pop[j][a] = pop[b][a]
        # 7、对每个个体进行变异操作：
        for j in range(m):
            # 计算变异概率Mu
            Mu = 0.05 * (1 - (worstfitness - pop_fitness[j]) / (worstfitness - bestfitness))
            for a in range(dimen):
                # 进行变异操作
                r2 = np.random.rand()
                if r2 < Mu:
                    r3 = np.random.rand()
                    rangem = list(range(0, m))
                    rangem.remove(j)
                    c = np.random.choice(rangem)
                    rangem.remove(c)
                    d = np.random.choice(rangem)
                    pop[j][a] = bestplace[a] + r3 * (pop[c][a] - pop[d][a])

        # 计算交叉、变异后的个体适应度值，对个体历史最优适应度值与位置信息进行更新
        for j in range(m):
            pop_fitness[j] = function(vocab_wai=vocab_wai, w2v_model_wai=w2v_model_wai, maxLen_wai=maxLen_wai, trainSeq_wai=trainSeq_wai, trainCate_wai=trainCate_wai, testSeq_wai=testSeq_wai, testCate_wai=testCate_wai, conv1_op=int(pop[j, 0]), conv2_op=int(pop[j, 1]), maxpool_op=int(pop[j, 2]), stride=int(pop[j, 3]), qlj_op=int(pop[j, 4]), fznh_1=pop[j, 5], fznh_2=pop[j, 6], epoch=int(pop[j, 7]), batch=int(pop[j, 8]))
            if pop_fitness[j] < bestfit[j]:
                bestfit[j] = pop_fitness[j]
                bestpop[j] = pop[j]
        # 对种群历史最优位置信息与适应度值进行更新
        if bestfit.min() < allbestfit:
            allbestfit = bestfit.min()
            allbestpop = bestpop[bestfit.argmin()].copy()
        # 对当前种群适应度最优个体适应度值、位置信息与适应度最差个体适应度值进行更新
        bestfitness = pop_fitness.min()
        worstfitness = pop_fitness.max()
        bestplace = pop[pop_fitness.argmin()].copy()

        # 存储当前迭代下的种群历史最优适应度值并输出
        his_bestfit[i] = allbestfit
        print("The best fitness is:", allbestfit)
    print("After iteration, the best pop is:", allbestpop)
    print("After iteration, the best fitness is:", allbestfit)

# #将结果进行绘图
# fig=plt.figure(figsize=(12, 10), dpi=300)
# plt.rcParams['font.sans-serif']=['Arial Unicode MS']
# plt.title('最优适应度值的变化情况',fontdict={'weight':'normal','size': 30})
# x=range(1,101,1)
# plt.plot(x,his_bestfit,color="red",linewidth=3.0, linestyle="-")
# plt.tick_params(labelsize=25)
# plt.xlim(0,101)
# plt.yscale("log")
# plt.xlabel("迭代次数",fontdict={'weight':'normal','size': 30})
# plt.ylabel("适应度值",fontdict={'weight':'normal','size': 30})
# plt.xticks(range(0,101,10))
# plt.savefig("KHA.png")
# plt.show()

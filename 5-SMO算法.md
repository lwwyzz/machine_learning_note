# SMO算法

        SMO算法要解如下凸二次规划的对偶问题：

$$
\min \limits_{\alpha}\frac{1}{2}\displaystyle\sum_{i=1}^N\sum_{j=1}^N
\alpha_i\alpha_jy_iy_jx_i^tx_j-\displaystyle\sum_{i=1}^N\alpha_i \\
s.t. \ \ \displaystyle\sum_{i=1}^N\alpha_iy_i=0, \\
0 \leq \alpha_i  \leq C \ (i \in [N])   \tag{5.1}
$$

在这个问题中，变量是拉格朗日乘子，一个变量$\alpha_i$对应于一个样本点$(x_i,y_i)$；变量的总数等于训练样本容量N。

        SMO算法是一种启发式算法，其基本思路是：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。SMO算法包括两个部分：求解两个变量二次规划的解析方法和选择变量的启发式方法。

## 两个变量二次规划的求解方法

不失一般性，假设选择的两个变量是$\alpha_1,\alpha_2$，其他变量$\alpha_i(i=3,4,...,N)$是固定的。于是SMO的最优化问题5.1的子问题可以写成：

$$
\min_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=
\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2
+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2) \\
+y_1\alpha_1\displaystyle\sum_{i=3}^Ny_i\alpha_iK_{i1}
+y_2\alpha_2\displaystyle\sum_{i=3}^Ny_i\alpha_iK_{i2} \tag{5.2}
$$

$$
s.t. \ \ \ \ \ \ \  \alpha_1y_1+\alpha_2y_2=
-\displaystyle\sum_{i=3}^Ny_i\alpha_i=\zeta \tag{5.3}
$$

$$
0 \leq \alpha_i \leq C, \ \ i=1,2   \tag{5.4}
$$

其中，$K_{ij}=K(x_i,x_j),i,j=1,2,...,N, \zeta$是常数，目标函数5.2省略了不含$\alpha_1,\alpha_2$的常数项。

    根据式5.3，将$\alpha_1$用$\alpha_2$表示，最终得到只有$\alpha_2$的目标函数，然后对$\alpha_2$求偏导，并置为零，最终得沿着约束方向未经剪辑时的解是：

$$
\alpha_2^{new,unc}=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{\eta} \\
其中，\ \ E_i=(\displaystyle\sum_{j=1}^N\alpha_jy_jK(x_j,x_i)+b)-y_i,i=1,2 \\
\eta=K_{11}+K_{22}-2K_{12}=\|\Phi(x_1)-\Phi(x_1)\|^2 \tag{5.5}
$$

由于不等式约束5.4的存在，再加上等式5.3的约束，最终可得$\alpha_2^{new}$的取值范围可以表示为：

$$
\left\{ 
\begin{matrix}
L=\max(0,\alpha_2^{old}-\alpha_1^{old}), 
H=\min(C,C+\alpha_2^{old}-\alpha_1^{old}), \ if \ y_1 \neq y_2\\
L=\max(0,\alpha_2^{old}+\alpha_1^{old}-C), 
H=\min(C,\alpha_2^{old}+\alpha_1^{old}), \ if \ y_1 = y_2\\
\end{matrix}
\right. \tag{5.6}
$$

所以经剪辑后$\alpha_2$的解是：

$$
\alpha_2^{new}=\left\{ 
\begin{matrix}
H, \ \ \ \alpha_2^{new,unc}>H \\
\alpha_2^{new,unc}, \ \ \ L \leq \alpha_2^{new,unc} \leq H \\
L, \ \\ \ \alpha_2^{new,unc} < L
\end{matrix}
\right. \tag{5.7}
$$

由$\alpha_2^{new}$求得$\alpha_1^{new}$是：$\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_1^{old})$ 

## 变量的选择方法

        SMO算法在每个子问题中选择两个变量优化，其中至少一个变量是违反KKT条件的。

### 第一个变量的选择

SMO称选第一个变$$量的过程为外层循环。外层循环在训练样本中选择违反KKT条件最严重的样本点，并将其对应的变量作为第一个变量。具体的，检验训练样本点$(x_i,y_i)$是否满足KKT条件，即：

![5-1 KKT条件](images/5-1%20KKT条件.png)

### 第2个变量的选择

        SMO称选择第二个变量的过程为内层循环。假设在外层循环中已经找到第一个变量$\alpha_1$，现在要在内层循环中找到第二个变量$\alpha_2$。第二个变量选择的标准是希望能使$\alpha_2$有足够大的变化。由式5.5，5.7可知，$\alpha_2^{new}$依赖于$|E_1-E_2|$的，为了加快速度，选择最大的$|E_1-E_2|$对应的α2​即可。如果能使目标函数有足够下降的$\alpha_2$，则更换$\alpha_2$。

### SMO算法：

![5-2 SMO算法](images/5-2%20SMO算法.png)

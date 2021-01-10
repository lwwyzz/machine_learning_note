前向传播

$x：shape=(?, 784), \ y：shape=(?,10), \ w：shape=(784, 10), \ b：shape=(10,)$

$z = xw + b shape(?,10)$

$$
z_i = \sum_jx_j w_{ji}+b_i
$$

$a = softmax(z) ：shape=(?,10)$

$$
a_i = softmax(z)=\frac{e^{z_i}}{\sum_je^{z_j}}  
$$

一个样本的损失：$loss = -\displaystyle\sum_iy_iloga_i$，其中$y_i$为真实值，如果为第$i$个类别则为1，否则为0。

损失函数对$z$求导：

$$
\frac{\partial{loss}}{\partial{z_i}}
=-(\frac{\partial{(y_i*loga_i)}}
{\partial{a_i}}\dot \frac{\partial{a_i}}{\partial{z_i}}+
\sum_{j\ne i}\frac{\partial{(y_j*loga_j)}}{\partial{a_j}}\dot 
\frac{\partial{a_j}}{\partial{z_i}})  \tag{1}

$$

softmax函数对z求导：

$$
\frac{\partial{a_i}}{\partial{z_i}}
=\frac{e^{z_i}}{\sum_je^{z_j}}-{\frac{e^{z_i}}{\sum_je^{z_j}}}^2
=a_i*(1-a_i) \\
\frac{\partial{a_i}}{\partial{z_j}}
=-\frac{e^{z_i}}{{\sum_je^{z_j}}^2}*e^{z_j}
=a_i*a_j,\ j \ne i  
 \tag{2}
$$

将式2代入式1得：

$$
\frac{\partial{loss}}{\partial{z_i}}
=-(\frac{y_i}{ai}*a_i*(1-a_i)+\sum_{j \ne i}\frac{y_j}{a_j}*a_i*a_j)  \\
=-(y_i-y_i*a_i+\sum_{j\ne i}y_j*a_i)=a_i-y_i
$$

对$w_{ji},b_i$求导，

$$
\frac{\partial{loss}}{\partial{w_{ji}}}
=\frac{\partial{loss}}{\partial{z_i}}
*\frac{\partial{z_i}}{{\partial{w_{ji}}}}
=(a_i-y_i)*x_j \\
\frac{\partial{loss}}{\partial{b_i}}
=\frac{\partial{loss}}{\partial{z_i}}
*\frac{\partial{z_i}}{{\partial{b_i}}}
=(a_i-y_i)
$$

x很大，$z_i$可能值较大，当$z_i$为100时，造成softmax为inf/inf,导致$a_i$为nan
x归一化后，将w初始化为很小的值，这样避免$z_i$值过大。实际上，当i不是真实类别是，ai趋向于0，最终造成log以0为底数，此时loss会为nan

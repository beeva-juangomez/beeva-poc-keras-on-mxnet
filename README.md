# Keras on MXNet

### Table of contents
1. [Topic](#topic)
2. [Goal](#goal)
3. [Scenarios](#scenarios)
4. [What is Keras?](#what_is_keras)
5. [What is MXNet?](#what_is_mxnet)
6. [Project structure](#project_structure)
7. [Hardware](#hardware)
8. [Result](#result)
  1. [CNN LeNet MNIST](#result_lenet)
9. [Conclusion](#conclusion)

### Topic <a name="topic"></a>

To use MXNet as a backend with Keras

### Goal <a name="goal"></a>

Generate results and conclusions that describe the use of MXNet as a backend with Keras in the following cases:

* CPU
* GPU
* Multi-GPU

### Scenarios <a name="scenarios"></a>

* Convolutional NN with LeNet and MNIST

### What is Keras? <a name="what_is_keras"></a>

>Keras is a high-level neural networks API, written in Python and capable of
running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on
enabling fast experimentation. Being able to go from idea to result with the
least possible delay is key to doing good research.

The official description of Keras does not include MXNet. **Today is not possible
to use MXNet with Keras officially**. However, you can find a [**fork**](https://github.com/dmlc/keras)
on GitHub of Keras 1.2.2 supporting it.

### What is MXNet? <a name="what_is_mxnet"></a>

>MXNet is an modern open-source deep learning framework used to train,
and deploy deep neural networks. It is scalable, allowing for fast model training,
and supports a flexible programming model and multiple languages.

### Project structure <a name="project_structure"></a>

* **lenet-mnist**: Contains an implementation of LeNet architecture (CNN) with MNIST.

Moreover, in the project root exists the 'requirement' folder with three bash scripts containing the software needed to run both projects, depending on which hardware you have.

Inside lenet-mnist project you can find scripts that run the neural networks n times with different batch sizes and epochs.

### Hardware <a name="hardware"></a>

AWS EC2 instances:

* Infrastructure 0: AWS c4.4xlarge (16 vCPU). Ubuntu 16.04 LTS - Xenial (HVM), Keras 1.2.2 (fork), MXNet 0.10.0
* Infrastructure 1: AWS p2.8xlarge (8 GPUs Nvidia Tesla K80). Ubuntu 16.04 LTS - Xenial (HVM), NVIDIA Driver 375.66, CUDA 8.0, Keras 1.2.2 (fork), MXNet 0.10.0
* Infrastructure 2: AWS p2.8xlarge (8 GPUs Nvidia Tesla K80). Ubuntu 16.04 LTS - Xenial (HVM), NVIDIA Driver 375.66, CUDA 8.0, Keras 1.2.2 (fork), MXNet 0.10.0, Tensorflow 1.0.1, cuDNN libcudnn5

### Result <a name="result"></a>

The output is located inside 'output' folder. It contains a txt file with the following fields comma-separated:

* Batch size
* Epochs
* Loss
* Accuracy
* Time (seconds)

The final results have been moved to 'result' folder where you can find 'cpu', 'gpu' and 'multigpu' folders.

| infrastructure | model | script                      | batch_size | gpus | accuracy       | epochs | time               | samples_sec        | 
|----------------|-------|-----------------------------|------------|------|----------------|--------|--------------------|--------------------| 
| 0              | lenet | keras_mxnet_lenet           | 64         | 0    | 0.991633506785 | 12     | 255.7              | 235.0              | 
| 0              | lenet | keras_mxnet_lenet           | 128        | 0    | 0.988674624984 | 12     | 263.2              | 228.0              | 
| 0              | lenet | keras_mxnet_lenet           | 256        | 0    | 0.987552290528 | 12     | 232.4              | 258.0              | 
| 0              | lenet | keras_mxnet_lenet           | 512        | 0    | 0.986327925722 | 12     | 212.4              | 283.0              | 
| 2              | lenet | keras_mxnet_lenet           | 64         | 1    | 0.9892         | 12     | 5.570223689079285  | 10771.56023691349  | 
| 2              | lenet | keras_mxnet_lenet           | 128        | 1    | 0.9876         | 12     | 4.544280529022217  | 13203.410224524601 | 
| 2              | lenet | keras_mxnet_lenet           | 256        | 1    | 0.9832         | 12     | 3.754633903503418  | 15980.253079804797 | 
| 2              | lenet | keras_mxnet_lenet           | 512        | 1    | 0.975899999714 | 12     | 3.350304961204529  | 17908.817464314747 | 
| 2              | lenet | keras_mxnet_lenet           | 1024       | 1    | 0.974399998951 | 12     | 3.1241434812545776 | 19205.2638939315   | 
| 2              | lenet | keras_mxnet_lenet           | 2048       | 1    | 0.961499997711 | 12     | 3.0668808221817017 | 19563.85118262193  | 
| 2              | lenet | keras_mxnet_lenet           | 4096       | 1    | 0.936299995327 | 12     | 3.022548198699951  | 19850.80007187545  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 64         | 1    | 0.9907         | 12     | 5.554294943809509  | 10802.451185433081 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 128        | 1    | 0.988          | 12     | 4.575205564498901  | 13114.16485098883  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 256        | 1    | 0.9849         | 12     | 3.7631174325942993 | 15944.227379222631 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 512        | 1    | 0.973499999332 | 12     | 3.3812716007232666 | 17744.803460084597 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 1024       | 1    | 0.962899999142 | 12     | 3.129989743232727  | 19169.39189009309  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 2048       | 1    | 0.941199998188 | 12     | 3.063093066215515  | 19588.043426356173 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 4096       | 1    | 0.920900001144 | 12     | 3.0096840858459473 | 19935.647160501063 | 
| 2              | lenet | keras_mxnet_lenet           | 64         | 4    | 0.9898         | 12     | 5.279034614562988  | 11365.714449850591 | 
| 2              | lenet | keras_mxnet_lenet           | 128        | 4    | 0.9873         | 12     | 2.6917389631271362 | 22290.422965195263 | 
| 2              | lenet | keras_mxnet_lenet           | 256        | 4    | 0.9829         | 12     | 1.7480956315994263 | 34323.06500594752  | 
| 2              | lenet | keras_mxnet_lenet           | 512        | 4    | 0.969099999046 | 12     | 1.3011418581008911 | 46113.34238956409  | 
| 2              | lenet | keras_mxnet_lenet           | 1024       | 4    | 0.972500003624 | 12     | 1.1067509651184082 | 54212.73790674379  | 
| 2              | lenet | keras_mxnet_lenet           | 2048       | 4    | 0.954900003147 | 12     | 0.9431552886962891 | 63616.24720668979  | 
| 2              | lenet | keras_mxnet_lenet           | 4096       | 4    | 0.928300002956 | 12     | 1.3472495079040527 | 44535.18049031867  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 64         | 4    | 0.9893         | 12     | 5.258136868476868  | 11410.88592799987  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 128        | 4    | 0.9875         | 12     | 2.680631995201111  | 22382.781414014487 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 256        | 4    | 0.9836         | 12     | 1.7380064725875854 | 34522.31101917047  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 512        | 4    | 0.975799998951 | 12     | 1.2994126081466675 | 46174.70972948084  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 1024       | 4    | 0.956899999142 | 12     | 1.1052311658859253 | 54287.28563938524  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 2048       | 4    | 0.939000004292 | 12     | 0.9494647979736328 | 63193.49609174898  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 4096       | 4    | 0.908800001621 | 12     | 1.3228462934494019 | 45356.74348343704  | 
| 2              | lenet | keras_mxnet_lenet           | 64         | 8    | 0.9915         | 12     | 7.250983953475952  | 8274.739040242586  | 
| 2              | lenet | keras_mxnet_lenet           | 128        | 8    | 0.99           | 12     | 3.9483689069747925 | 15196.14843841213  | 
| 2              | lenet | keras_mxnet_lenet           | 256        | 8    | 0.9868         | 12     | 2.160717487335205  | 27768.55389549214  | 
| 2              | lenet | keras_mxnet_lenet           | 512        | 8    | 0.983299999714 | 12     | 1.2230634689331055 | 49057.143414101636 | 
| 2              | lenet | keras_mxnet_lenet           | 1024       | 8    | 0.978099998474 | 12     | 0.8028659820556641 | 74732.2733071534   | 
| 2              | lenet | keras_mxnet_lenet           | 2048       | 8    | 0.965099995518 | 12     | 0.8196406364440918 | 73202.81271107114  | 
| 2              | lenet | keras_mxnet_lenet           | 4096       | 8    | 0.950599990273 | 12     | 0.6276061534881592 | 95601.35710353898  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 64         | 8    | 0.9896         | 12     | 7.205681920051575  | 8326.762222605928  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 128        | 8    | 0.9902         | 12     | 3.933151364326477  | 15254.943032245736 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 256        | 8    | 0.9869         | 12     | 2.1634011268615723 | 27734.107769020855 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 512        | 8    | 0.983099999619 | 12     | 1.214330792427063  | 49409.930452376146 | 
| 2              | lenet | keras_mxnet_lenet_tanh | 1024       | 8    | 0.97290000267  | 12     | 0.8035461902618408 | 74669.01184666013  | 
| 2              | lenet | keras_mxnet_lenet_tanh | 2048       | 8    | 0.958999995899 | 12     | 0.801931619644165  | 74819.3468498266   | 
| 2              | lenet | keras_mxnet_lenet_tanh | 4096       | 8    | 0.939600005722 | 12     | 0.6795275211334229 | 88296.64455668045  | 
| 2              | lenet | keras_mxnet_official_sample | 64         | 1    | 0.992237       | 12     | 2.720              | 22058.823529411762 | 
| 2              | lenet | keras_mxnet_official_sample | 128        | 1    | 0.992385       | 12     | 2.154              | 27855.15320334262  | 
| 2              | lenet | keras_mxnet_official_sample | 256        | 1    | 0.990234       | 12     | 1.787              | 33575.82540570789  | 
| 2              | lenet | keras_mxnet_official_sample | 512        | 1    | 0.990625       | 12     | 1.615              | 37151.70278637771  | 
| 2              | lenet | keras_mxnet_official_sample | 1024       | 1    | 0.988574       | 12     | 1.537              | 39037.085230969424 | 
| 2              | lenet | keras_mxnet_official_sample | 2048       | 1    | 0.985645       | 12     | 1.514              | 39630.11889035667  | 
| 2              | lenet | keras_mxnet_official_sample | 4096       | 1    | 0.978760       | 12     | 1.446              | 41493.77593360996  | 
| 2              | lenet | keras_mxnet_official_sample | 64         | 4    | 0.992038       | 12     | 2.970              | 20202.0202020202   | 
| 2              | lenet | keras_mxnet_official_sample | 128        | 4    | 0.992089       | 12     | 1.368              | 43859.649122807015 | 
| 2              | lenet | keras_mxnet_official_sample | 256        | 4    | 0.992090       | 12     | 0.764              | 78534.03141361257  | 
| 2              | lenet | keras_mxnet_official_sample | 512        | 4    | 0.991797       | 12     | 0.568              | 105633.80281690141 | 
| 2              | lenet | keras_mxnet_official_sample | 1024       | 4    | 0.988965       | 12     | 0.463              | 129589.63282937364 | 
| 2              | lenet | keras_mxnet_official_sample | 2048       | 4    | 0.987598       | 12     | 0.418              | 143540.66985645934 | 
| 2              | lenet | keras_mxnet_official_sample | 4096       | 4    | 0.978841       | 12     | 0.392              | 153061.22448979592 | 
| 2              | lenet | keras_mxnet_official_sample | 64         | 8    | 0.992138       | 12     | 5.123              | 11711.887565879368 | 
| 2              | lenet | keras_mxnet_official_sample | 128        | 8    | 0.992385       | 12     | 2.631              | 22805.017103762828 | 
| 2              | lenet | keras_mxnet_official_sample | 256        | 8    | 0.991602       | 12     | 1.295              | 46332.04633204633  | 
| 2              | lenet | keras_mxnet_official_sample | 512        | 8    | 0.990625       | 12     | 0.676              | 88757.39644970413  | 
| 2              | lenet | keras_mxnet_official_sample | 1024       | 8    | 0.989551       | 12     | 0.372              | 161290.32258064515 | 
| 2              | lenet | keras_mxnet_official_sample | 2048       | 8    | 0.986914       | 12     | 0.250              | 240000.0           | 
| 2              | lenet | keras_mxnet_official_sample | 4096       | 8    | 0.978353       | 12     | 0.216              | 277777.77777777775 | 
| 2              | lenet | keras_tensorflow_lenet      | 64         | 1    | 0.9904         | 12     | 18.77412700653076  | 3195.8876159263446 | 
| 2              | lenet | keras_tensorflow_lenet      | 128        | 1    | 0.986          | 12     | 16.5407977104187   | 3627.3945821976445 | 
| 2              | lenet | keras_tensorflow_lenet      | 256        | 1    | 0.9833         | 12     | 16.199799060821533 | 3703.749643729053  | 
| 2              | lenet | keras_tensorflow_lenet      | 512        | 1    | 0.978299999523 | 12     | 17.56779968738556  | 3415.339488591881  | 
| 2              | lenet | keras_tensorflow_lenet      | 1024       | 1    | 0.96280000124  | 12     | 17.76717460155487  | 3377.0141480316843 | 
| 2              | lenet | keras_tensorflow_lenet      | 2048       | 1    | 0.936800004673 | 12     | 18.004580974578857 | 3332.4852205511243 | 
| 2              | lenet | keras_tensorflow_lenet      | 4096       | 1    | 0.922100008678 | 12     | 18.467231035232544 | 3248.9981787485913 | 
| 2              | lenet | keras_tensorflow_lenet      | 64         | 4    | 0.9818         | 12     | 11.567815780639648 | 5186.8045910982055 | 
| 2              | lenet | keras_tensorflow_lenet      | 128        | 4    | 0.9757         | 12     | 7.137097477912903  | 8406.778832106656  | 
| 2              | lenet | keras_tensorflow_lenet      | 256        | 4    | 0.9598         | 12     | 5.472914218902588  | 10963.080655050175 | 
| 2              | lenet | keras_tensorflow_lenet      | 512        | 4    | 0.934699999237 | 12     | 4.652027726173401  | 12897.601547477008 | 
| 2              | lenet | keras_tensorflow_lenet      | 1024       | 4    | 0.916200001335 | 12     | 4.517949342727661  | 13280.361387092418 | 
| 2              | lenet | keras_tensorflow_lenet      | 2048       | 4    | 0.885999994183 | 12     | 5.285510897636414  | 11351.788154827367 | 
| 2              | lenet | keras_tensorflow_lenet      | 4096       | 4    | 0.825099996758 | 12     | 6.025966167449951  | 9956.909536614708  | 
| 2              | lenet | keras_tensorflow_lenet      | 64         | 8    | 0.9829         | 12     | 14.296846389770508 | 4196.7297097722485 | 
| 2              | lenet | keras_tensorflow_lenet      | 128        | 8    | 0.9774         | 12     | 8.327619791030884  | 7204.939887460033  | 
| 2              | lenet | keras_tensorflow_lenet      | 256        | 8    | 0.9578         | 12     | 4.893328785896301  | 12261.591776324902 | 
| 2              | lenet | keras_tensorflow_lenet      | 512        | 8    | 0.937499998283 | 12     | 3.0603524446487427 | 19605.58500538542  | 
| 2              | lenet | keras_tensorflow_lenet      | 1024       | 8    | 0.912699993515 | 12     | 2.736750841140747  | 21923.808005477942 | 
| 2              | lenet | keras_tensorflow_lenet      | 2048       | 8    | 0.891500013924 | 12     | 2.6365342140197754 | 22757.148259616693 | 
| 2              | lenet | keras_tensorflow_lenet      | 4096       | 8    | 0.822699994946 | 12     | 3.0443185567855835 | 19708.844157016352 | 
| 1              | lenet | keras_mxnet_lenet           | 64         | 1    | 0.9893         | 12     | 5.569332003593445  | 10773.284832236037 | 
| 1              | lenet | keras_mxnet_lenet           | 128        | 1    | 0.9881         | 12     | 4.566421866416931  | 13139.390480161514 | 
| 1              | lenet | keras_mxnet_lenet           | 256        | 1    | 0.9859         | 12     | 3.762406587600708  | 15947.239779383355 | 
| 1              | lenet | keras_mxnet_lenet           | 512        | 1    | 0.977299999809 | 12     | 3.352225422859192  | 17898.55765392549  | 
| 1              | lenet | keras_mxnet_lenet           | 1024       | 1    | 0.971999998379 | 12     | 3.114904284477234  | 19262.22911535455  | 
| 1              | lenet | keras_mxnet_lenet           | 2048       | 1    | 0.958700000572 | 12     | 3.076540946960449  | 19502.422049437893 | 
| 1              | lenet | keras_mxnet_lenet           | 4096       | 1    | 0.935599994659 | 12     | 3.028161644935608  | 19814.00170639697  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 64         | 1    | 0.9895         | 12     | 5.561059594154358  | 10789.310739102752 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 128        | 1    | 0.9902         | 12     | 4.576619505882263  | 13110.113244695754 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 256        | 1    | 0.9853         | 12     | 3.7619993686676025 | 15948.96599391253  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 512        | 1    | 0.978499999523 | 12     | 3.3807499408721924 | 17747.541536455883 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 1024       | 1    | 0.964000000858 | 12     | 3.12713086605072   | 19186.91688006473  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 2048       | 1    | 0.947699999237 | 12     | 3.0674573183059692 | 19560.17436393721  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 4096       | 1    | 0.916499995804 | 12     | 3.0113775730133057 | 19924.43609120778  | 
| 1              | lenet | keras_mxnet_lenet           | 64         | 4    | 0.9873         | 12     | 5.258297920227051  | 11410.536434080408 | 
| 1              | lenet | keras_mxnet_lenet           | 128        | 4    | 0.9876         | 12     | 2.699516177177429  | 22226.205016757867 | 
| 1              | lenet | keras_mxnet_lenet           | 256        | 4    | 0.9495         | 12     | 1.7466647624969482 | 34351.18248691688  | 
| 1              | lenet | keras_mxnet_lenet           | 512        | 4    | 0.969199999714 | 12     | 1.3023080825805664 | 46072.04762263935  | 
| 1              | lenet | keras_mxnet_lenet           | 1024       | 4    | 0.973199996567 | 12     | 1.1029914617538452 | 54397.51990880797  | 
| 1              | lenet | keras_mxnet_lenet           | 2048       | 4    | 0.948600003529 | 12     | 0.943806529045105  | 63572.35106299267  | 
| 1              | lenet | keras_mxnet_lenet           | 4096       | 4    | 0.912700003719 | 12     | 1.3126788139343262 | 45708.05848550994  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 64         | 4    | 0.9899         | 12     | 5.219395160675049  | 11495.584862411513 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 128        | 4    | 0.986          | 12     | 2.6854575872421265 | 22342.56101643294  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 256        | 4    | 0.982          | 12     | 1.7449589967727661 | 34384.762112443714 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 512        | 4    | 0.975999999332 | 12     | 1.3129220008850098 | 45699.59217650052  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 1024       | 4    | 0.957399999046 | 12     | 1.0990432500839233 | 54592.937989854705 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 2048       | 4    | 0.932499996758 | 12     | 0.9452526569366455 | 63475.092674636544 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 4096       | 4    | 0.911199993324 | 12     | 1.3139889240264893 | 45662.48535500626  | 
| 1              | lenet | keras_mxnet_lenet           | 64         | 8    | 0.991          | 12     | 7.22563111782074   | 8303.772919160047  | 
| 1              | lenet | keras_mxnet_lenet           | 128        | 8    | 0.9889         | 12     | 3.9757736921310425 | 15091.402239205316 | 
| 1              | lenet | keras_mxnet_lenet           | 256        | 8    | 0.9854         | 12     | 2.1688480377197266 | 27664.455488122865 | 
| 1              | lenet | keras_mxnet_lenet           | 512        | 8    | 0.975999999142 | 12     | 1.2249841690063477 | 48980.224820921    | 
| 1              | lenet | keras_mxnet_lenet           | 1024       | 8    | 0.978400002861 | 12     | 0.8054009675979614 | 74497.05477626232  | 
| 1              | lenet | keras_mxnet_lenet           | 2048       | 8    | 0.969499995518 | 12     | 0.7947095632553101 | 75499.28020776098  | 
| 1              | lenet | keras_mxnet_lenet           | 4096       | 8    | 0.945800000668 | 12     | 0.6301528215408325 | 95214.99856699782  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 64         | 8    | 0.9906         | 12     | 7.195716381072998  | 8338.294177049405  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 128        | 8    | 0.9896         | 12     | 3.937455654144287  | 15238.266858154517 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 256        | 8    | 0.9851         | 12     | 2.1681896448135376 | 27672.856082273167 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 512        | 8    | 0.983199999619 | 12     | 1.2619986534118652 | 47543.632346823455 | 
| 1              | lenet | keras_mxnet_lenet_tanh | 1024       | 8    | 0.973200000763 | 12     | 0.80363929271698   | 74660.3613633043   | 
| 1              | lenet | keras_mxnet_lenet_tanh | 2048       | 8    | 0.95989999876  | 12     | 0.8050094842910767 | 74533.28335980834  | 
| 1              | lenet | keras_mxnet_lenet_tanh | 4096       | 8    | 0.933400000381 | 12     | 0.7097641229629517 | 84535.12661294643  | 
| 1              | lenet | keras_mxnet_official_sample | 64         | 1    | 0.991640       | 12     | 2.745              | 21857.92349726776  | 
| 1              | lenet | keras_mxnet_official_sample | 128        | 1    | 0.992089       | 12     | 2.126              | 28222.013170272814 | 
| 1              | lenet | keras_mxnet_official_sample | 256        | 1    | 0.991016       | 12     | 1.781              | 33688.93879842785  | 
| 1              | lenet | keras_mxnet_official_sample | 512        | 1    | 0.990918       | 12     | 1.614              | 37174.72118959107  | 
| 1              | lenet | keras_mxnet_official_sample | 1024       | 1    | 0.989453       | 12     | 1.539              | 38986.35477582846  | 
| 1              | lenet | keras_mxnet_official_sample | 2048       | 1    | 0.987891       | 12     | 1.520              | 39473.68421052631  | 
| 1              | lenet | keras_mxnet_official_sample | 4096       | 1    | 0.980062       | 12     | 1.444              | 41551.246537396124 | 
| 1              | lenet | keras_mxnet_official_sample | 64         | 4    | 0.992536       | 12     | 2.798              | 21443.888491779842 | 
| 1              | lenet | keras_mxnet_official_sample | 128        | 4    | 0.991792       | 12     | 1.537              | 39037.085230969424 | 
| 1              | lenet | keras_mxnet_official_sample | 256        | 4    | 0.991113       | 12     | 0.762              | 78740.15748031496  | 
| 1              | lenet | keras_mxnet_official_sample | 512        | 4    | 0.990723       | 12     | 0.576              | 104166.66666666667 | 
| 1              | lenet | keras_mxnet_official_sample | 1024       | 4    | 0.989844       | 12     | 0.464              | 129310.3448275862  | 
| 1              | lenet | keras_mxnet_official_sample | 2048       | 4    | 0.987402       | 12     | 0.418              | 143540.66985645934 | 
| 1              | lenet | keras_mxnet_official_sample | 4096       | 4    | 0.980876       | 12     | 0.392              | 153061.22448979592 | 
| 1              | lenet | keras_mxnet_official_sample | 64         | 8    | 0.992337       | 12     | 5.268              | 11389.521640091116 | 
| 1              | lenet | keras_mxnet_official_sample | 128        | 8    | 0.991891       | 12     | 2.594              | 23130.300693909023 | 
| 1              | lenet | keras_mxnet_official_sample | 256        | 8    | 0.991602       | 12     | 1.310              | 45801.52671755725  | 
| 1              | lenet | keras_mxnet_official_sample | 512        | 8    | 0.991406       | 12     | 0.675              | 88888.88888888888  | 
| 1              | lenet | keras_mxnet_official_sample | 1024       | 8    | 0.990234       | 12     | 0.355              | 169014.08450704225 | 
| 1              | lenet | keras_mxnet_official_sample | 2048       | 8    | 0.987305       | 12     | 0.251              | 239043.8247011952  | 
| 1              | lenet | keras_mxnet_official_sample | 4096       | 8    | 0.979818       | 12     | 0.217              | 276497.69585253455 | 



#### CNN LeNet MNIST  <a name="result_lenet"></a>

With MXNet and cuDNN enabled:

- x3.25 throughput with 4 GPUs over 1 GPU
- x5 throughput with 8 GPUs over 1 GPU

With the official MXNet sample we have different results:

- x3.68 throughput with 4 GPUs over 1 GPU
- x6.7 throughput with 8 GPUs over 1 GPU

### Conclusion <a name="conclusion"></a>

- Nowadays, Keras is not ready to support MXNet for professional development
- A forked repository is not enough to long-term development
- You have to stick in 1.2.2 if you want to use Keras on MXNet
- Tensorflow is trendy right now but there are alternatives showing better performance
- mxnet-cu80, mxnet-cu75… have CUDA and cuDNN flags enabled. Install this versions if you are going to execute in multi-GPU environment
- With MXNet is complicated to know if cuDNN is executing
- cuDNN is worth it when you have a big network
- MXNet shows better performance than Tensorflow (5-10%)
- Tensorflow can not execute without cuDNN. It is a must to implement a “make_parallel” function (available on GitHub) to execute Tensorflow in multi-GPU

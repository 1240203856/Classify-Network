1、使用网络前先修改参数及保存数据的路径、model=resnet18()在网络定义部分修改
2、根目录下应包含：
     .py文件（代码）
      data文件夹：其中又包含 train_data、test_data、val_data 三个文件夹
                         数据存放形式如下：（root即上述三个文件夹）
                                                    root/dog/xxx.png
                                                    root/dog/xxy.png
                                                    root/dog/xxz.png
                                                    root/cat/123.png
                                                    root/cat/nsd.png
                                                    root/cat/932.png
        model文件夹：存放训练好的模型  （该文件夹可自动生成）
        result.txt：存放测试结果               （该文件可自动生成）
        label.txt：存放标签值                    （该文件可自动生成）
        prob.txt：存放softmax后得到的概率值  （该文件可自动生成）
3、评估标准：
     二分类：准确率、精确率、召回率、ROC曲线与AUC、F1-score
     多分类：准确率、精确率、召回率、F1-score
                  每个类别的精确率、召回率以及ROC曲线与AUC
                  
1��ʹ������ǰ���޸Ĳ������������ݵ�·����model=resnet18()�����綨�岿���޸�
2����Ŀ¼��Ӧ������
     .py�ļ������룩
      data�ļ��У������ְ��� train_data��test_data��val_data �����ļ���
                         ���ݴ����ʽ���£���root�����������ļ��У�
                                                    root/dog/xxx.png
                                                    root/dog/xxy.png
                                                    root/dog/xxz.png
                                                    root/cat/123.png
                                                    root/cat/nsd.png
                                                    root/cat/932.png
        model�ļ��У����ѵ���õ�ģ��  �����ļ��п��Զ����ɣ�
        result.txt����Ų��Խ��               �����ļ����Զ����ɣ�
        label.txt����ű�ǩֵ                    �����ļ����Զ����ɣ�
        prob.txt�����softmax��õ��ĸ���ֵ  �����ļ����Զ����ɣ�
3��������׼��
     �����ࣺ׼ȷ�ʡ���ȷ�ʡ��ٻ��ʡ�ROC������AUC��F1-score
     ����ࣺ׼ȷ�ʡ���ȷ�ʡ��ٻ��ʡ�F1-score
                  ÿ�����ľ�ȷ�ʡ��ٻ����Լ�ROC������AUC
                  